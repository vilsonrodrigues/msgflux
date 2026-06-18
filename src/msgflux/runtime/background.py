from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError
from functools import partial
from threading import Lock
from typing import Any, Callable, Dict, Mapping
from uuid import uuid4

from msgflux._private.executor import Executor
from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import TaskPauseRequestedError, TaskStopRequestedError
from msgflux.logger import logger
from msgflux.runtime.agent_inbox import AgentInbox, AgentNotification
from msgflux.runtime.context import (
    DEFAULT_SESSION_ID,
    ExecutionScope,
    execution_context,
    get_execution_context,
)
from msgflux.tasks import TaskActivityRecorder, TaskHandle


class BackgroundTaskDispatcher:
    """Dispatches tools into the task runtime and tracks live executions."""

    def __init__(self, library: Any, *, tool_call_factory: Callable[..., Any]):
        self.library = library
        self._tool_call_factory = tool_call_factory
        self._task_futures: Dict[str, Any] = {}
        self._task_futures_lock = Lock()
        self._task_inboxes: Dict[str, AgentInbox] = {}
        self._task_inboxes_lock = Lock()

    def clear(self) -> None:
        with self._task_futures_lock:
            self._task_futures.clear()
        with self._task_inboxes_lock:
            self._task_inboxes.clear()

    def register_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            self._task_futures[task_id] = future

    def get_task_future(self, task_id: str) -> Any | None:
        with self._task_futures_lock:
            return self._task_futures.get(task_id)

    def cleanup_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            current = self._task_futures.get(task_id)
            if current is future:
                self._task_futures.pop(task_id, None)

    def register_task_inbox(self, task_id: str, inbox: AgentInbox) -> None:
        with self._task_inboxes_lock:
            self._task_inboxes[task_id] = inbox

    def get_task_inbox(self, task_id: str) -> AgentInbox | None:
        with self._task_inboxes_lock:
            return self._task_inboxes.get(task_id)

    def run_tool(
        self,
        *,
        tool: Any,
        task_handle: TaskHandle,
        tool_name: str,
        call_params: Dict[str, Any],
        execution_scope: Dict[str, Any] | None = None,
        agent_inbox: AgentInbox | None = None,
    ) -> Any:
        scope = execution_scope or {}
        with execution_context(**scope):
            task_handle.set_running()
            try:
                result = tool(**call_params)
            except TaskStopRequestedError as exc:
                task_handle.stop(reason=str(exc))
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="stopped",
                    hint=(
                        f"Use task_status(task_id='{task_handle.task_id}') "
                        "if you need stop details."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            except TaskPauseRequestedError as exc:
                task_handle.pause(reason=str(exc))
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="paused",
                    hint=(
                        f"Use task_message(task_id='{task_handle.task_id}', "
                        "message='...') to resume the paused task."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            except Exception as exc:
                task_handle.fail(exc)
                self.publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="failed",
                    hint=(
                        f"Use task_status(task_id='{task_handle.task_id}') "
                        "if you need error details."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            task_handle.complete(result)
            self.publish_task_notification(
                task_id=task_handle.task_id,
                tool_name=tool_name,
                status="completed",
                hint=(
                    f"Use task_output(task_id='{task_handle.task_id}') "
                    "if you need the result."
                ),
                agent_inbox=agent_inbox,
            )
            return result

    def resume_agent_task(self, *, task: Any, message: str) -> str:
        tool_name = task.tool_name
        if tool_name not in self.library.library:
            raise ValueError(f"The tool `{tool_name}` is no longer available.")
        tool = self.library.library[tool_name]
        checkpoint_namespace = (
            tool.impl.get_module_name()
            if hasattr(tool, "impl") and hasattr(tool.impl, "get_module_name")
            else tool.get_module_name()
        )

        checkpoint_store = get_execution_context().get("checkpoint_store")
        session_id = task.metadata.get("checkpoint_session_id") or task.metadata.get(
            "session_id"
        )
        run_id = task.metadata.get("checkpoint_run_id") or task.task_id
        restored_messages = ChatMessages()
        restored_vars: Dict[str, Any] = {}
        restored_model_preference = None

        if checkpoint_store is not None and isinstance(session_id, str) and session_id:
            state = checkpoint_store.load_state(
                checkpoint_namespace,
                session_id,
                run_id,
            )
            if state is not None:
                restored_messages._hydrate_state(state.get("messages", {}))
                restored_vars = state.get("vars", {}) or {}
                restored_model_preference = state.get("model_preference")

        root_inbox = (
            get_execution_context().get("agent_inbox") or self.library.agent_inbox
        )
        task_inbox = self.get_task_inbox(task.task_id)
        if task_inbox is None:
            task_inbox = root_inbox.fork(
                owner=f"{tool_name}:{task.task_id}",
                namespace=checkpoint_namespace,
                session_id=(
                    session_id if isinstance(session_id, str) and session_id else None
                ),
                run_id=run_id,
            )
            self.register_task_inbox(task.task_id, task_inbox)

        self.library.task_store.requeue(task.task_id)
        self.library.task_store.add_activity(
            task.task_id,
            kind="message",
            summary=f"Root message: {self.library._truncate_activity_text(message)}",
            metadata={"direction": "root_to_task", "resume": True},
        )

        execution_scope = {
            "session_id": session_id
            if isinstance(session_id, str) and session_id
            else None,
            "run_id": run_id,
            "parent_run_id": task.metadata.get("parent_run_id"),
            "root_run_id": task.metadata.get("root_run_id"),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox,
            "task_activity_recorder": TaskActivityRecorder(
                task.task_id, self.library.task_store
            ),
        }
        future = Executor.get_instance().submit(
            partial(
                self.run_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    self.library.task_store,
                    tool_name=tool_name,
                    agent_inbox=root_inbox,
                ),
                tool_name=tool_name,
                call_params={
                    "messages": restored_messages,
                    "scope": ExecutionScope(
                        session_id=(
                            session_id
                            if isinstance(session_id, str)
                            else DEFAULT_SESSION_ID
                        ),
                        namespace=checkpoint_namespace,
                        run_id=run_id,
                        parent_run_id=task.metadata.get("parent_run_id"),
                        root_run_id=task.metadata.get("root_run_id"),
                    ),
                    "model_preference": restored_model_preference,
                    "vars": restored_vars,
                    "tool_call_id": f"task_message_{task.task_id}",
                    "task": message,
                },
                execution_scope=execution_scope,
                agent_inbox=root_inbox,
            )
        )
        self.register_task_future(task.task_id, future)
        future.add_done_callback(partial(self.cleanup_task_future, task.task_id))
        future.add_done_callback(self.log_task_failure)
        return "Message scheduled and background agent resumed."

    def log_task_failure(self, future: Any) -> None:
        try:
            future.result()
        except FutureCancelledError:
            return
        except TaskStopRequestedError:
            return
        except TaskPauseRequestedError:
            return
        except Exception as exc:
            logger.error(f"Background task error: {exc!s}", exc_info=True)

    def dispatch(
        self,
        *,
        tool: Any,
        tool_id: str,
        tool_name: str,
        call_params: Dict[str, Any],
        config: Mapping[str, Any],
    ) -> Any:
        task_kind = config.get("tool_kind", "tool")
        context = get_execution_context()
        session_id = context.get("session_id")
        parent_run_id = context.get("run_id")
        root_run_id = context.get("root_run_id")
        checkpoint_store = context.get("checkpoint_store")
        root_agent_inbox = context.get("agent_inbox") or self.library.agent_inbox
        task_id = uuid4().hex[:8]
        task_inbox = None
        if task_kind == "agent":
            task_inbox = root_agent_inbox.fork(
                owner=f"{tool_name}:{task_id}",
                namespace=tool_name,
                session_id=session_id if isinstance(session_id, str) else None,
                run_id=task_id,
            )
            self.register_task_inbox(task_id, task_inbox)
        task = self.library.task_store.create(
            task_id=task_id,
            tool_name=tool_name,
            metadata={
                "tool_call_id": tool_id,
                "task_kind": task_kind,
                "session_id": session_id,
                "parent_run_id": parent_run_id,
                "root_run_id": root_run_id,
                "checkpoint_session_id": session_id,
                "checkpoint_run_id": task_id if task_kind == "agent" else None,
                "supports_activity": task_kind == "agent",
                "supports_message": task_kind == "agent",
                "stop_requested": False,
            },
        )
        runner_params = dict(call_params)
        if config.get("inject_task", False) and task_kind != "agent":
            runner_params["task"] = TaskHandle(
                task.task_id,
                self.library.task_store,
                tool_name=tool_name,
                agent_inbox=root_agent_inbox,
            )
        if config.get("inject_notification", False) and task_kind != "agent":
            runner_params["notification"] = self.library._build_notification_handle(
                tool_name=tool_name,
                ref=task.task_id,
                agent_inbox=root_agent_inbox,
            )
        if task_kind == "agent":
            runner_params["scope"] = ExecutionScope(
                session_id=(
                    session_id if isinstance(session_id, str) else DEFAULT_SESSION_ID
                ),
                namespace=tool_name,
                run_id=task.task_id,
                parent_run_id=(
                    parent_run_id
                    if isinstance(parent_run_id, str) and parent_run_id
                    else None
                ),
                root_run_id=(
                    root_run_id
                    if isinstance(root_run_id, str) and root_run_id
                    else task.task_id
                ),
            )
        runner_params["tool_call_id"] = tool_id
        execution_scope = {
            "session_id": session_id
            if isinstance(session_id, str) and session_id
            else None,
            "run_id": task.task_id,
            "parent_run_id": (
                parent_run_id
                if isinstance(parent_run_id, str) and parent_run_id
                else None
            ),
            "root_run_id": (
                root_run_id if isinstance(root_run_id, str) and root_run_id else None
            ),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox or root_agent_inbox,
            "task_handle": TaskHandle(
                task.task_id,
                self.library.task_store,
                tool_name=tool_name,
                agent_inbox=root_agent_inbox,
            ),
            "task_activity_recorder": TaskActivityRecorder(
                task.task_id, self.library.task_store
            ),
        }
        future = Executor.get_instance().submit(
            partial(
                self.run_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    self.library.task_store,
                    tool_name=tool_name,
                    agent_inbox=root_agent_inbox,
                ),
                tool_name=tool_name,
                call_params=runner_params,
                execution_scope=execution_scope,
                agent_inbox=root_agent_inbox,
            )
        )
        self.register_task_future(task.task_id, future)
        future.add_done_callback(partial(self.cleanup_task_future, task.task_id))
        future.add_done_callback(self.log_task_failure)
        return self._tool_call_factory(
            id=tool_id,
            name=tool_name,
            parameters=self.library._build_call_parameters_for_response(call_params),
            result=self.library._build_background_dispatch_result(
                task_id=task.task_id,
                tool_name=tool_name,
                task_kind=task_kind,
            ),
        )

    def publish_task_notification(
        self,
        *,
        task_id: str,
        tool_name: str,
        status: str,
        hint: str,
        agent_inbox: AgentInbox | None = None,
    ) -> AgentNotification | None:
        inbox = agent_inbox or self.library.agent_inbox
        if inbox is None:
            return None
        return inbox.publish(
            AgentNotification(
                notification_id=uuid4().hex[:8],
                source="task",
                ref=task_id,
                status=status,
                hint=hint,
                metadata={"tool": tool_name},
                dedupe_key=f"task:{task_id}:{status}",
            )
        )
