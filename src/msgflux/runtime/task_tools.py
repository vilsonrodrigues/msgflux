from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


class TaskRuntimeTools:
    """Registers and implements task-control tools for background execution."""

    def __init__(
        self,
        library: Any,
    ):
        self.library = library
        self._base_enabled = False
        self._agent_enabled = False

    def reset(self) -> None:
        self._base_enabled = False
        self._agent_enabled = False

    def _add_runtime_tool(
        self,
        *,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        impl: Callable,
    ) -> None:
        self.library.add_runtime_tool(
            name=name,
            description=description,
            annotations=annotations,
            impl=impl,
        )

    def ensure_base_tools(self) -> None:
        if self._base_enabled:
            return
        self._base_enabled = True
        self._add_runtime_tool(
            name="task_status",
            description="Get the current status of a background task by task_id.",
            annotations={"task_id": str},
            impl=self.task_status,
        )
        self._add_runtime_tool(
            name="task_list",
            description="List background tasks registered in the current tool library.",
            annotations={"status": Optional[str]},
            impl=self.task_list,
        )
        self._add_runtime_tool(
            name="task_output",
            description="Get the final output of a background task by task_id.",
            annotations={"task_id": str},
            impl=self.task_output,
        )
        self._add_runtime_tool(
            name="task_wait",
            description=(
                "Wait for a background task to finish. "
                "Returns the final output, failed payload, or a timeout status."
            ),
            annotations={"task_id": str, "timeout": Optional[float]},
            impl=self.task_wait,
        )
        self._add_runtime_tool(
            name="task_stop",
            description=(
                "Request a cooperative stop for a background task. "
                "Stops immediately only if the task has not started yet."
            ),
            annotations={"task_id": str},
            impl=self.task_stop,
        )

    def ensure_agent_tools(self) -> None:
        if self._agent_enabled:
            return
        self._agent_enabled = True
        self._add_runtime_tool(
            name="task_activity",
            description="List compact activity entries for a background agent task.",
            annotations={"task_id": str, "limit": Optional[int]},
            impl=self.task_activity,
        )
        self._add_runtime_tool(
            name="task_message",
            description=(
                "Send a message to a background agent task. "
                "If it is still running, deliver the message to its inbox. "
                "If it already stopped, resume the task from its checkpoint."
            ),
            annotations={"task_id": str, "message": str},
            impl=self.task_message,
        )

    def task_status(self, task_id: str) -> Dict[str, Any]:
        task = self.library.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = task.to_dict()
        payload.update(self.build_task_timing_fields(task))
        last_activity = self.library.task_store.get_last_activity(task_id)
        if last_activity is not None:
            payload["last_activity_summary"] = self.format_task_activity_entry(
                last_activity
            )
        return payload

    def task_list(self, status: str | None = None) -> list[Dict[str, Any]]:
        tasks = []
        for task in self.library.task_store.list(status=status):
            payload = task.to_dict()
            payload.update(self.build_task_timing_fields(task))
            last_activity = self.library.task_store.get_last_activity(task.task_id)
            if last_activity is not None:
                payload["last_activity_summary"] = self.format_task_activity_entry(
                    last_activity
                )
            tasks.append(payload)
        return tasks

    def task_output(self, task_id: str) -> Any:
        task = self.library.task_store.get(task_id)
        return self.build_task_result(task_id=task_id, task=task)

    def task_activity(
        self,
        task_id: str,
        limit: int | None = 10,
    ) -> Any:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
            if limit <= 0:
                raise ValueError("`limit` must be greater than 0.")
        task = self.library.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_activity is only available for background agent tasks.",
            }
        activity = self.library.task_store.list_activity(task_id, limit=limit)
        return [self.format_task_activity_entry(item) for item in activity]

    def task_wait(self, task_id: str, timeout: float | None = None) -> Any:  # noqa: C901
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    f"`timeout` must be float, int or None, given `{type(timeout)}`"
                )
            if timeout < 0:
                raise ValueError("`timeout` must be greater than or equal to 0.")

        task = self.library.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status in {"completed", "failed", "stopped"}:
            return self.build_task_result(task_id=task_id, task=task)

        future = self.library.get_task_future(task_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                task = self.library.task_store.get(task_id)
                return self.build_task_timeout_result(task_id=task_id, task=task)
            except Exception:
                task = self.library.task_store.get(task_id)
                return self.build_task_result(task_id=task_id, task=task)
            task = self.library.task_store.get(task_id)
            return self.build_task_result(task_id=task_id, task=task)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = self.library.task_store.get(task_id)
            if task is None or task.status in {"completed", "failed", "stopped"}:
                return self.build_task_result(task_id=task_id, task=task)
            if deadline is not None and time.monotonic() >= deadline:
                return self.build_task_timeout_result(task_id=task_id, task=task)
            time.sleep(0.05)

    def task_stop(self, task_id: str) -> Dict[str, Any]:
        task = self.library.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}

        if task.status in {"completed", "failed", "stopped"}:
            return {
                "task_id": task_id,
                "status": task.status,
                "message": "Task already reached a terminal state.",
            }

        self.library.task_store.request_stop(task_id)
        future = self.library.get_task_future(task_id)
        if future is not None and future.cancel():
            stopped = self.library.task_store.stop(
                task_id,
                reason="Task was cancelled before it started running.",
            )
            return {
                "task_id": task_id,
                "status": "stopped",
                "message": "Task stopped before execution started.",
                "task_status": stopped.status if stopped is not None else "stopped",
            }

        return {
            "task_id": task_id,
            "status": "stop_requested",
            "message": (
                "Stop requested. The task will stop at the next cooperative checkpoint."
            ),
        }

    def task_message(self, task_id: str, message: str) -> Dict[str, Any]:
        task = self.library.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_message is only available for background agent tasks.",
            }
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")

        task_inbox = self.library.get_task_inbox(task_id)
        if task.status == "running" and task_inbox is not None:
            task_inbox.publish(
                {
                    "source": "task_message",
                    "ref": task_id,
                    "status": "message",
                    "hint": message.strip(),
                    "metadata": {"direction": "root_to_task"},
                }
            )
            self.library.task_store.add_activity(
                task_id,
                kind="message",
                summary=f"Root message: {self.truncate_activity_text(message)}",
                metadata={"direction": "root_to_task"},
            )
            return {
                "task_id": task_id,
                "status": "delivered",
                "message": "Message delivered to the running background agent.",
            }

        resumed = self.library.resume_background_agent_task(
            task=task, message=message.strip()
        )
        return {
            "task_id": task_id,
            "status": "resumed",
            "message": resumed,
        }

    def build_task_result(self, *, task_id: str, task: Any | None) -> Any:
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status == "completed":
            return task.result
        if task.status == "failed":
            return {"task_id": task_id, "status": task.status, "error": task.error}
        if task.status == "stopped":
            return {
                "task_id": task_id,
                "status": task.status,
                "reason": task.metadata.get("stop_reason"),
            }
        return {
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress.to_dict(),
        }

    def build_task_timeout_result(
        self, *, task_id: str, task: Any | None
    ) -> Dict[str, Any]:
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = {
            "task_id": task_id,
            "status": "timeout",
            "task_status": task.status,
        }
        if task.status not in {"completed", "failed"}:
            if task.status == "stopped":
                payload["reason"] = task.metadata.get("stop_reason")
                return payload
            payload["progress"] = task.progress.to_dict()
        elif task.status == "failed":
            payload["error"] = task.error
        return payload

    def build_task_timing_fields(self, task: Any) -> Dict[str, Any]:
        started_at = task.created_at
        now = time.time()
        created_ts = self.parse_utc_timestamp(task.created_at)
        completed_ts = self.parse_utc_timestamp(task.completed_at)
        payload: Dict[str, Any] = {"started_at": started_at}
        if created_ts is None:
            return payload
        if completed_ts is not None:
            payload["elapsed_seconds"] = round(completed_ts - created_ts, 3)
        else:
            payload["running_for_seconds"] = round(now - created_ts, 3)
        return payload

    def format_task_activity_entry(self, activity: Any) -> str:
        label_map = {
            "status": "Status",
            "progress": "Progress",
            "tool_call": "ToolCall",
            "error": "Error",
            "message": "Message",
        }
        label = label_map.get(activity.kind, activity.kind.title())
        return f"{label}: {activity.summary}"

    def build_background_dispatch_result(
        self,
        *,
        task_id: str,
        tool_name: str,
        task_kind: str,
    ) -> str:
        actions = ["`task_status`", "`task_stop`", "`task_wait`", "`task_output`"]
        if task_kind == "agent":
            actions.insert(1, "`task_activity`")
            actions.insert(2, "`task_message`")
        return (
            f"The `{tool_name}` tool is running in the background with "
            f"task_id='{task_id}'. Use that task_id with "
            + ", ".join(actions[:-1])
            + f", or {actions[-1]}."
        )

    @staticmethod
    def parse_utc_timestamp(value: str | None) -> float | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return (
                datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()
            )
        except ValueError:
            return None

    @staticmethod
    def truncate_activity_text(value: str, *, limit: int = 140) -> str:
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
