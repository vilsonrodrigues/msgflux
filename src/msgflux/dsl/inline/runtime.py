"""Durable inline DSL — checkpoint per step with cursor-based resume."""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional

from msgflux.context import session_context
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.core import AsyncInlineDSL, InlineDSL
from msgflux.logger import logger

if TYPE_CHECKING:
    from msgflux.data.stores.base import CheckpointStore


# ── Cursor helpers ───────────────────────────────────────────────────────────


def _make_cursor(
    step_index: int,
    frames: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {"step_index": step_index, "frames": frames or []}


def _make_state(
    run_id: str,
    expression: str,
    status: str,
    cursor: Dict[str, Any],
    message: dotdict,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "run_id": run_id,
        "expression": expression,
        "status": status,
        "cursor": cursor,
        "message_snapshot": message.to_dict(),
    }
    if error is not None:
        state["error"] = error
    return state


def _step_name(step: Dict[str, Any]) -> str:
    """Return a human-readable name for a parsed step."""
    if step["type"] == "module":
        return step["module"]
    if step["type"] == "parallel":
        return f"[{', '.join(step['modules'])}]"
    if step["type"] == "conditional":
        return f"{{{step['condition']}?...}}"
    if step["type"] == "while":
        return f"@{{{step['condition']}}}"
    return step["type"]


def _make_event(
    event_type: str,
    *,
    step_index: Optional[int] = None,
    step_name: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": event_type,
        "timestamp": time.time(),
    }
    if step_index is not None:
        event["step_index"] = step_index
    if step_name is not None:
        event["step_name"] = step_name
    if error is not None:
        event["error"] = error
    return event


# ── Sync durable DSL ────────────────────────────────────────────────────────


class DurableInlineDSL(InlineDSL):
    """InlineDSL with checkpoint-per-step and cursor-based resume.

    When a ``store`` is provided, every completed step is checkpointed.
    If the process crashes, re-running with the same ``(namespace, session_id,
    run_id)`` resumes from the last completed step.
    """

    def __init__(
        self,
        store: "CheckpointStore",
        *,
        namespace: str = "inline",
        session_id: str = "default",
        run_id: Optional[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        max_iterations: int = 1000,
    ):
        super().__init__(max_iterations=max_iterations)
        self.store = store
        self.namespace = namespace
        self.session_id = session_id
        self.run_id = run_id or str(uuid.uuid4())
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── Resume ───────────────────────────────────────────────────────────

    def _try_resume(self, expression: str) -> tuple:
        """Return ``(start_index, while_frames, message | None)``."""
        state = self.store.load_state(self.namespace, self.session_id, self.run_id)
        if state is None or state.get("status") == "completed":
            return 0, [], None
        if state.get("expression") != expression:
            return 0, [], None
        cursor = state.get("cursor", {})
        snapshot = state.get("message_snapshot")
        if snapshot is None:
            return 0, [], None
        msg = dotdict(snapshot)
        return cursor.get("step_index", 0), cursor.get("frames", []), msg

    # ── Save ─────────────────────────────────────────────────────────────

    def _save(
        self,
        expression: str,
        cursor: Dict[str, Any],
        message: dotdict,
        status: str = "running",
        error: Optional[str] = None,
        event: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = _make_state(
            self.run_id, expression, status, cursor, message, error,
        )
        if event is not None:
            self.store.save_with_event(
                self.namespace, self.session_id, self.run_id, state, event,
            )
        else:
            self.store.save_state(
                self.namespace, self.session_id, self.run_id, state,
            )

    # ── Execution (overrides) ────────────────────────────────────────────

    def _execute_steps(  # noqa: C901
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        _expression: str = "",
        _start_index: int = 0,
        _frames: Optional[List[Dict[str, Any]]] = None,
    ) -> dotdict:
        current_message = message
        attempts = self.max_retries + 1

        for i, step in enumerate(steps):
            if i < _start_index:
                continue

            for attempt in range(attempts):
                try:
                    if step["type"] == "module":
                        module = modules.get(step["module"])
                        if not module:
                            raise ValueError(
                                f"Module `{step['module']}` not found."
                            )
                        self._call_module(module, current_message)

                    elif step["type"] == "parallel":
                        parallel_modules = self._resolve_parallel(
                            step, modules,
                        )
                        self._execute_parallel(
                            parallel_modules, current_message,
                        )

                    elif step["type"] == "conditional":
                        self._execute_conditional(
                            step, modules, current_message,
                        )

                    elif step["type"] == "while":
                        current_message = self._execute_while_durable(
                            step, modules, current_message,
                            expression=_expression,
                            outer_step_index=i,
                            frames=_frames,
                        )

                    break  # step succeeded

                except Exception as e:
                    if attempt >= self.max_retries:
                        name = _step_name(step)
                        self._save(
                            _expression,
                            _make_cursor(i, _frames),
                            current_message,
                            status="failed",
                            error=str(e),
                            event=_make_event(
                                "step_failed",
                                step_index=i,
                                step_name=name,
                                error=str(e),
                            ),
                        )
                        raise
                    logger.warning(
                        "Step %d failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs…",
                        i, attempt + 1, attempts, e, self.retry_delay,
                    )
                    time.sleep(self.retry_delay)

            # Checkpoint + event after each completed step
            self._save(
                _expression,
                _make_cursor(i + 1, _frames),
                current_message,
                event=_make_event(
                    "step_completed",
                    step_index=i,
                    step_name=_step_name(step),
                ),
            )

        return current_message

    def _execute_conditional(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> None:
        condition_result = self._evaluate_condition(step["condition"], message)
        branch = step["true_branch"] if condition_result else step["false_branch"]
        for module_name in branch:
            module = modules.get(module_name)
            if not module:
                raise ValueError(
                    f"Module `{module_name}` not found in conditional branch."
                )
            self._call_module(module, message)

    def _execute_while_durable(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        expression: str,
        outer_step_index: int,
        frames: Optional[List[Dict[str, Any]]] = None,
    ) -> dotdict:
        condition = step["condition"]
        actions = step["actions"]
        current_message = message

        # Determine starting iteration from frames
        start_iteration = 0
        inner_start = 0
        current_frames = list(frames or [])
        if current_frames:
            last = current_frames[-1]
            if last.get("condition") == condition:
                start_iteration = last.get("iteration", 0)
                inner_start = last.get("inner_step", 0)
                current_frames = current_frames[:-1]  # pop consumed frame

        iteration = 0
        while self._evaluate_condition(condition, current_message):
            if iteration >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            actions_steps = self.parse(actions)

            # Skip to the right inner position when resuming
            effective_start = inner_start if iteration == start_iteration else 0

            frame = {
                "type": "while",
                "condition": condition,
                "iteration": iteration,
                "inner_step": 0,
            }
            active_frames = [*current_frames, frame]

            current_message = self._execute_steps(
                actions_steps,
                modules,
                current_message,
                _expression=expression,
                _start_index=effective_start,
                _frames=active_frames,
            )

            iteration += 1

            # Checkpoint after each loop iteration
            self._save(
                expression,
                _make_cursor(outer_step_index, [
                    *current_frames,
                    {"type": "while", "condition": condition,
                     "iteration": iteration, "inner_step": 0},
                ]),
                current_message,
            )

        return current_message

    @staticmethod
    def _resolve_parallel(
        step: Dict[str, Any], modules: Mapping[str, Callable]
    ) -> list:
        parallel_modules = []
        for mod_name in step["modules"]:
            module = modules.get(mod_name)
            if not module:
                raise ValueError(
                    f"Module {mod_name} not found for parallel execution."
                )
            parallel_modules.append(module)
        if not parallel_modules:
            raise ValueError(
                f"No valid modules found for parallel execution in {step['modules']}."
            )
        return parallel_modules

    def __call__(
        self, expression: str, modules: Mapping[str, Callable], message: dotdict
    ) -> dotdict:
        with session_context(
            session_id=self.session_id,
            namespace=self.namespace,
        ):
            start_index, frames, resumed_msg = self._try_resume(expression)
            if resumed_msg is not None:
                logger.info(
                    "Resuming inline run '%s' from step %d",
                    self.run_id, start_index,
                )
                current = resumed_msg
                event_type = "run_resumed"
            else:
                current = message
                event_type = "run_started"

            self.store.append_event(
                self.namespace, self.session_id, self.run_id,
                _make_event(event_type),
            )

            result = self._execute_steps(
                self.parse(expression),
                modules,
                current,
                _expression=expression,
                _start_index=start_index,
                _frames=frames,
            )

            # Mark completed
            total = len(self.parse(expression))
            self._save(
                expression, _make_cursor(total), result, status="completed",
                event=_make_event("run_completed"),
            )
            return result


# ── Async durable DSL ────────────────────────────────────────────────────────


class AsyncDurableInlineDSL(AsyncInlineDSL):
    """Async version of DurableInlineDSL."""

    def __init__(
        self,
        store: "CheckpointStore",
        *,
        namespace: str = "inline",
        session_id: str = "default",
        run_id: Optional[str] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        max_iterations: int = 1000,
    ):
        super().__init__(max_iterations=max_iterations)
        self.store = store
        self.namespace = namespace
        self.session_id = session_id
        self.run_id = run_id or str(uuid.uuid4())
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── Resume ───────────────────────────────────────────────────────────

    async def _atry_resume(self, expression: str) -> tuple:
        store = self.store
        if hasattr(store, "aload_state"):
            state = await store.aload_state(
                self.namespace, self.session_id, self.run_id,
            )
        else:
            state = store.load_state(self.namespace, self.session_id, self.run_id)
        if state is None or state.get("status") == "completed":
            return 0, [], None
        if state.get("expression") != expression:
            return 0, [], None
        cursor = state.get("cursor", {})
        snapshot = state.get("message_snapshot")
        if snapshot is None:
            return 0, [], None
        msg = dotdict(snapshot)
        return cursor.get("step_index", 0), cursor.get("frames", []), msg

    # ── Save ─────────────────────────────────────────────────────────────

    async def _asave(
        self,
        expression: str,
        cursor: Dict[str, Any],
        message: dotdict,
        status: str = "running",
        error: Optional[str] = None,
        event: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = _make_state(
            self.run_id, expression, status, cursor, message, error,
        )
        if event is not None:
            if hasattr(self.store, "asave_with_event"):
                await self.store.asave_with_event(
                    self.namespace, self.session_id, self.run_id,
                    state, event,
                )
            else:
                self.store.save_with_event(
                    self.namespace, self.session_id, self.run_id,
                    state, event,
                )
        elif hasattr(self.store, "asave_state"):
            await self.store.asave_state(
                self.namespace, self.session_id, self.run_id, state,
            )
        else:
            self.store.save_state(
                self.namespace, self.session_id, self.run_id, state,
            )

    # ── Execution (overrides) ────────────────────────────────────────────

    async def _aexecute_steps(  # noqa: C901
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        _expression: str = "",
        _start_index: int = 0,
        _frames: Optional[List[Dict[str, Any]]] = None,
    ) -> dotdict:
        current_message = message
        attempts = self.max_retries + 1

        for i, step in enumerate(steps):
            if i < _start_index:
                continue

            for attempt in range(attempts):
                try:
                    if step["type"] == "module":
                        module = modules.get(step["module"])
                        if not module:
                            raise ValueError(
                                f"Module `{step['module']}` not found."
                            )
                        await self._acall_module(module, current_message)

                    elif step["type"] == "parallel":
                        parallel_modules = (
                            DurableInlineDSL._resolve_parallel(step, modules)
                        )
                        await self._aexecute_parallel(
                            parallel_modules, current_message,
                        )

                    elif step["type"] == "conditional":
                        await self._aexecute_conditional(
                            step, modules, current_message,
                        )

                    elif step["type"] == "while":
                        current_message = (
                            await self._aexecute_while_durable(
                                step, modules, current_message,
                                expression=_expression,
                                outer_step_index=i,
                                frames=_frames,
                            )
                        )

                    break  # step succeeded

                except Exception as e:
                    if attempt >= self.max_retries:
                        name = _step_name(step)
                        await self._asave(
                            _expression,
                            _make_cursor(i, _frames),
                            current_message,
                            status="failed",
                            error=str(e),
                            event=_make_event(
                                "step_failed",
                                step_index=i,
                                step_name=name,
                                error=str(e),
                            ),
                        )
                        raise
                    logger.warning(
                        "Step %d failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs…",
                        i, attempt + 1, attempts, e, self.retry_delay,
                    )
                    await asyncio.sleep(self.retry_delay)

            # Checkpoint + event after each completed step
            await self._asave(
                _expression,
                _make_cursor(i + 1, _frames),
                current_message,
                event=_make_event(
                    "step_completed",
                    step_index=i,
                    step_name=_step_name(step),
                ),
            )

        return current_message

    async def _aexecute_conditional(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> None:
        condition_result = self._evaluate_condition(step["condition"], message)
        branch = step["true_branch"] if condition_result else step["false_branch"]
        for module_name in branch:
            module = modules.get(module_name)
            if not module:
                raise ValueError(
                    f"Module `{module_name}` not found in conditional branch."
                )
            await self._acall_module(module, message)

    async def _aexecute_while_durable(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        expression: str,
        outer_step_index: int,
        frames: Optional[List[Dict[str, Any]]] = None,
    ) -> dotdict:
        condition = step["condition"]
        actions = step["actions"]
        current_message = message

        start_iteration = 0
        inner_start = 0
        current_frames = list(frames or [])
        if current_frames:
            last = current_frames[-1]
            if last.get("condition") == condition:
                start_iteration = last.get("iteration", 0)
                inner_start = last.get("inner_step", 0)
                current_frames = current_frames[:-1]

        iteration = 0
        while self._evaluate_condition(condition, current_message):
            if iteration >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            actions_steps = self.parse(actions)
            effective_start = inner_start if iteration == start_iteration else 0

            frame = {
                "type": "while",
                "condition": condition,
                "iteration": iteration,
                "inner_step": 0,
            }
            active_frames = [*current_frames, frame]

            current_message = await self._aexecute_steps(
                actions_steps,
                modules,
                current_message,
                _expression=expression,
                _start_index=effective_start,
                _frames=active_frames,
            )

            iteration += 1

            await self._asave(
                expression,
                _make_cursor(outer_step_index, [
                    *current_frames,
                    {"type": "while", "condition": condition,
                     "iteration": iteration, "inner_step": 0},
                ]),
                current_message,
            )

        return current_message

    async def _aemit_event(self, event: Dict[str, Any]) -> None:
        """Append an event to the store (async-aware)."""
        if hasattr(self.store, "aappend_event"):
            await self.store.aappend_event(
                self.namespace, self.session_id, self.run_id, event,
            )
        else:
            self.store.append_event(
                self.namespace, self.session_id, self.run_id, event,
            )

    async def __call__(
        self, expression: str, modules: Mapping[str, Callable], message: dotdict
    ) -> dotdict:
        with session_context(
            session_id=self.session_id,
            namespace=self.namespace,
        ):
            start_index, frames, resumed_msg = await self._atry_resume(expression)
            if resumed_msg is not None:
                logger.info(
                    "Resuming async inline run '%s' from step %d",
                    self.run_id, start_index,
                )
                current = resumed_msg
                event_type = "run_resumed"
            else:
                current = message
                event_type = "run_started"

            await self._aemit_event(_make_event(event_type))

            result = await self._aexecute_steps(
                self.parse(expression),
                modules,
                current,
                _expression=expression,
                _start_index=start_index,
                _frames=frames,
            )

            await self._asave(
                expression,
                _make_cursor(len(self.parse(expression))),
                result,
                status="completed",
                event=_make_event("run_completed"),
            )
            return result
