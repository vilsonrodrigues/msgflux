"""Durable inline DSL — checkpoint per step with cursor-based resume."""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

from msgflux.context import session_context
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.core import AsyncInlineDSL, InlineDSL
from msgflux.logger import logger

if TYPE_CHECKING:
    from msgflux.data.stores.base import CheckpointStore


# ── Event type constants ─────────────────────────────────────────────────────

_EV_RUN_STARTED = "run_started"
_EV_RUN_RESUMED = "run_resumed"
_EV_RUN_COMPLETED = "run_completed"
_EV_STEP_COMPLETED = "step_completed"
_EV_STEP_FAILED = "step_failed"


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
            self.run_id,
            expression,
            status,
            cursor,
            message,
            error,
        )
        if event is not None:
            self.store.save_with_event(
                self.namespace,
                self.session_id,
                self.run_id,
                state,
                event,
            )
        else:
            self.store.save_state(
                self.namespace,
                self.session_id,
                self.run_id,
                state,
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
    ) -> Tuple[dotdict, Dict[str, Any]]:
        current_message = message
        attempts = self.max_retries + 1
        signals: Dict[str, Any] = {}

        for i, step in enumerate(steps):
            if i < _start_index:
                continue

            for attempt in range(attempts):
                try:
                    if step["type"] == "module":
                        module = modules.get(step["module"])
                        if not module:
                            raise ValueError(f"Module `{step['module']}` not found.")
                        signals = self._call_module(module, current_message)

                    elif step["type"] == "parallel":
                        parallel_modules = self._resolve_parallel(
                            step,
                            modules,
                        )
                        signals = self._execute_parallel(
                            parallel_modules,
                            current_message,
                        )

                    elif step["type"] == "conditional":
                        signals = self._execute_conditional(
                            step,
                            modules,
                            current_message,
                        )

                    elif step["type"] == "while":
                        current_message, signals = self._execute_while_durable(
                            step,
                            modules,
                            current_message,
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
                                _EV_STEP_FAILED,
                                step_index=i,
                                step_name=name,
                                error=str(e),
                            ),
                        )
                        raise
                    logger.warning(
                        "Step %d failed (attempt %d/%d): %s. Retrying in %.1fs…",
                        i,
                        attempt + 1,
                        attempts,
                        e,
                        self.retry_delay,
                    )
                    time.sleep(self.retry_delay)

            # Checkpoint + event after each completed step
            self._save(
                _expression,
                _make_cursor(i + 1, _frames),
                current_message,
                event=_make_event(
                    _EV_STEP_COMPLETED,
                    step_index=i,
                    step_name=_step_name(step),
                ),
            )

            if signals:
                return current_message, signals

        return current_message, {}

    def _execute_while_durable(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        expression: str,
        outer_step_index: int,
        frames: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[dotdict, Dict[str, Any]]:
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
        actions_steps = self.parse(actions)

        while self._evaluate_condition(condition, current_message):
            if iteration >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            # Skip to the right inner position when resuming
            effective_start = inner_start if iteration == start_iteration else 0

            frame = {
                "type": "while",
                "condition": condition,
                "iteration": iteration,
                "inner_step": 0,
            }
            active_frames = [*current_frames, frame]

            current_message, signals = self._execute_steps(
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
                _make_cursor(
                    outer_step_index,
                    [
                        *current_frames,
                        {
                            "type": "while",
                            "condition": condition,
                            "iteration": iteration,
                            "inner_step": 0,
                        },
                    ],
                ),
                current_message,
            )

            if "$stop" in signals:
                return current_message, signals
            if "$break" in signals:
                break

        return current_message, {}

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
                    self.run_id,
                    start_index,
                )
                current = resumed_msg
                event_type = _EV_RUN_RESUMED
            else:
                current = message
                event_type = _EV_RUN_STARTED

            self.store.append_event(
                self.namespace,
                self.session_id,
                self.run_id,
                _make_event(event_type),
            )

            steps = self.parse(expression)
            result, _signals = self._execute_steps(
                steps,
                modules,
                current,
                _expression=expression,
                _start_index=start_index,
                _frames=frames,
            )

            self._save(
                expression,
                _make_cursor(len(steps)),
                result,
                status="completed",
                event=_make_event(_EV_RUN_COMPLETED),
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

        from msgflux.data.stores.base import AsyncCheckpointStore  # noqa: PLC0415

        self._async_store = isinstance(store, AsyncCheckpointStore)

    # ── Resume ───────────────────────────────────────────────────────────

    async def _atry_resume(self, expression: str) -> tuple:
        store = self.store
        if self._async_store:
            state = await store.aload_state(
                self.namespace,
                self.session_id,
                self.run_id,
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
            self.run_id,
            expression,
            status,
            cursor,
            message,
            error,
        )
        if event is not None:
            if self._async_store:
                await self.store.asave_with_event(
                    self.namespace,
                    self.session_id,
                    self.run_id,
                    state,
                    event,
                )
            else:
                self.store.save_with_event(
                    self.namespace,
                    self.session_id,
                    self.run_id,
                    state,
                    event,
                )
        elif self._async_store:
            await self.store.asave_state(
                self.namespace,
                self.session_id,
                self.run_id,
                state,
            )
        else:
            self.store.save_state(
                self.namespace,
                self.session_id,
                self.run_id,
                state,
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
    ) -> Tuple[dotdict, Dict[str, Any]]:
        current_message = message
        attempts = self.max_retries + 1
        signals: Dict[str, Any] = {}

        for i, step in enumerate(steps):
            if i < _start_index:
                continue

            for attempt in range(attempts):
                try:
                    if step["type"] == "module":
                        module = modules.get(step["module"])
                        if not module:
                            raise ValueError(f"Module `{step['module']}` not found.")
                        signals = await self._acall_module(
                            module,
                            current_message,
                        )

                    elif step["type"] == "parallel":
                        parallel_modules = self._resolve_parallel(step, modules)
                        signals = await self._aexecute_parallel(
                            parallel_modules,
                            current_message,
                        )

                    elif step["type"] == "conditional":
                        signals = await self._aexecute_conditional(
                            step,
                            modules,
                            current_message,
                        )

                    elif step["type"] == "while":
                        current_message, signals = await self._aexecute_while_durable(
                            step,
                            modules,
                            current_message,
                            expression=_expression,
                            outer_step_index=i,
                            frames=_frames,
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
                                _EV_STEP_FAILED,
                                step_index=i,
                                step_name=name,
                                error=str(e),
                            ),
                        )
                        raise
                    logger.warning(
                        "Step %d failed (attempt %d/%d): %s. Retrying in %.1fs…",
                        i,
                        attempt + 1,
                        attempts,
                        e,
                        self.retry_delay,
                    )
                    await asyncio.sleep(self.retry_delay)

            # Checkpoint + event after each completed step
            await self._asave(
                _expression,
                _make_cursor(i + 1, _frames),
                current_message,
                event=_make_event(
                    _EV_STEP_COMPLETED,
                    step_index=i,
                    step_name=_step_name(step),
                ),
            )

            if signals:
                return current_message, signals

        return current_message, {}

    async def _aexecute_while_durable(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
        *,
        expression: str,
        outer_step_index: int,
        frames: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[dotdict, Dict[str, Any]]:
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
        actions_steps = self.parse(actions)

        while self._evaluate_condition(condition, current_message):
            if iteration >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            effective_start = inner_start if iteration == start_iteration else 0

            frame = {
                "type": "while",
                "condition": condition,
                "iteration": iteration,
                "inner_step": 0,
            }
            active_frames = [*current_frames, frame]

            current_message, signals = await self._aexecute_steps(
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
                _make_cursor(
                    outer_step_index,
                    [
                        *current_frames,
                        {
                            "type": "while",
                            "condition": condition,
                            "iteration": iteration,
                            "inner_step": 0,
                        },
                    ],
                ),
                current_message,
            )

            if "$stop" in signals:
                return current_message, signals
            if "$break" in signals:
                break

        return current_message, {}

    async def _aemit_event(self, event: Dict[str, Any]) -> None:
        """Append an event to the store (async-aware)."""
        if self._async_store:
            await self.store.aappend_event(
                self.namespace,
                self.session_id,
                self.run_id,
                event,
            )
        else:
            self.store.append_event(
                self.namespace,
                self.session_id,
                self.run_id,
                event,
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
                    self.run_id,
                    start_index,
                )
                current = resumed_msg
                event_type = _EV_RUN_RESUMED
            else:
                current = message
                event_type = _EV_RUN_STARTED

            await self._aemit_event(_make_event(event_type))

            steps = self.parse(expression)
            result, _signals = await self._aexecute_steps(
                steps,
                modules,
                current,
                _expression=expression,
                _start_index=start_index,
                _frames=frames,
            )

            await self._asave(
                expression,
                _make_cursor(len(steps)),
                result,
                status="completed",
                event=_make_event(_EV_RUN_COMPLETED),
            )
            return result
