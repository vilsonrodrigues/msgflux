"""Durable execution runtime for DSL inline pipelines.

Provides checkpoint/resume support on top of InlineDSL and AsyncInlineDSL.
Each arrow (->) in the DSL expression is a checkpoint boundary.
"""

import time
from typing import Any, Callable, Dict, List, Mapping, Optional

import msgspec

from msgflux.context import _current_while_scope, get_run_id
from msgflux.data.stores import (
    AsyncCheckpointStore,
    CheckpointStore,
    MemoryCheckpointStore,
    StepEvent,
    StepStatus,
)
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.parser import AsyncInlineDSL, InlineDSL
from msgflux.logger import logger


class MaxRetriesExceededError(RuntimeError):
    """Raised when a step exceeds the maximum number of retries."""

    def __init__(self, step_name: str, max_retries: int, last_error: str):
        self.step_name = step_name
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(
            f"Step `{step_name}` failed after {max_retries} retries. "
            f"Last error: {last_error}"
        )


def _step_name(step: Dict[str, Any], index: int) -> str:
    """Generate a deterministic name for a pipeline step.

    Uses step content for version resilience, with #index for disambiguation.
    """
    step_type = step["type"]
    if step_type == "module":
        base = f"module:{step['module']}"
    elif step_type == "parallel":
        mods = ",".join(step["modules"])
        base = f"parallel:[{mods}]"
    elif step_type == "conditional":
        base = f"conditional:{step['condition']}"
    elif step_type == "while":
        base = f"while:{step['condition']}"
    else:
        base = f"unknown:{step_type}"
    return f"{base}#{index}"


def _scoped_step_name(step: Dict[str, Any], index: int) -> str:
    """Generate step name with while-loop scope prefix."""
    scope = _current_while_scope.get()
    name = _step_name(step, index)
    if scope:
        return f"{scope}{name}"
    return name


class DurableInlineDSL(InlineDSL):
    """InlineDSL with checkpoint/resume support.

    On each step:
      1. Check store: COMPLETED -> skip and restore snapshot
      2. Check store: FAILED with retry_count >= max_retries -> raise
      3. Mark step as IN_PROGRESS
      4. Execute step
      5. Save message snapshot, mark as COMPLETED
      6. On exception: mark as FAILED with retry_count, re-raise
    """

    def __init__(
        self,
        store: Optional[CheckpointStore] = None,
        max_iterations: int = 1000,
        max_retries: int = 3,
    ):
        super().__init__(max_iterations=max_iterations)
        self.store = store or MemoryCheckpointStore()
        self.max_retries = max_retries

    def _execute_steps(
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> dotdict:
        """Execute steps with checkpoint/resume logic."""
        run_id = get_run_id()
        if run_id is None:
            return super()._execute_steps(steps, modules, message)

        current_message = message

        for index, step in enumerate(steps):
            name = _scoped_step_name(step, index)
            status = self.store.get_step_status(run_id, name)

            if status == StepStatus.COMPLETED:
                snapshot = self._load_step_snapshot(run_id, name)
                if snapshot is not None:
                    current_message = snapshot
                    logger.info(f"[DurableInline] Skipping completed step: {name}")
                    continue
                logger.warning(
                    f"[DurableInline] Step `{name}` marked completed "
                    "but snapshot missing. Re-executing."
                )

            if status == StepStatus.FAILED:
                retry_count = self.store.get_step_retry_count(run_id, name)
                if retry_count >= self.max_retries:
                    last_event = self._get_last_event(run_id, name)
                    raise MaxRetriesExceededError(
                        step_name=name,
                        max_retries=self.max_retries,
                        last_error=last_event.error if last_event else "unknown",
                    )

            self.store.save_event(
                run_id,
                StepEvent(
                    step_name=name,
                    status=StepStatus.IN_PROGRESS,
                    timestamp=time.time(),
                ),
            )

            try:
                current_message = super()._execute_steps(
                    [step], modules, current_message
                )
            except Exception as e:
                retry_count = self.store.get_step_retry_count(run_id, name) + 1
                self.store.save_event(
                    run_id,
                    StepEvent(
                        step_name=name,
                        status=StepStatus.FAILED,
                        timestamp=time.time(),
                        error=str(e),
                        retry_count=retry_count,
                    ),
                )
                raise

            snapshot_bytes = msgspec.json.encode(current_message.to_dict())
            self.store.save_event(
                run_id,
                StepEvent(
                    step_name=name,
                    status=StepStatus.COMPLETED,
                    timestamp=time.time(),
                    message_snapshot=snapshot_bytes,
                ),
            )

        return current_message

    def _execute_while_loop(
        self,
        condition: str,
        actions: str,
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> dotdict:
        """Execute while loop with per-iteration checkpoint scoping."""
        run_id = get_run_id()
        if run_id is None:
            return super()._execute_while_loop(condition, actions, modules, message)

        iterations = 0
        current_message = message
        parent_scope = _current_while_scope.get()

        while self._evaluate_condition(condition, current_message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            scope = f"{parent_scope}while:{condition}/iter:{iterations}/"
            token = _current_while_scope.set(scope)
            try:
                actions_steps = self.parse(actions)
                current_message = self._execute_steps(
                    actions_steps, modules, current_message
                )
            finally:
                _current_while_scope.reset(token)

            iterations += 1

        return current_message

    def _load_step_snapshot(self, run_id: str, step_name: str) -> Optional[dotdict]:
        """Load the message snapshot for a completed step."""
        run = self.store.load_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if (
                event.step_name == step_name
                and event.status == StepStatus.COMPLETED
                and event.message_snapshot is not None
            ):
                data = msgspec.json.decode(event.message_snapshot)
                return dotdict(data)
        return None

    def _get_last_event(self, run_id: str, step_name: str) -> Optional[StepEvent]:
        """Get the most recent event for a step."""
        run = self.store.load_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if event.step_name == step_name:
                return event
        return None


class AsyncDurableInlineDSL(AsyncInlineDSL):
    """Async version of DurableInlineDSL.

    Supports both sync ``CheckpointStore`` and async ``AsyncCheckpointStore``.
    When an ``AsyncCheckpointStore`` is provided, all store operations use
    the async methods (``asave_event``, ``aget_step_status``, etc.).
    """

    def __init__(
        self,
        store: Optional[CheckpointStore] = None,
        max_iterations: int = 1000,
        max_retries: int = 3,
    ):
        super().__init__(max_iterations=max_iterations)
        self.store = store or MemoryCheckpointStore()
        self.max_retries = max_retries
        self._async_store = isinstance(self.store, AsyncCheckpointStore)

    async def _store_get_step_status(self, run_id: str, name: str) -> Optional[str]:
        if self._async_store:
            return await self.store.aget_step_status(run_id, name)
        return self.store.get_step_status(run_id, name)

    async def _store_get_step_retry_count(self, run_id: str, name: str) -> int:
        if self._async_store:
            return await self.store.aget_step_retry_count(run_id, name)
        return self.store.get_step_retry_count(run_id, name)

    async def _store_save_event(self, run_id: str, event: StepEvent) -> None:
        if self._async_store:
            await self.store.asave_event(run_id, event)
        else:
            self.store.save_event(run_id, event)

    async def _aexecute_steps(
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> dotdict:
        """Async execute steps with checkpoint/resume logic."""
        run_id = get_run_id()
        if run_id is None:
            return await super()._aexecute_steps(steps, modules, message)

        current_message = message

        for index, step in enumerate(steps):
            name = _scoped_step_name(step, index)
            status = await self._store_get_step_status(run_id, name)

            if status == StepStatus.COMPLETED:
                snapshot = await self._aload_step_snapshot(run_id, name)
                if snapshot is not None:
                    current_message = snapshot
                    logger.info(f"[DurableInline] Skipping completed step: {name}")
                    continue
                logger.warning(
                    f"[DurableInline] Step `{name}` marked completed "
                    "but snapshot missing. Re-executing."
                )

            if status == StepStatus.FAILED:
                retry_count = await self._store_get_step_retry_count(run_id, name)
                if retry_count >= self.max_retries:
                    last_event = await self._aget_last_event(run_id, name)
                    raise MaxRetriesExceededError(
                        step_name=name,
                        max_retries=self.max_retries,
                        last_error=last_event.error if last_event else "unknown",
                    )

            await self._store_save_event(
                run_id,
                StepEvent(
                    step_name=name,
                    status=StepStatus.IN_PROGRESS,
                    timestamp=time.time(),
                ),
            )

            try:
                current_message = await super()._aexecute_steps(
                    [step], modules, current_message
                )
            except Exception as e:
                retry_count = await self._store_get_step_retry_count(run_id, name) + 1
                await self._store_save_event(
                    run_id,
                    StepEvent(
                        step_name=name,
                        status=StepStatus.FAILED,
                        timestamp=time.time(),
                        error=str(e),
                        retry_count=retry_count,
                    ),
                )
                raise

            snapshot_bytes = msgspec.json.encode(current_message.to_dict())
            await self._store_save_event(
                run_id,
                StepEvent(
                    step_name=name,
                    status=StepStatus.COMPLETED,
                    timestamp=time.time(),
                    message_snapshot=snapshot_bytes,
                ),
            )

        return current_message

    async def _aexecute_while_loop(
        self,
        condition: str,
        actions: str,
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> dotdict:
        """Async execute while loop with per-iteration checkpoint scoping."""
        run_id = get_run_id()
        if run_id is None:
            return await super()._aexecute_while_loop(
                condition, actions, modules, message
            )

        iterations = 0
        current_message = message
        parent_scope = _current_while_scope.get()

        while self._evaluate_condition(condition, current_message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            scope = f"{parent_scope}while:{condition}/iter:{iterations}/"
            token = _current_while_scope.set(scope)
            try:
                actions_steps = self.parse(actions)
                current_message = await self._aexecute_steps(
                    actions_steps, modules, current_message
                )
            finally:
                _current_while_scope.reset(token)

            iterations += 1

        return current_message

    async def _aload_step_snapshot(
        self, run_id: str, step_name: str
    ) -> Optional[dotdict]:
        """Load snapshot — uses async store when available."""
        if self._async_store:
            run = await self.store.aload_run(run_id)
        else:
            run = self.store.load_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if (
                event.step_name == step_name
                and event.status == StepStatus.COMPLETED
                and event.message_snapshot is not None
            ):
                data = msgspec.json.decode(event.message_snapshot)
                return dotdict(data)
        return None

    async def _aget_last_event(
        self, run_id: str, step_name: str
    ) -> Optional[StepEvent]:
        """Get the most recent event for a step."""
        if self._async_store:
            run = await self.store.aload_run(run_id)
        else:
            run = self.store.load_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if event.step_name == step_name:
                return event
        return None
