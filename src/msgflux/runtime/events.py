"""Execution-local event streaming primitives."""

from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping

from msgflux.runtime.context import ExecutionScope, get_execution_scope

__all__ = ["ExecutionEvent", "EventType", "emit_event"]


class EventType:
    RUN_START = "run.start"
    RUN_RESUME = "run.resume"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"
    RUN_INTERRUPTED = "run.interrupted"
    TURN_START = "turn.start"
    TURN_END = "turn.end"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MESSAGE_START = "message.start"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_END = "message.end"
    REASONING_DELTA = "reasoning.delta"
    REASONING_SUMMARY_DELTA = "reasoning_summary.delta"
    TOOL_START = "tool.start"
    TOOL_UPDATE = "tool.update"
    TOOL_END = "tool.end"
    COMPACTION_START = "compaction.start"
    COMPACTION_END = "compaction.end"
    CHECKPOINT_SAVED = "checkpoint.saved"
    HANDLER_ERROR = "handler.error"


@dataclass(frozen=True)
class ExecutionEvent:
    """One ordered event emitted by an execution."""

    type: str
    sequence: int
    timestamp: str
    data: Mapping[str, Any] = field(default_factory=dict)
    thread_id: str | None = None
    namespace: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None


class _EventSink:
    def __init__(self, publish: Callable[[ExecutionEvent], None]) -> None:
        self._publish = publish
        self._sequence = 0
        self._lock = threading.Lock()
        self._scope = ExecutionScope()

    def emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        current = get_execution_scope()
        if current.run_id is not None or current.thread_id is not None:
            self._scope = current
        scope = current if current.run_id is not None else self._scope
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        self._publish(
            ExecutionEvent(
                type=event_type,
                sequence=sequence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data=dict(data or {}),
                thread_id=scope.thread_id,
                namespace=scope.namespace,
                run_id=scope.run_id,
                parent_run_id=scope.parent_run_id,
                root_run_id=scope.root_run_id,
            )
        )


_CURRENT_EVENT_SINK: contextvars.ContextVar[_EventSink | None] = contextvars.ContextVar(
    "msgflux_event_sink", default=None
)


@contextmanager
def _capture_events(sink: _EventSink):
    token = _CURRENT_EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_EVENT_SINK.reset(token)


def emit_event(event_type: str, data: Mapping[str, Any] | None = None) -> None:
    """Emit an event when the current execution is being observed."""
    sink = _CURRENT_EVENT_SINK.get()
    if sink is not None:
        sink.emit(event_type, data)


_CLOSED = object()


class _SyncEventChannel:
    def __init__(self) -> None:
        self._queue: queue.Queue[ExecutionEvent | object] = queue.Queue()
        self.sink = _EventSink(self._queue.put_nowait)

    def close(self) -> None:
        self._queue.put_nowait(_CLOSED)

    def __iter__(self) -> Iterator[ExecutionEvent]:
        while True:
            item = self._queue.get()
            if item is _CLOSED:
                return
            yield item


class _AsyncEventChannel:
    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[ExecutionEvent | object] = asyncio.Queue()
        self.sink = _EventSink(self._publish)

    def _publish(self, event: ExecutionEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, _CLOSED)

    async def get(self) -> ExecutionEvent | None:
        item = await self._queue.get()
        return None if item is _CLOSED else item
