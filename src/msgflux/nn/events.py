"""Event streaming system for nn.Modules.

Provides real-time event capture during module execution via two
complementary mechanisms:

1. **EventBus** (infrastructure): A context-variable-based event bus
   that captures events emitted during normal ``forward()`` execution.
   Zero overhead when no bus is active.

2. **``astream()``/``stream()``** (convenience): Methods on ``Module``
   that wrap ``acall()`` and yield events as an
   ``AsyncGenerator[ModuleEvent]``.

Example usage::

    import msgflux.nn as nn
    from msgflux.nn.events import EventBus, EventType

    agent = nn.Agent("researcher", model=model, tools=[search])

    # Via astream
    async for event in agent.astream("What is 2+2?"):
        print(event.event_type, event.data)

    # Via EventBus context manager
    with EventBus.listen(callback=lambda e: print(e.event_type)):
        result = agent("What is 2+2?")
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class EventType:
    """Constants for event types emitted during module execution."""

    # Agent lifecycle
    AGENT_START = "agent.start"
    AGENT_STEP = "agent.step"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"

    # Model
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MODEL_RESPONSE_CHUNK = "model.response.chunk"
    MODEL_REASONING = "model.reasoning"
    MODEL_REASONING_CHUNK = "model.reasoning.chunk"

    # Tool
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    # Flow control (ReAct, CoT)
    FLOW_STEP = "flow.step"
    FLOW_REASONING = "flow.reasoning"
    FLOW_COMPLETE = "flow.complete"

    # Module lifecycle
    MODULE_START = "module.start"
    MODULE_COMPLETE = "module.complete"
    MODULE_ERROR = "module.error"


@dataclass(frozen=True)
class ModuleEvent:
    """An event emitted during module execution.

    Attributes:
        event_type: Classification of the event
            (an ``EventType`` constant).
        module_name: Name of the module that emitted this event.
        module_type: Type of module
            (``"agent"``, ``"tool"``, ``"lm"``, etc.).
        data: Event-specific payload
            (tool name, response chunk, etc.).
        timestamp: When the event occurred (seconds since epoch).
        parent_module: Name of the parent module for hierarchical
            sub-agent events.
        metadata: Additional key-value metadata.
    """

    event_type: str
    module_name: str
    module_type: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    parent_module: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Context variables for implicit propagation
# ---------------------------------------------------------------------------

_current_event_bus: contextvars.ContextVar[EventBus | None] = contextvars.ContextVar(
    "_current_event_bus", default=None
)

_module_stack: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_module_stack", default=None
)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Collects and distributes events during module execution.

    Events are emitted by modules via :func:`emit_event` during
    ``forward()``. The bus can be consumed via callbacks, async
    iteration, or post-hoc collection.
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callable]] = {}
        self._all_callbacks: list[Callable] = []
        self._events: list[ModuleEvent] = []
        self._queue: asyncio.Queue | None = None
        self._collect: bool = False

    # -- Registration ---------------------------------------------------

    def on(self, event_type: str) -> Callable:
        """Decorator to register a callback for a specific event type.

        Example::

            @bus.on(EventType.TOOL_CALL)
            def handle_tool(event):
                print(event.data["tool_name"])
        """

        def decorator(func: Callable) -> Callable:
            self._callbacks.setdefault(event_type, []).append(func)
            return func

        return decorator

    def on_all(self, callback: Callable) -> None:
        """Register a callback invoked for every event."""
        self._all_callbacks.append(callback)

    # -- Emission -------------------------------------------------------

    def emit(self, event: ModuleEvent) -> None:
        """Emit an event to all registered consumers.

        Called internally by :func:`emit_event`. Users should not
        call this directly.
        """
        if self._collect:
            self._events.append(event)

        for cb in self._callbacks.get(event.event_type, []):
            cb(event)

        for cb in self._all_callbacks:
            cb(event)

        if self._queue is not None:
            self._queue.put_nowait(event)

    # -- Consumption ----------------------------------------------------

    @property
    def events(self) -> list[ModuleEvent]:
        """All collected events (requires ``collect=True``)."""
        return self._events

    async def __aiter__(self):
        """Async iteration over events as they arrive."""
        if self._queue is None:
            self._queue = asyncio.Queue()
        while True:
            event = await self._queue.get()
            if event is None:  # Sentinel for end-of-stream
                break
            yield event

    def close(self) -> None:
        """Signal end of event stream."""
        if self._queue is not None:
            self._queue.put_nowait(None)

    # -- Context managers -----------------------------------------------

    @classmethod
    def capture(cls, *, collect: bool = True) -> _EventBusContext:
        """Context manager that activates an ``EventBus``.

        Args:
            collect: If ``True``, store events in ``bus.events``
                for post-hoc access.

        Example::

            async with EventBus.capture() as bus:
                result = await agent.acall("hello")
            print(bus.events)
        """
        return _EventBusContext(collect=collect)

    @classmethod
    def listen(cls, callback: Callable | None = None) -> _EventBusContext:
        """Convenience context manager: capture + callback.

        Example::

            with EventBus.listen(callback=print):
                result = agent("hello")
        """
        return _EventBusContext(callback=callback)


class _EventBusContext:
    """Context manager for :class:`EventBus` activation."""

    def __init__(
        self,
        *,
        collect: bool = False,
        callback: Callable | None = None,
    ) -> None:
        self.bus = EventBus()
        self.bus._collect = collect or (callback is not None)
        if callback is not None:
            self.bus.on_all(callback)
        self._token: contextvars.Token | None = None

    # Sync context manager
    def __enter__(self) -> EventBus:
        self._token = _current_event_bus.set(self.bus)
        return self.bus

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _current_event_bus.reset(self._token)
        self.bus.close()

    # Async context manager
    async def __aenter__(self) -> EventBus:
        self._token = _current_event_bus.set(self.bus)
        return self.bus

    async def __aexit__(self, *exc) -> None:
        if self._token is not None:
            _current_event_bus.reset(self._token)
        self.bus.close()


# ---------------------------------------------------------------------------
# Convenience emission function
# ---------------------------------------------------------------------------


def emit_event(
    event_type: str,
    module_name: str,
    module_type: str,
    data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit a :class:`ModuleEvent` if an :class:`EventBus` is active.

    This is a **no-op** when no ``EventBus`` is set in the current
    context, resulting in negligible overhead (~20 ns for the
    ``ContextVar.get()`` check).

    Args:
        event_type: One of the :class:`EventType` constants.
        module_name: Name of the emitting module.
        module_type: Type of the emitting module.
        data: Event-specific payload.
        metadata: Optional extra metadata.
    """
    bus = _current_event_bus.get()
    if bus is None:
        return

    stack = _module_stack.get()
    parent = stack[-1] if stack else None

    event = ModuleEvent(
        event_type=event_type,
        module_name=module_name,
        module_type=module_type,
        data=data,
        parent_module=parent,
        metadata=metadata or {},
    )
    bus.emit(event)


def get_current_event_bus() -> EventBus | None:
    """Return the currently active ``EventBus``, or ``None``."""
    return _current_event_bus.get()
