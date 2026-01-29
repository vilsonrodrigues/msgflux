"""Event streaming system for nn.Modules.

This module re-exports the event streaming infrastructure from msgtrace-sdk,
providing real-time event capture during module execution.

The event system uses OpenTelemetry span events for dual emission:
1. **span.add_event()** - Events are persisted to OTel backends (Jaeger, etc.)
2. **Streaming queue** - Events are streamed in real-time to astream() consumers

Example usage::

    import msgflux.nn as nn
    from msgflux.nn import EventType, EventStream

    agent = nn.Agent("researcher", model=model, tools=[search])

    # Via astream - yields StreamEvent objects in real-time
    async for event in agent.astream("What is 2+2?"):
        print(event.name, event.attributes)

    # Via EventStream context manager
    async with EventStream() as stream:
        task = asyncio.create_task(agent.acall("Hello"))
        async for event in stream:
            print(event.name)
        await task
"""

from __future__ import annotations

import contextvars
import warnings
from typing import Any

# Re-export from msgtrace-sdk
from msgtrace.sdk import (
    EventStream,
    EventType,
    StreamEvent,
    add_agent_complete_event,
    add_agent_start_event,
    add_agent_step_event,
    add_event,
    add_flow_complete_event,
    add_flow_reasoning_event,
    add_flow_step_event,
    add_model_reasoning_event,
    add_model_request_event,
    add_model_response_chunk_event,
    add_model_response_event,
    add_tool_call_event,
    add_tool_error_event,
    add_tool_result_event,
)

# Alias for backward compatibility
ModuleEvent = StreamEvent

# Legacy EventBus compatibility - deprecated
EventBus = EventStream

# Explicit exports for documentation and IDE support
__all__ = [
    # Core types
    "EventStream",
    "EventType",
    "StreamEvent",
    # Convenience functions
    "add_event",
    "add_agent_start_event",
    "add_agent_complete_event",
    "add_agent_step_event",
    "add_model_request_event",
    "add_model_response_event",
    "add_model_response_chunk_event",
    "add_model_reasoning_event",
    "add_tool_call_event",
    "add_tool_result_event",
    "add_tool_error_event",
    "add_flow_step_event",
    "add_flow_reasoning_event",
    "add_flow_complete_event",
    # Backward compatibility
    "ModuleEvent",
    "EventBus",
    "emit_event",
    "get_current_event_bus",
    # Internal (backward compatibility)
    "_module_stack",
    "_current_event_bus",
]


def emit_event(
    event_type: str,
    module_name: str,
    module_type: str,
    data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Emit an event using the msgtrace-sdk add_event function.

    .. deprecated::
        Use :func:`add_event` or the convenience functions
        (e.g., :func:`add_tool_call_event`) directly instead.

    This function is provided for backward compatibility during migration.
    It converts the old emit_event signature to the new add_event format.

    Args:
        event_type: One of the :class:`EventType` constants.
        module_name: Name of the emitting module.
        module_type: Type of the emitting module.
        data: Event-specific payload.
        metadata: Optional extra metadata.
    """
    warnings.warn(
        "emit_event() is deprecated. Use add_event() or convenience functions "
        "(add_tool_call_event, add_agent_start_event, etc.) from msgtrace.sdk instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Build attributes from old format
    attributes: dict[str, Any] = {
        "module_name": module_name,
        "module_type": module_type,
    }
    if data is not None:
        if isinstance(data, dict):
            attributes.update(data)
        else:
            attributes["data"] = data
    if metadata:
        attributes.update(metadata)

    add_event(event_type, attributes)


def get_current_event_bus() -> EventStream | None:
    """Return the currently active EventStream, or None.

    .. deprecated::
        This function is deprecated. EventStream now manages its own
        context via contextvars internally.
    """
    warnings.warn(
        "get_current_event_bus() is deprecated. EventStream manages its own context.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


# For backward compatibility with code that imports _module_stack
# This is now a no-op since hierarchy is tracked via OTel span parent-child
_module_stack: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "_module_stack", default=None
)

_current_event_bus: contextvars.ContextVar[EventStream | None] = contextvars.ContextVar(
    "_current_event_bus", default=None
)
