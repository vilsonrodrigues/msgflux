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
]
