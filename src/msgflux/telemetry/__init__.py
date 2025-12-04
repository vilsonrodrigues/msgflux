"""
msgflux.telemetry - msgtrace-sdk integration.

Provides telemetry capabilities through msgtrace-sdk with msgflux-specific
decorators for detailed tool and agent instrumentation.
"""

from msgflux.telemetry.span import (
    Spans,
    aset_agent_attributes,
    aset_tool_attributes,
    set_agent_attributes,
    set_tool_attributes,
    spans,
)

__all__ = [
    "spans",
    "Spans",
    "set_tool_attributes",
    "aset_tool_attributes",
    "set_agent_attributes",
    "aset_agent_attributes",
]
