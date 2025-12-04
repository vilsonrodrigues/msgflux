"""
msgflux telemetry attributes - msgtrace-sdk integration.

Re-exports MsgTraceAttributes from msgtrace-sdk with all
standard GenAI attributes and msgflux-specific extensions.
"""

from msgtrace.sdk import MsgTraceAttributes

__all__ = ["MsgTraceAttributes"]
