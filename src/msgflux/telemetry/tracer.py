"""
msgflux telemetry tracer - msgtrace-sdk integration.

This module provides a tracer instance using msgtrace-sdk for
consistent telemetry across msgflux operations.
"""

from msgtrace.sdk import tracer as msgtrace_tracer

# Re-export msgtrace-sdk tracer for msgflux use
tracer = msgtrace_tracer


def get_tracer():
    """Get the configured tracer instance.

    Returns:
        The msgtrace-sdk tracer instance.
    """
    return tracer
