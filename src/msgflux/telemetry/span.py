import asyncio
import hashlib
import os
import platform
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import msgspec
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from msgflux.envs import envs
from msgflux.models.response import ModelStreamResponse
from msgflux.telemetry.attributes import GenAIAttributes, MsgTraceAttributes
from msgflux.telemetry.tracer import get_tracer
from msgflux.version import __version__ as msgflux_version


class Spans:
    def __init__(self):
        self.tracer = get_tracer()

    @contextmanager
    def span_context(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = SpanKind.INTERNAL,
    ):
        """Generic context manager to create and manage a span."""
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    @contextmanager
    def init_flow(
        self,
        module_name: str,
        module_type: str,
        encoded_state_dict: Optional[bytes] = None,
    ):
        attributes = {}
        attributes[MsgTraceAttributes.SERVICE_NAME] = "msgflux"
        attributes[MsgTraceAttributes.SERVICE_VERSION] = msgflux_version
        attributes[MsgTraceAttributes.MODULE_NAME] = module_name
        attributes[MsgTraceAttributes.MODULE_TYPE] = module_type
        if encoded_state_dict:
            attributes["msgflux.state_dict"] = encoded_state_dict
        if envs.telemetry_capture_platform:
            attributes[MsgTraceAttributes.PLATFORM_NAME] = platform.platform()
            attributes[MsgTraceAttributes.PLATFORM_VERSION] = platform.version()
            attributes[MsgTraceAttributes.PLATFORM_NUM_CPUS] = os.cpu_count()
            attributes[MsgTraceAttributes.PLATFORM_PYTHON_VERSION] = platform.python_version()            

        with self.span_context(module_name, attributes) as span:
            yield span

    @contextmanager
    def init_module(self, module_name: str, module_type: str):
        attributes = {}
        attributes[MsgTraceAttributes.MODULE_NAME] = module_name
        attributes[MsgTraceAttributes.MODULE_TYPE] = module_type
        with self.span_context(module_name, attributes) as span:
            yield span

    @asynccontextmanager
    async def aspan_context(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[str] = SpanKind.INTERNAL,
    ):
        """Async generic context manager to create and manage a span."""
        with self.tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    @asynccontextmanager
    async def ainit_flow(
        self,
        module_name: str,
        module_type: str,
        encoded_state_dict: Optional[bytes] = None,
    ):
        attributes = {}
        attributes[MsgTraceAttributes.SERVICE_NAME] = "msgflux"
        attributes[MsgTraceAttributes.SERVICE_VERSION] = msgflux_version
        attributes[MsgTraceAttributes.MODULE_NAME] = module_name
        attributes[MsgTraceAttributes.MODULE_TYPE] = module_type
        if encoded_state_dict:
            attributes["msgflux.state_dict"] = encoded_state_dict
        if envs.telemetry_capture_platform:
            attributes[MsgTraceAttributes.PLATFORM_NAME] = platform.platform()
            attributes[MsgTraceAttributes.PLATFORM_VERSION] = platform.version()
            attributes[MsgTraceAttributes.PLATFORM_NUM_CPUS] = os.cpu_count()
            attributes[MsgTraceAttributes.PLATFORM_PYTHON_VERSION] = platform.python_version()            

        async with self.aspan_context(module_name, attributes) as span:
            yield span

    @asynccontextmanager
    async def ainit_module(self, module_name: str, module_type: str):
        attributes = {}
        attributes[MsgTraceAttributes.MODULE_NAME] = module_name
        attributes[MsgTraceAttributes.MODULE_TYPE] = module_type
        async with self.aspan_context(module_name, attributes) as span:
            yield span

spans = Spans()


def instrument(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> Callable:
    """Decorator that instruments a function with a tracing span.

    This decorator creates a span using the provided name and attributes
    when the decorated function is called. Supports both sync and async functions.
    If an exception occurs during execution, the exception is recorded in the
    span and the span status is set to error.

    When telemetry is disabled (envs.telemetry_requires_trace=False), this
    decorator adds zero overhead by returning the function result directly.

    Args:
        name:
            The name of the span. If not provided, the function's name is used.
        attributes:
            A dict of attributes to attach to the span.
            Useful for adding metadata for tracing purposes.

    Returns:
        A decorated function that is wrapped with span instrumentation.

    Raises:
        Exception: Reraises any exception thrown by the wrapped function after
            recording it in the span.
    """

    def decorator(func):
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Early return for zero overhead when telemetry is disabled
                if not envs.telemetry_requires_trace:
                    return await func(*args, **kwargs)

                with spans.span_context(
                    name or func.__name__, attributes=attributes
                ) as span:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Early return for zero overhead when telemetry is disabled
                if not envs.telemetry_requires_trace:
                    return func(*args, **kwargs)

                with spans.span_context(
                    name or func.__name__, attributes=attributes
                ) as span:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            return sync_wrapper

    return decorator


def set_tool_attributes(execution_type: str, protocol: Optional[str] = None) -> Callable:
    """Decorator to set tool telemetry attributes for sync methods."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, **kwargs):
            # Extract tool_call_id from kwargs (passed by ToolLibrary)
            tool_call_id = kwargs.pop("tool_call_id", None)

            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute(GenAIAttributes.OPERATION_NAME, "execute_tool")
                span.set_attribute(GenAIAttributes.TOOL_NAME, self.name)
                if hasattr(self, "description") and self.description:
                    span.set_attribute(GenAIAttributes.TOOL_DESCRIPTION, self.description)
                span.set_attribute(GenAIAttributes.TOOL_TYPE, "function")

                # Set tool call ID if available
                if tool_call_id:
                    span.set_attribute(GenAIAttributes.TOOL_CALL_ID, tool_call_id)

                # Set arguments (without tool_call_id)
                try:
                    encoded_args = msgspec.json.encode(kwargs)
                    span.set_attribute(GenAIAttributes.TOOL_CALL_ARGS, encoded_args)
                except (TypeError, ValueError):
                    pass

                span.set_attribute(MsgTraceAttributes.TOOL_EXECUTION_TYPE, execution_type)
                if protocol:
                    span.set_attribute(MsgTraceAttributes.TOOL_PROTOCOL, protocol)

            # Execute the actual function
            response = func(self, **kwargs)

            # Capture result
            span = trace.get_current_span()
            if span.is_recording():
                try:
                    if isinstance(response, (str, int, float, bool)):
                        span.set_attribute(GenAIAttributes.TOOL_CALL_RESULT, str(response))
                    else:
                        encoded_response = msgspec.json.encode(response)
                        span.set_attribute(GenAIAttributes.TOOL_CALL_RESULT, encoded_response)
                except (TypeError, ValueError):
                    pass

            return response
        return wrapper
    return decorator


def aset_tool_attributes(execution_type: str, protocol: Optional[str] = None) -> Callable:
    """Decorator to set tool telemetry attributes for async methods."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Extract tool_call_id from kwargs (passed by ToolLibrary)
            tool_call_id = kwargs.pop("tool_call_id", None)

            span = trace.get_current_span()
            if span.is_recording():
                span.set_attribute(GenAIAttributes.OPERATION_NAME, "execute_tool")
                span.set_attribute(GenAIAttributes.TOOL_NAME, self.name)
                if hasattr(self, "description") and self.description:
                    span.set_attribute(GenAIAttributes.TOOL_DESCRIPTION, self.description)
                span.set_attribute(GenAIAttributes.TOOL_TYPE, "function")

                # Set tool call ID if available
                if tool_call_id:
                    span.set_attribute(GenAIAttributes.TOOL_CALL_ID, tool_call_id)

                # Set arguments (without tool_call_id)
                try:
                    encoded_args = msgspec.json.encode(kwargs)
                    span.set_attribute(GenAIAttributes.TOOL_CALL_ARGS, encoded_args)
                except (TypeError, ValueError):
                    pass

                span.set_attribute(MsgTraceAttributes.TOOL_EXECUTION_TYPE, execution_type)
                if protocol:
                    span.set_attribute(MsgTraceAttributes.TOOL_PROTOCOL, protocol)

            # Execute the actual function
            response = await func(self, *args, **kwargs)

            # Capture result
            span = trace.get_current_span()
            if span.is_recording():
                try:
                    if isinstance(response, (str, int, float, bool)):
                        span.set_attribute(GenAIAttributes.TOOL_CALL_RESULT, str(response))
                    else:
                        encoded_response = msgspec.json.encode(response)
                        span.set_attribute(GenAIAttributes.TOOL_CALL_RESULT, encoded_response)
                except (TypeError, ValueError):
                    pass

            return response
        return wrapper
    return decorator


def set_agent_attributes(func: Callable) -> Callable:
    """Decorator to set agent telemetry attributes for sync methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        span = trace.get_current_span()
        if span.is_recording():
            # Set agent name
            if hasattr(self, "name") and self.name:
                span.set_attribute(GenAIAttributes.AGENT_NAME, self.name)

                # Generate agent ID from name hash
                agent_id = f"asst_{hashlib.sha256(self.name.encode()).hexdigest()[:24]}"
                span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)

            # Set agent description if available
            if hasattr(self, "description") and self.description:
                span.set_attribute(GenAIAttributes.AGENT_DESCRIPTION, self.description)

        # Execute the actual function
        response = func(self, *args, **kwargs)
        # Capture result
        span = trace.get_current_span()
        if span.is_recording() and not isinstance(response, ModelStreamResponse):
            span.set_attribute(MsgTraceAttributes.AGENT_RESPONSE, response)
        return response

    return wrapper


def aset_agent_attributes(func: Callable) -> Callable:
    """Decorator to set agent telemetry attributes for async methods."""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        span = trace.get_current_span()
        if span.is_recording():
            # Set agent name
            if hasattr(self, "name") and self.name:
                span.set_attribute(GenAIAttributes.AGENT_NAME, self.name)

                # Generate agent ID from name hash
                agent_id = f"asst_{hashlib.sha256(self.name.encode()).hexdigest()[:24]}"
                span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)

            # Set agent description if available
            if hasattr(self, "description") and self.description:
                span.set_attribute(GenAIAttributes.AGENT_DESCRIPTION, self.description)

        # Execute the actual function
        response = await func(self, *args, **kwargs)

        # Capture result
        span = trace.get_current_span()
        if span.is_recording() and not isinstance(response, ModelStreamResponse):
            span.set_attribute(MsgTraceAttributes.AGENT_RESPONSE, response)
        return response

    return wrapper
