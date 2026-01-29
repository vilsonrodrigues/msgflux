"""Unit tests for msgflux.nn.events module."""

import dataclasses
import inspect
from unittest.mock import MagicMock

import pytest

from msgflux import nn
from msgflux.nn import events as events_module
from msgflux.nn.events import (
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
from msgflux.nn.modules.module import Module


class TestReExports:
    """Test that all re-exports from msgtrace-sdk are available."""

    def test_event_stream_available(self):
        """EventStream should be re-exported from msgtrace-sdk."""
        assert EventStream is not None
        # Should be a class/type
        assert isinstance(EventStream, type)

    def test_event_type_available(self):
        """EventType should be re-exported with expected constants."""
        assert EventType is not None
        # Check some expected constants exist
        assert hasattr(EventType, "AGENT_START")
        assert hasattr(EventType, "AGENT_COMPLETE")
        assert hasattr(EventType, "TOOL_CALL")
        assert hasattr(EventType, "TOOL_RESULT")
        assert hasattr(EventType, "MODULE_START")
        assert hasattr(EventType, "MODULE_COMPLETE")

    def test_stream_event_available(self):
        """StreamEvent should be re-exported from msgtrace-sdk."""
        assert StreamEvent is not None

    def test_add_event_available(self):
        """add_event function should be available."""
        assert callable(add_event)

    def test_convenience_functions_available(self):
        """All convenience functions should be available."""
        functions = [
            add_agent_start_event,
            add_agent_complete_event,
            add_agent_step_event,
            add_model_request_event,
            add_model_response_event,
            add_model_response_chunk_event,
            add_model_reasoning_event,
            add_tool_call_event,
            add_tool_result_event,
            add_tool_error_event,
            add_flow_step_event,
            add_flow_reasoning_event,
            add_flow_complete_event,
        ]
        for func in functions:
            assert callable(func), f"{func.__name__} should be callable"


class TestEventTypeConstants:
    """Test EventType constant values follow GenAI semantic conventions."""

    def test_agent_event_types(self):
        """Agent event types should follow gen_ai.agent.* pattern."""
        assert EventType.AGENT_START == "gen_ai.agent.start"
        assert EventType.AGENT_COMPLETE == "gen_ai.agent.complete"
        assert EventType.AGENT_STEP == "gen_ai.agent.step"
        assert EventType.AGENT_ERROR == "gen_ai.agent.error"

    def test_tool_event_types(self):
        """Tool event types should follow gen_ai.tool.* pattern."""
        assert EventType.TOOL_CALL == "gen_ai.tool.call"
        assert EventType.TOOL_RESULT == "gen_ai.tool.result"
        assert EventType.TOOL_ERROR == "gen_ai.tool.error"

    def test_model_event_types(self):
        """Model event types should follow gen_ai.model.* pattern."""
        assert EventType.MODEL_REQUEST == "gen_ai.model.request"
        assert EventType.MODEL_RESPONSE == "gen_ai.model.response"
        assert EventType.MODEL_RESPONSE_CHUNK == "gen_ai.model.response.chunk"
        assert EventType.MODEL_REASONING == "gen_ai.model.reasoning"

    def test_module_event_types(self):
        """Module event types should follow gen_ai.module.* pattern."""
        assert EventType.MODULE_START == "gen_ai.module.start"
        assert EventType.MODULE_COMPLETE == "gen_ai.module.complete"
        assert EventType.MODULE_ERROR == "gen_ai.module.error"

    def test_flow_event_types(self):
        """Flow event types should follow gen_ai.flow.* pattern."""
        assert EventType.FLOW_STEP == "gen_ai.flow.step"
        assert EventType.FLOW_REASONING == "gen_ai.flow.reasoning"
        assert EventType.FLOW_COMPLETE == "gen_ai.flow.complete"


class TestEventStreamContextManager:
    """Test EventStream as context manager."""

    @pytest.mark.asyncio
    async def test_event_stream_async_context_manager(self):
        """EventStream should work as async context manager."""
        async with EventStream() as stream:
            assert stream is not None
            # Stream should have close method
            assert hasattr(stream, "close")
            # Stream should be async iterable
            assert hasattr(stream, "__aiter__")

    def test_event_stream_sync_context_manager(self):
        """EventStream should work as sync context manager."""
        with EventStream() as stream:
            assert stream is not None
            assert hasattr(stream, "close")
            # Should have events property for collected events
            assert hasattr(stream, "events")

    def test_event_stream_on_event_callback(self):
        """EventStream should support on_event callback registration."""
        with EventStream() as stream:
            callback = MagicMock()
            stream.on_event(callback)
            # on_event should work without error


class TestModuleStreamMethods:
    """Test stream_events() and astream_events() methods on Module base class."""

    def test_module_has_stream_events_method(self):
        """Module should have stream_events() method."""
        assert hasattr(Module, "stream_events")
        assert callable(Module.stream_events)

    def test_module_has_astream_events_method(self):
        """Module should have astream_events() method."""
        assert hasattr(Module, "astream_events")
        assert callable(Module.astream_events)

    def test_stream_events_method_signature(self):
        """stream_events() should accept *args, callback=None, **kwargs."""
        sig = inspect.signature(Module.stream_events)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "callback" in params
        # Should have *args and **kwargs
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        assert has_var_positional
        assert has_var_keyword

    def test_astream_events_returns_async_generator(self):
        """astream_events() should return an AsyncGenerator."""
        sig = inspect.signature(Module.astream_events)
        # Return annotation should indicate AsyncGenerator
        return_annotation = sig.return_annotation
        # Check that it's annotated as AsyncGenerator
        assert "AsyncGenerator" in str(return_annotation) or "Generator" in str(
            return_annotation
        )


class TestStreamEventStructure:
    """Test StreamEvent dataclass structure."""

    def test_stream_event_is_frozen_dataclass(self):
        """StreamEvent should be a frozen dataclass."""
        assert dataclasses.is_dataclass(StreamEvent)
        # Check if frozen by trying to see if fields exist
        fields = {f.name for f in dataclasses.fields(StreamEvent)}
        # Should have expected fields
        assert "name" in fields
        assert "attributes" in fields
        assert "timestamp_ns" in fields


class TestAllExports:
    """Test __all__ exports are complete."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        for name in events_module.__all__:
            assert hasattr(events_module, name), f"{name} in __all__ but not in module"

    def test_nn_package_exports_events(self):
        """msgflux.nn should export event-related items."""
        assert hasattr(nn, "EventStream")
        assert hasattr(nn, "EventType")
        assert hasattr(nn, "StreamEvent")
        assert hasattr(nn, "add_event")
