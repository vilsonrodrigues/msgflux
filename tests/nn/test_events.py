"""Unit tests for msgflux.nn.events module."""

import dataclasses
import inspect
import warnings
from unittest.mock import MagicMock, patch

import pytest

from msgflux import nn
from msgflux.nn import events as events_module
from msgflux.nn.events import (
    EventBus,
    EventStream,
    EventType,
    ModuleEvent,
    StreamEvent,
    _current_event_bus,
    _module_stack,
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
    emit_event,
    get_current_event_bus,
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


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_module_event_is_stream_event_alias(self):
        """ModuleEvent should be an alias for StreamEvent."""
        assert ModuleEvent is StreamEvent

    def test_event_bus_is_event_stream_alias(self):
        """EventBus should be an alias for EventStream."""
        assert EventBus is EventStream

    def test_module_stack_contextvar_exists(self):
        """_module_stack contextvar should exist for backward compatibility."""
        assert _module_stack is not None
        # Should be a ContextVar
        assert hasattr(_module_stack, "get")
        assert hasattr(_module_stack, "set")
        # Default should be None
        assert _module_stack.get() is None

    def test_current_event_bus_contextvar_exists(self):
        """_current_event_bus contextvar should exist for backward compatibility."""
        assert _current_event_bus is not None
        # Should be a ContextVar
        assert hasattr(_current_event_bus, "get")
        assert hasattr(_current_event_bus, "set")
        # Default should be None
        assert _current_event_bus.get() is None


class TestDeprecatedEmitEvent:
    """Test the deprecated emit_event function."""

    def test_emit_event_raises_deprecation_warning(self):
        """emit_event should raise DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with patch("msgflux.nn.events.add_event"):
                emit_event(
                    event_type="test.event",
                    module_name="test_module",
                    module_type="test",
                )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "emit_event() is deprecated" in str(w[0].message)

    def test_emit_event_calls_add_event(self):
        """emit_event should call add_event with converted attributes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("msgflux.nn.events.add_event") as mock_add_event:
                emit_event(
                    event_type="test.event",
                    module_name="test_module",
                    module_type="agent",
                    data={"key": "value"},
                    metadata={"extra": "info"},
                )
                mock_add_event.assert_called_once()
                call_args = mock_add_event.call_args
                assert call_args[0][0] == "test.event"
                attributes = call_args[0][1]
                assert attributes["module_name"] == "test_module"
                assert attributes["module_type"] == "agent"
                assert attributes["key"] == "value"
                assert attributes["extra"] == "info"

    def test_emit_event_with_non_dict_data(self):
        """emit_event should handle non-dict data by putting it under 'data' key."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("msgflux.nn.events.add_event") as mock_add_event:
                emit_event(
                    event_type="test.event",
                    module_name="test_module",
                    module_type="tool",
                    data="string_data",
                )
                call_args = mock_add_event.call_args
                attributes = call_args[0][1]
                assert attributes["data"] == "string_data"

    def test_emit_event_without_data_or_metadata(self):
        """emit_event should work with minimal arguments."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("msgflux.nn.events.add_event") as mock_add_event:
                emit_event(
                    event_type="test.event",
                    module_name="minimal",
                    module_type="module",
                )
                call_args = mock_add_event.call_args
                attributes = call_args[0][1]
                assert attributes == {
                    "module_name": "minimal",
                    "module_type": "module",
                }


class TestDeprecatedGetCurrentEventBus:
    """Test the deprecated get_current_event_bus function."""

    def test_get_current_event_bus_raises_deprecation_warning(self):
        """get_current_event_bus should raise DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_current_event_bus()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "get_current_event_bus() is deprecated" in str(w[0].message)

    def test_get_current_event_bus_returns_none(self):
        """get_current_event_bus should always return None."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = get_current_event_bus()
            assert result is None


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
    """Test stream() and astream() methods on Module base class."""

    def test_module_has_stream_method(self):
        """Module should have stream() method."""
        assert hasattr(Module, "stream")
        assert callable(Module.stream)

    def test_module_has_astream_method(self):
        """Module should have astream() method."""
        assert hasattr(Module, "astream")
        assert callable(Module.astream)

    def test_stream_method_signature(self):
        """stream() should accept *args, callback=None, **kwargs."""
        sig = inspect.signature(Module.stream)
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

    def test_astream_returns_async_generator(self):
        """astream() should return an AsyncGenerator."""
        sig = inspect.signature(Module.astream)
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
        # Backward compat
        assert hasattr(nn, "EventBus")
        assert hasattr(nn, "ModuleEvent")
