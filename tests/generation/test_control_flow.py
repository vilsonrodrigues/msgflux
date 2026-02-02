"""Tests for control flow functionality."""

from typing import Any, List, Mapping, Optional
from unittest.mock import MagicMock

import pytest

from msgflux.generation.control_flow import FlowControl, FlowResult


class TestFlowResult:
    """Tests for FlowResult dataclass."""

    def test_flow_result_creation_complete(self):
        """Test creating a complete FlowResult."""
        result = FlowResult(
            is_complete=True,
            tool_calls=None,
            reasoning=None,
            final_response={"answer": "test"},
        )
        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.final_response == {"answer": "test"}

    def test_tool_flow_result_creation_with_tool_calls(self):
        """Test creating a FlowResult with tool calls."""
        tool_calls = [
            ("id1", "search", {"query": "test"}),
            ("id2", "calculate", {"a": 1, "b": 2}),
        ]
        result = FlowResult(
            is_complete=False,
            tool_calls=tool_calls,
            reasoning="Need to search first",
            final_response=None,
        )
        assert result.is_complete is False
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0] == ("id1", "search", {"query": "test"})
        assert result.reasoning == "Need to search first"
        assert result.final_response is None


class TestFlowControl:
    """Tests for FlowControl base class."""

    def test_tool_flow_control_is_class(self):
        """Test that FlowControl exists and is a class."""
        assert isinstance(FlowControl, type)

    def test_tool_flow_control_can_be_inherited(self):
        """Test that FlowControl can be inherited with required methods."""

        class CustomControl(FlowControl):
            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> FlowResult:
                return FlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_tool_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        assert issubclass(CustomControl, FlowControl)

    def test_tool_flow_control_class_attributes(self):
        """Test that FlowControl has class attributes."""
        assert hasattr(FlowControl, "system_message")
        assert hasattr(FlowControl, "tools_template")
        assert FlowControl.system_message is None
        assert FlowControl.tools_template is None

    def test_tool_flow_control_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            FlowControl.extract_flow_result({})

        with pytest.raises(NotImplementedError):
            FlowControl.inject_tool_results({}, MagicMock())

        with pytest.raises(NotImplementedError):
            FlowControl.build_history({}, [])


class TestCustomFlowControl:
    """Tests for custom FlowControl implementation."""

    def test_simple_tool_loop(self):
        """Test a simple custom tool flow control."""

        class SimpleToolLoop(FlowControl):
            """Simple tool loop without ReAct structure."""

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> FlowResult:
                if raw_response.get("done"):
                    return FlowResult(
                        is_complete=True,
                        tool_calls=None,
                        reasoning=None,
                        final_response={"done": True},
                    )

                calls = raw_response.get("calls", [])
                if calls:
                    tool_calls = [
                        (f"id_{i}", call["name"], call["args"])
                        for i, call in enumerate(calls)
                    ]
                    return FlowResult(
                        is_complete=False,
                        tool_calls=tool_calls,
                        reasoning="Processing calls",
                        final_response=None,
                    )

                return FlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response={"done": True, "no_calls": True},
                )

            @classmethod
            def inject_tool_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                calls = raw_response.get("calls", [])
                for i, call in enumerate(calls):
                    call_id = f"id_{i}"
                    result = tool_results.get_by_id(call_id)
                    if result:
                        call["result"] = result.result
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                messages.append(
                    {"role": "assistant", "content": str(raw_response)}
                )
                return messages

        # Test completed state
        result = SimpleToolLoop.extract_flow_result({"done": True})
        assert result.is_complete is True

        # Test with pending calls
        raw = {"calls": [{"name": "search", "args": {"q": "test"}}]}
        result = SimpleToolLoop.extract_flow_result(raw)
        assert result.is_complete is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][1] == "search"

        # Test build_history
        messages = []
        messages = SimpleToolLoop.build_history(raw, messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"


class TestFlowControlAsync:
    """Tests for async methods of FlowControl."""

    @pytest.mark.asyncio
    async def test_async_methods_default_to_sync(self):
        """Test that async methods default to calling sync versions."""

        class CustomControl(FlowControl):
            sync_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> FlowResult:
                cls.sync_called = True
                return FlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_tool_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        # Call async version
        result = await CustomControl.aextract_flow_result({})

        # Verify sync was called
        assert CustomControl.sync_called is True
        assert result.is_complete is True

    @pytest.mark.asyncio
    async def test_async_inject_tool_results(self):
        """Test async inject_tool_results defaults to sync."""

        class CustomControl(FlowControl):
            inject_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> FlowResult:
                return FlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_tool_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                cls.inject_called = True
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                return messages

        await CustomControl.ainject_tool_results({}, MagicMock())
        assert CustomControl.inject_called is True

    @pytest.mark.asyncio
    async def test_async_build_history(self):
        """Test async build_history defaults to sync."""

        class CustomControl(FlowControl):
            history_called = False

            @classmethod
            def extract_flow_result(
                cls, raw_response: Mapping[str, Any]
            ) -> FlowResult:
                return FlowResult(
                    is_complete=True,
                    tool_calls=None,
                    reasoning=None,
                    final_response=None,
                )

            @classmethod
            def inject_tool_results(
                cls, raw_response: Mapping[str, Any], tool_results
            ) -> Mapping[str, Any]:
                return raw_response

            @classmethod
            def build_history(
                cls, raw_response: Mapping[str, Any], messages: List[Mapping[str, Any]]
            ) -> List[Mapping[str, Any]]:
                cls.history_called = True
                messages.append({"role": "test"})
                return messages

        result = await CustomControl.abuild_history({}, [])
        assert CustomControl.history_called is True
        assert len(result) == 1
