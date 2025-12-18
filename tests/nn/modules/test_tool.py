"""Tests for msgflux.nn.modules.tool module."""

import pytest
from unittest.mock import Mock, AsyncMock

from msgflux.nn.modules.tool import ToolCall, ToolResponses, Tool


class TestToolCall:
    """Test suite for ToolCall dataclass."""

    def test_tool_call_initialization(self):
        """Test ToolCall basic initialization."""
        tool_call = ToolCall(id="call_123", name="test_tool")
        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.parameters == {}
        assert tool_call.result is None
        assert tool_call.error is None

    def test_tool_call_with_parameters(self):
        """Test ToolCall with parameters."""
        params = {"arg1": "value1", "arg2": 42}
        tool_call = ToolCall(id="call_456", name="my_tool", parameters=params)
        assert tool_call.parameters == params

    def test_tool_call_with_result(self):
        """Test ToolCall with result."""
        tool_call = ToolCall(id="call_789", name="calculator", result={"sum": 10})
        assert tool_call.result == {"sum": 10}

    def test_tool_call_with_error(self):
        """Test ToolCall with error."""
        tool_call = ToolCall(id="call_err", name="broken_tool", error="Tool failed")
        assert tool_call.error == "Tool failed"


class TestToolResponses:
    """Test suite for ToolResponses dataclass."""

    def test_tool_responses_initialization(self):
        """Test ToolResponses basic initialization."""
        responses = ToolResponses(return_directly=False)
        assert responses.return_directly is False
        assert responses.tool_calls == []

    def test_tool_responses_with_calls(self):
        """Test ToolResponses with tool calls."""
        call1 = ToolCall(id="call_1", name="tool1", result="result1")
        call2 = ToolCall(id="call_2", name="tool2", result="result2")
        responses = ToolResponses(return_directly=True, tool_calls=[call1, call2])
        
        assert responses.return_directly is True
        assert len(responses.tool_calls) == 2
        assert responses.tool_calls[0].id == "call_1"
        assert responses.tool_calls[1].id == "call_2"

    def test_tool_responses_to_dict(self):
        """Test ToolResponses to_dict conversion."""
        call = ToolCall(id="call_x", name="toolx", parameters={"key": "val"})
        responses = ToolResponses(return_directly=False, tool_calls=[call])
        result_dict = responses.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["return_directly"] is False
        assert len(result_dict["tool_calls"]) == 1
        assert result_dict["tool_calls"][0]["id"] == "call_x"

    def test_tool_responses_to_json(self):
        """Test ToolResponses to_json conversion."""
        responses = ToolResponses(return_directly=True)
        result_json = responses.to_json()
        
        assert isinstance(result_json, bytes)

    def test_tool_responses_get_by_id(self):
        """Test ToolResponses get_by_id method."""
        call1 = ToolCall(id="call_abc", name="tool1")
        call2 = ToolCall(id="call_def", name="tool2")
        responses = ToolResponses(return_directly=False, tool_calls=[call1, call2])
        
        found = responses.get_by_id("call_abc")
        assert found is not None
        assert found.name == "tool1"
        
        not_found = responses.get_by_id("call_xyz")
        assert not_found is None

    def test_tool_responses_get_by_name(self):
        """Test ToolResponses get_by_name method."""
        call1 = ToolCall(id="call_1", name="calculator")
        call2 = ToolCall(id="call_2", name="search")
        responses = ToolResponses(return_directly=False, tool_calls=[call1, call2])
        
        found = responses.get_by_name("search")
        assert found is not None
        assert found.id == "call_2"
        
        not_found = responses.get_by_name("unknown")
        assert not_found is None


class TestTool:
    """Test suite for Tool base class."""

    def test_tool_inheritance(self):
        """Test that Tool inherits from Module."""
        from msgflux.nn.modules.module import Module
        assert issubclass(Tool, Module)

    def test_tool_get_json_schema(self):
        """Test Tool get_json_schema method."""
        class SimpleTool(Tool):
            """A simple tool for testing."""

            def forward(self, x: int) -> int:
                """Add one to x.

                Args:
                    x: The input number.

                Returns:
                    The input number plus one.
                """
                return x + 1

        tool = SimpleTool()
        schema = tool.get_json_schema()

        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "function"
