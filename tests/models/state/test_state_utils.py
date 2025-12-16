"""Tests for ModelState utility functions.

This module provides comprehensive test coverage for the utility functions in
msgflux.models.state.utils, achieving 100% line coverage.

Test coverage includes:
- chatml_to_model_state: Converting ChatML format to ModelState
- ensure_model_state: Type checking and conversion of ModelState objects
- get_tool_lifecycle_configs: Extracting tool lifecycle configurations

Coverage improved from 17% to 100% for utils.py (lines 34-74, 96-101, 127-145).
"""

import pytest

from msgflux.models.state import (
    ChatMessage,
    ModelState,
    Policy,
    Role,
    ToolCall,
)
from msgflux.models.state.utils import (
    chatml_to_model_state,
    ensure_model_state,
    get_tool_lifecycle_configs,
)


class TestChatMLToModelState:
    """Tests for chatml_to_model_state function."""

    def test_convert_simple_messages(self):
        """Test converting simple user and assistant messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        model_state = chatml_to_model_state(messages)

        assert model_state.message_count == 2
        assert model_state.messages[0].role == Role.USER
        assert model_state.messages[0].text == "Hello"
        assert model_state.messages[1].role == Role.ASSISTANT
        assert model_state.messages[1].text == "Hi there!"

    def test_convert_with_policy(self):
        """Test converting messages with a policy."""
        messages = [{"role": "user", "content": "Hello"}]
        policy = Policy(type="sliding_window", max_messages=10)

        model_state = chatml_to_model_state(messages, policy=policy)

        assert model_state.message_count == 1
        assert model_state._policy is not None

    def test_convert_assistant_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        messages = [
            {"role": "user", "content": "Calculate 2+2"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expr": "2+2"}',
                        },
                    }
                ],
            },
        ]

        model_state = chatml_to_model_state(messages)

        assert model_state.message_count == 2
        assistant_msg = model_state.messages[1]
        assert assistant_msg.role == Role.ASSISTANT
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0].id == "call_123"
        assert assistant_msg.tool_calls[0].name == "calculator"
        assert assistant_msg.tool_calls[0].arguments == {"expr": "2+2"}

    def test_convert_assistant_with_multiple_tool_calls(self):
        """Test converting assistant message with multiple tool calls."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me help with that",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "fetch", "arguments": '{"url": "example.com"}'},
                    },
                ],
            }
        ]

        model_state = chatml_to_model_state(messages)

        assistant_msg = model_state.messages[0]
        assert len(assistant_msg.tool_calls) == 2
        assert assistant_msg.tool_calls[0].name == "search"
        assert assistant_msg.tool_calls[1].name == "fetch"

    def test_convert_tool_calls_with_dict_arguments(self):
        """Test converting tool calls with dict arguments (not string)."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test", "limit": 5},
                        },
                    }
                ],
            }
        ]

        model_state = chatml_to_model_state(messages)

        assistant_msg = model_state.messages[0]
        assert assistant_msg.tool_calls[0].arguments == {"query": "test", "limit": 5}

    def test_convert_tool_calls_with_invalid_json_arguments(self):
        """Test converting tool calls with invalid JSON arguments."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": "invalid json {",
                        },
                    }
                ],
            }
        ]

        model_state = chatml_to_model_state(messages)

        assistant_msg = model_state.messages[0]
        # Should fall back to empty dict on parse error
        assert assistant_msg.tool_calls[0].arguments == {}

    def test_convert_tool_result_message(self):
        """Test converting tool result messages."""
        messages = [
            {"role": "user", "content": "Calculate"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "calc", "arguments": '{"x": 1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "Result: 42"},
        ]

        model_state = chatml_to_model_state(messages)

        assert model_state.message_count == 3
        tool_msg = model_state.messages[2]
        assert tool_msg.role == Role.TOOL
        assert tool_msg.tool_result.call_id == "call_123"
        assert tool_msg.tool_result.content == "Result: 42"

    def test_convert_system_messages_ignored(self):
        """Test that system messages are skipped."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        model_state = chatml_to_model_state(messages)

        # System message should be skipped
        assert model_state.message_count == 2
        assert model_state.messages[0].role == Role.USER
        assert model_state.messages[1].role == Role.ASSISTANT

    def test_convert_empty_messages_list(self):
        """Test converting empty messages list."""
        messages = []

        model_state = chatml_to_model_state(messages)

        assert model_state.message_count == 0

    def test_convert_messages_without_content_field(self):
        """Test converting messages without content field."""
        messages = [
            {"role": "user"},  # No content field
            {"role": "assistant"},  # No content field
        ]

        model_state = chatml_to_model_state(messages)

        assert model_state.message_count == 2
        assert model_state.messages[0].text == ""
        assert model_state.messages[1].text == ""

    def test_convert_tool_calls_with_missing_id(self):
        """Test converting tool calls with missing id."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {"name": "test", "arguments": "{}"},
                        # Missing 'id'
                    }
                ],
            }
        ]

        model_state = chatml_to_model_state(messages)

        assistant_msg = model_state.messages[0]
        assert assistant_msg.tool_calls[0].id == ""

    def test_convert_tool_calls_with_missing_function_name(self):
        """Test converting tool calls with missing function name."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"arguments": "{}"},
                        # Missing 'name'
                    }
                ],
            }
        ]

        model_state = chatml_to_model_state(messages)

        assistant_msg = model_state.messages[0]
        assert assistant_msg.tool_calls[0].name == ""

    def test_convert_tool_result_without_call_id(self):
        """Test converting tool result without tool_call_id."""
        messages = [
            {"role": "tool", "content": "Some result"},
            # Missing tool_call_id
        ]

        model_state = chatml_to_model_state(messages)

        tool_msg = model_state.messages[0]
        assert tool_msg.tool_result.call_id == ""


class TestEnsureModelState:
    """Tests for ensure_model_state function."""

    def test_ensure_with_existing_model_state(self):
        """Test that existing ModelState is passed through unchanged."""
        original = ModelState()
        original.add_user("Hello")

        result = ensure_model_state(original)

        assert result is original
        assert result.message_count == 1

    def test_ensure_with_dict_list(self):
        """Test converting list of dicts to ModelState."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = ensure_model_state(messages)

        assert isinstance(result, ModelState)
        assert result.message_count == 2
        assert result.messages[0].text == "Hello"
        assert result.messages[1].text == "Hi"

    def test_ensure_with_dict_list_and_policy(self):
        """Test converting list of dicts with policy."""
        messages = [{"role": "user", "content": "Hello"}]
        policy = Policy(type="sliding_window", max_messages=5)

        result = ensure_model_state(messages, policy=policy)

        assert isinstance(result, ModelState)
        assert result._policy is not None

    def test_ensure_with_empty_list(self):
        """Test converting empty list."""
        result = ensure_model_state([])

        assert isinstance(result, ModelState)
        assert result.message_count == 0

    def test_ensure_with_invalid_type_raises_error(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Expected ModelState or List\\[Dict\\]"):
            ensure_model_state("invalid string")

        with pytest.raises(TypeError, match="Expected ModelState or List\\[Dict\\]"):
            ensure_model_state(123)

        with pytest.raises(TypeError, match="Expected ModelState or List\\[Dict\\]"):
            ensure_model_state(None)

        with pytest.raises(TypeError, match="Expected ModelState or List\\[Dict\\]"):
            ensure_model_state({"role": "user", "content": "Hello"})


class TestGetToolLifecycleConfigs:
    """Tests for get_tool_lifecycle_configs function."""

    def test_get_configs_with_tool_config(self):
        """Test extracting lifecycle configs from tools with tool_config."""

        # Create a mock tool with tool_config
        class MockTool:
            def __init__(self):
                self.tool_config = type(
                    "Config",
                    (),
                    {
                        "ephemeral": True,
                        "ephemeral_ttl": 5,
                        "result_importance": 0.8,
                        "summarize_result": True,
                    },
                )()

        # Create a mock ToolLibrary
        class MockToolLibrary:
            def __init__(self):
                self.tools = {"search": MockTool()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [("call_1", "search", {"query": "test"})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        assert "call_1" in configs
        assert configs["call_1"]["ephemeral"] is True
        assert configs["call_1"]["ephemeral_ttl"] == 5
        assert configs["call_1"]["result_importance"] == 0.8
        assert configs["call_1"]["summarize_result"] is True

    def test_get_configs_with_partial_tool_config(self):
        """Test extracting configs when tool_config has only some attributes."""

        class MockTool:
            def __init__(self):
                self.tool_config = type("Config", (), {"ephemeral": True})()

        class MockToolLibrary:
            def __init__(self):
                self.tools = {"fetch": MockTool()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [("call_2", "fetch", {"url": "example.com"})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        assert "call_2" in configs
        assert configs["call_2"]["ephemeral"] is True
        # Missing attributes should default to None/False
        assert configs["call_2"]["ephemeral_ttl"] is None
        assert configs["call_2"]["result_importance"] is None
        assert configs["call_2"]["summarize_result"] is False

    def test_get_configs_without_tool_config(self):
        """Test extracting configs from tools without tool_config attribute."""

        class MockTool:
            pass  # No tool_config attribute

        class MockToolLibrary:
            def __init__(self):
                self.tools = {"simple_tool": MockTool()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [("call_3", "simple_tool", {})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        # Should return default values when tool_config is missing
        assert "call_3" in configs
        assert configs["call_3"]["ephemeral"] is False
        assert configs["call_3"]["ephemeral_ttl"] is None
        assert configs["call_3"]["result_importance"] is None
        assert configs["call_3"]["summarize_result"] is False

    def test_get_configs_with_tool_not_found(self):
        """Test extracting configs when tool is not found in library."""

        class MockToolLibrary:
            def get_tool(self, name):
                return None  # Tool not found

        tool_library = MockToolLibrary()
        tool_callings = [("call_4", "nonexistent_tool", {})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        # Should return default values when tool is not found
        assert "call_4" in configs
        assert configs["call_4"]["ephemeral"] is False
        assert configs["call_4"]["ephemeral_ttl"] is None
        assert configs["call_4"]["result_importance"] is None
        assert configs["call_4"]["summarize_result"] is False

    def test_get_configs_with_multiple_tools(self):
        """Test extracting configs for multiple tool calls."""

        class MockTool1:
            def __init__(self):
                self.tool_config = type(
                    "Config", (), {"ephemeral": True, "ephemeral_ttl": 3}
                )()

        class MockTool2:
            def __init__(self):
                self.tool_config = type(
                    "Config",
                    (),
                    {"result_importance": 0.5, "summarize_result": True},
                )()

        class MockToolLibrary:
            def __init__(self):
                self.tools = {"tool1": MockTool1(), "tool2": MockTool2()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [
            ("call_1", "tool1", {"arg1": "val1"}),
            ("call_2", "tool2", {"arg2": "val2"}),
            ("call_3", "tool1", {"arg3": "val3"}),
        ]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        assert len(configs) == 3

        # call_1 and call_3 use tool1
        assert configs["call_1"]["ephemeral"] is True
        assert configs["call_1"]["ephemeral_ttl"] == 3
        assert configs["call_3"]["ephemeral"] is True
        assert configs["call_3"]["ephemeral_ttl"] == 3

        # call_2 uses tool2
        assert configs["call_2"]["ephemeral"] is False
        assert configs["call_2"]["result_importance"] == 0.5
        assert configs["call_2"]["summarize_result"] is True

    def test_get_configs_with_empty_tool_callings(self):
        """Test extracting configs with empty tool callings list."""

        class MockToolLibrary:
            def get_tool(self, name):
                return None

        tool_library = MockToolLibrary()
        tool_callings = []

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        assert configs == {}

    def test_get_configs_preserves_false_boolean_values(self):
        """Test that False boolean values are preserved, not defaulted."""

        class MockTool:
            def __init__(self):
                self.tool_config = type(
                    "Config",
                    (),
                    {
                        "ephemeral": False,
                        "ephemeral_ttl": 0,  # Explicitly 0
                        "result_importance": 0.0,  # Explicitly 0.0
                        "summarize_result": False,
                    },
                )()

        class MockToolLibrary:
            def __init__(self):
                self.tools = {"tool": MockTool()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [("call_1", "tool", {})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        # Even though values are falsy, they should be the actual values
        assert configs["call_1"]["ephemeral"] is False
        assert configs["call_1"]["ephemeral_ttl"] == 0
        assert configs["call_1"]["result_importance"] == 0.0
        assert configs["call_1"]["summarize_result"] is False

    def test_get_configs_with_different_tool_config_types(self):
        """Test configs with different tool_config object types."""

        # Using a dict-like object
        class DictLikeTool:
            def __init__(self):
                from msgflux.dotdict import dotdict

                self.tool_config = dotdict(
                    {
                        "ephemeral": True,
                        "ephemeral_ttl": 10,
                        "result_importance": 0.9,
                        "summarize_result": False,
                    }
                )

        class MockToolLibrary:
            def __init__(self):
                self.tools = {"dictlike": DictLikeTool()}

            def get_tool(self, name):
                return self.tools.get(name)

        tool_library = MockToolLibrary()
        tool_callings = [("call_1", "dictlike", {})]

        configs = get_tool_lifecycle_configs(tool_library, tool_callings)

        assert configs["call_1"]["ephemeral"] is True
        assert configs["call_1"]["ephemeral_ttl"] == 10
        assert configs["call_1"]["result_importance"] == 0.9
        assert configs["call_1"]["summarize_result"] is False
