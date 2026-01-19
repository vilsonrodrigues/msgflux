"""Tests for reasoning strategies (CoT, ReAct, Self-Consistency)."""

from unittest.mock import MagicMock

import pytest
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult
from msgflux.generation.reasoning.cot import ChainOfThought
from msgflux.generation.reasoning.react import (
    REACT_SYSTEM_MESSAGE,
    REACT_TOOLS_TEMPLATE,
    ReAct,
    ReActStep,
    ToolCall,
)
from msgflux.generation.reasoning.self_consistency import ReasoningPath, SelfConsistency


class TestChainOfThought:
    """Tests for Chain of Thought reasoning strategy."""

    def test_chain_of_thought_is_struct(self):
        """Test that ChainOfThought is a Struct."""
        assert issubclass(ChainOfThought, Struct)

    def test_chain_of_thought_has_required_fields(self):
        """Test that ChainOfThought has required fields."""
        cot = ChainOfThought(
            reasoning="First, we need to analyze the problem. Then we solve it.",
            final_answer="42",
        )
        assert (
            cot.reasoning == "First, we need to analyze the problem. Then we solve it."
        )
        assert cot.final_answer == "42"


class TestReActToolFlowControl:
    """Tests for ReAct ToolFlowControl interface implementation."""

    def test_react_inherits_tool_flow_control(self):
        """Test that ReAct inherits from ToolFlowControl."""
        assert issubclass(ReAct, ToolFlowControl)

    def test_react_has_class_attributes(self):
        """Test that ReAct has system_message and tools_template."""
        assert ReAct.system_message == REACT_SYSTEM_MESSAGE
        assert ReAct.tools_template == REACT_TOOLS_TEMPLATE

    def test_tool_call_has_id_and_result_fields(self):
        """Test that ToolCall has id and result fields."""
        tool_call = ToolCall(name="search", arguments={"q": "test"})
        assert tool_call.name == "search"
        assert tool_call.arguments == {"q": "test"}
        assert tool_call.id is None
        assert tool_call.result is None

        # With id and result
        tool_call_with_result = ToolCall(
            name="search", arguments={"q": "test"}, id="123", result="found"
        )
        assert tool_call_with_result.id == "123"
        assert tool_call_with_result.result == "found"

    def test_extract_flow_result_with_final_answer(self):
        """Test extract_flow_result when final_answer is present."""
        raw_response = {"current_step": None, "final_answer": "The answer is 42"}
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.final_response is raw_response

    def test_extract_flow_result_with_current_step(self):
        """Test extract_flow_result when current_step has actions."""
        raw_response = {
            "current_step": {
                "thought": "I need to search for information",
                "actions": [
                    {"name": "search", "arguments": {"query": "Python docs"}},
                    {"name": "calculate", "arguments": {"a": 1, "b": 2}},
                ],
            },
            "final_answer": None,
        }
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is False
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0][1] == "search"
        assert result.tool_calls[0][2] == {"query": "Python docs"}
        assert result.tool_calls[1][1] == "calculate"
        assert result.reasoning == "I need to search for information"
        assert result.final_response is None

        # Verify IDs were assigned to actions
        actions = raw_response["current_step"]["actions"]
        assert actions[0].get("id") is not None
        assert actions[1].get("id") is not None

    def test_extract_flow_result_empty_state(self):
        """Test extract_flow_result with no step and no final_answer."""
        raw_response = {"current_step": None, "final_answer": None}
        result = ReAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.tool_calls is None
        assert result.final_response is raw_response

    def test_inject_results(self):
        """Test inject_results adds results to actions."""
        raw_response = {
            "current_step": {
                "thought": "Testing",
                "actions": [
                    {"name": "search", "arguments": {"q": "test"}, "id": "id1"},
                    {"name": "calc", "arguments": {"a": 1}, "id": "id2"},
                ],
            },
            "final_answer": None,
        }

        # Mock tool results
        mock_results = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.result = "search result"
        mock_result1.error = None
        mock_result2 = MagicMock()
        mock_result2.result = None
        mock_result2.error = "calculation error"

        def get_by_id(id):
            if id == "id1":
                return mock_result1
            elif id == "id2":
                return mock_result2
            return None

        mock_results.get_by_id = get_by_id

        raw_response = ReAct.inject_results(raw_response, mock_results)

        actions = raw_response["current_step"]["actions"]
        assert actions[0]["result"] == "search result"
        assert actions[1]["result"] == "calculation error"

    def test_build_history_new_message(self):
        """Test build_history adds new assistant message."""
        raw_response = {
            "current_step": {
                "thought": "Testing",
                "actions": [{"name": "search", "arguments": {}}],
            },
            "final_answer": None,
        }

        model_state = []
        result = ReAct.build_history(raw_response, model_state)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_build_history_append_to_existing(self):
        """Test build_history appends to existing assistant message."""
        import msgspec

        raw_response = {
            "current_step": {
                "thought": "Second step",
                "actions": [{"name": "calc", "arguments": {"a": 1}}],
            },
            "final_answer": None,
        }

        # Simulate existing ReAct state in model_state
        existing_react = {
            "current_step": {
                "thought": "First step",
                "actions": [{"name": "search", "arguments": {}}],
            },
            "final_answer": None,
        }
        existing_content = msgspec.json.encode([existing_react]).decode()
        model_state = [{"role": "assistant", "content": existing_content}]

        result = ReAct.build_history(raw_response, model_state)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_async_methods_work(self):
        """Test that async methods work (default to sync)."""
        raw_response = {"current_step": None, "final_answer": "Done"}

        result = await ReAct.aextract_flow_result(raw_response)
        assert result.is_complete is True

        raw_response2 = await ReAct.ainject_results(raw_response, MagicMock())
        assert raw_response2 is raw_response

        history = await ReAct.abuild_history(raw_response, [])
        assert len(history) == 1
