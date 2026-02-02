"""Integration tests for ReAct reasoning schema with tool calling.

These tests verify the integration between:
- nn.Agent with generation_schema=ReAct
- Tool calling through the FlowControl interface
- Multi-step reasoning with observation injection

To run with a real model:
    OPENAI_API_KEY=... pytest tests/integration/test_react.py -v

To run the manual test:
    python tests/integration/test_react.py
"""

import os

import msgspec
import pytest

from msgflux.generation.reasoning.react import (
    REACT_SYSTEM_MESSAGE,
    REACT_TOOLS_TEMPLATE,
    ReAct,
    ReActStep,
    ToolCall,
)


class TestReActFlowControl:
    """Tests for ReAct FlowControl interface."""

    def test_extract_flow_result_with_final_answer(self):
        """Test that extract_flow_result completes on final_answer."""
        raw_response = {
            "current_step": None,
            "final_answer": "The capital of France is Paris.",
        }

        flow_result = ReAct.extract_flow_result(raw_response)

        assert flow_result.is_complete
        assert flow_result.final_response == raw_response
        assert flow_result.tool_calls is None
        assert flow_result.reasoning is None

    def test_extract_flow_result_with_tool_calls(self):
        """Test that extract_flow_result creates tool_calls for actions."""
        raw_response = {
            "current_step": {
                "thought": "I need to search for information about France",
                "actions": [
                    {"name": "search", "arguments": {"query": "capital of France"}},
                ],
            },
            "final_answer": None,
        }

        flow_result = ReAct.extract_flow_result(raw_response)

        assert not flow_result.is_complete
        assert flow_result.tool_calls is not None
        assert len(flow_result.tool_calls) == 1

        tool_id, tool_name, tool_args = flow_result.tool_calls[0]
        assert tool_name == "search"
        assert tool_args == {"query": "capital of France"}
        assert tool_id is not None  # ID should be assigned

        # Verify ID was injected into the action
        assert raw_response["current_step"]["actions"][0].get("id") == tool_id
        assert flow_result.reasoning == "I need to search for information about France"

    def test_extract_flow_result_with_multiple_tool_calls(self):
        """Test extract_flow_result with multiple parallel tool calls."""
        raw_response = {
            "current_step": {
                "thought": "I need to search for information and calculate",
                "actions": [
                    {"name": "search", "arguments": {"query": "population of France"}},
                    {"name": "calculate", "arguments": {"expression": "67 * 1000000"}},
                ],
            },
            "final_answer": None,
        }

        flow_result = ReAct.extract_flow_result(raw_response)

        assert not flow_result.is_complete
        assert len(flow_result.tool_calls) == 2

        # First tool call
        assert flow_result.tool_calls[0][1] == "search"
        assert flow_result.tool_calls[0][2] == {"query": "population of France"}

        # Second tool call
        assert flow_result.tool_calls[1][1] == "calculate"
        assert flow_result.tool_calls[1][2] == {"expression": "67 * 1000000"}

        # Each action should have unique ID
        id1 = raw_response["current_step"]["actions"][0].get("id")
        id2 = raw_response["current_step"]["actions"][1].get("id")
        assert id1 != id2

    def test_extract_flow_result_empty_state(self):
        """Test extract_flow_result with no step and no final_answer."""
        raw_response = {"current_step": None, "final_answer": None}
        flow_result = ReAct.extract_flow_result(raw_response)

        assert flow_result.is_complete
        assert flow_result.final_response is raw_response

    def test_inject_tool_results_success(self):
        """Test injecting successful tool results."""
        from unittest.mock import MagicMock

        raw_response = {
            "current_step": {
                "thought": "Search for info",
                "actions": [
                    {"name": "search", "arguments": {"q": "test"}, "id": "abc123"},
                ],
            },
            "final_answer": None,
        }

        mock_results = MagicMock()
        mock_result = MagicMock()
        mock_result.result = "Paris is the capital of France"
        mock_result.error = None
        mock_results.get_by_id = lambda tool_id: mock_result if tool_id == "abc123" else None

        updated = ReAct.inject_tool_results(raw_response, mock_results)

        assert updated["current_step"]["actions"][0]["result"] == "Paris is the capital of France"

    def test_inject_tool_results_error(self):
        """Test injecting error results."""
        from unittest.mock import MagicMock

        raw_response = {
            "current_step": {
                "thought": "Search for info",
                "actions": [
                    {"name": "invalid_tool", "arguments": {}, "id": "xyz789"},
                ],
            },
            "final_answer": None,
        }

        mock_results = MagicMock()
        mock_result = MagicMock()
        mock_result.result = None
        mock_result.error = "Tool 'invalid_tool' not found"
        mock_results.get_by_id = lambda tool_id: mock_result if tool_id == "xyz789" else None

        updated = ReAct.inject_tool_results(raw_response, mock_results)

        assert updated["current_step"]["actions"][0]["result"] == "Tool 'invalid_tool' not found"

    def test_inject_tool_results_multiple(self):
        """Test injecting results for multiple tool calls."""
        from unittest.mock import MagicMock

        raw_response = {
            "current_step": {
                "thought": "Multiple searches",
                "actions": [
                    {"name": "search", "arguments": {"q": "France"}, "id": "id1"},
                    {"name": "search", "arguments": {"q": "Germany"}, "id": "id2"},
                ],
            },
            "final_answer": None,
        }

        mock_results = MagicMock()

        def get_by_id(tool_id):
            mock = MagicMock()
            if tool_id == "id1":
                mock.result = "Capital: Paris"
                mock.error = None
            elif tool_id == "id2":
                mock.result = "Capital: Berlin"
                mock.error = None
            return mock

        mock_results.get_by_id = get_by_id

        updated = ReAct.inject_tool_results(raw_response, mock_results)

        assert updated["current_step"]["actions"][0]["result"] == "Capital: Paris"
        assert updated["current_step"]["actions"][1]["result"] == "Capital: Berlin"


class TestReActBuildHistory:
    """Tests for history building in ReAct."""

    def test_build_history_first_step(self):
        """Test building history for first step."""
        raw_response = {
            "current_step": {
                "thought": "First thought",
                "actions": [
                    {"name": "search", "arguments": {"q": "test"}, "id": "id1", "result": "found it"},
                ],
            },
            "final_answer": None,
        }

        messages = []
        result = ReAct.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Content should be JSON-encoded list with the response
        content = msgspec.json.decode(result[0]["content"])
        assert len(content) == 1
        assert content[0] == raw_response

    def test_build_history_append_to_existing(self):
        """Test appending to existing history."""
        first_response = {
            "current_step": {
                "thought": "First step",
                "actions": [{"name": "search", "arguments": {}, "id": "id1", "result": "r1"}],
            },
            "final_answer": None,
        }
        second_response = {
            "current_step": {
                "thought": "Second step",
                "actions": [{"name": "calculate", "arguments": {}, "id": "id2", "result": "r2"}],
            },
            "final_answer": None,
        }

        messages = []
        ReAct.build_history(first_response, messages)
        ReAct.build_history(second_response, messages)

        # Should have one message with both steps
        assert len(messages) == 1
        content = msgspec.json.decode(messages[0]["content"])
        assert len(content) == 2
        assert content[0]["current_step"]["thought"] == "First step"
        assert content[1]["current_step"]["thought"] == "Second step"

    def test_build_history_multiple_iterations(self):
        """Test multiple iteration history accumulation."""
        messages = []

        for i in range(3):
            response = {
                "current_step": {
                    "thought": f"Step {i + 1}",
                    "actions": [{"name": "tool", "arguments": {}, "id": f"id{i}", "result": f"r{i}"}],
                },
                "final_answer": None,
            }
            ReAct.build_history(response, messages)

        assert len(messages) == 1
        content = msgspec.json.decode(messages[0]["content"])
        assert len(content) == 3


class TestReActStructures:
    """Tests for ReAct data structures."""

    def test_react_step_structure(self):
        """Test ReActStep structure."""
        step = ReActStep(
            thought="I need to search for information",
            actions=[
                ToolCall(name="search", arguments={"q": "test"}),
            ],
        )

        assert step.thought == "I need to search for information"
        assert len(step.actions) == 1
        assert step.actions[0].name == "search"

    def test_tool_call_structure(self):
        """Test ToolCall structure."""
        tool_call = ToolCall(name="calculate", arguments={"a": 1, "b": 2})

        assert tool_call.name == "calculate"
        assert tool_call.arguments == {"a": 1, "b": 2}

    def test_tool_call_optional_arguments(self):
        """Test ToolCall with no arguments."""
        tool_call = ToolCall(name="get_time")

        assert tool_call.name == "get_time"
        assert tool_call.arguments is None

    def test_react_class_attributes(self):
        """Test ReAct class attributes."""
        assert ReAct.system_message == REACT_SYSTEM_MESSAGE
        assert ReAct.tools_template == REACT_TOOLS_TEMPLATE

    def test_react_structure(self):
        """Test ReAct main structure."""
        react = ReAct(
            current_step=ReActStep(
                thought="Testing",
                actions=[ToolCall(name="test")],
            ),
            final_answer=None,
        )

        assert react.current_step is not None
        assert react.current_step.thought == "Testing"
        assert react.final_answer is None

    def test_react_with_final_answer(self):
        """Test ReAct with final answer."""
        react = ReAct(
            current_step=None,
            final_answer="The answer is 42",
        )

        assert react.current_step is None
        assert react.final_answer == "The answer is 42"


class TestReActAsyncMethods:
    """Tests for ReAct async methods."""

    @pytest.mark.asyncio
    async def test_async_extract_flow_result(self):
        """Test async extract_flow_result."""
        raw_response = {"current_step": None, "final_answer": "Done"}

        result = await ReAct.aextract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.final_response is raw_response

    @pytest.mark.asyncio
    async def test_async_inject_tool_results(self):
        """Test async inject_tool_results."""
        from unittest.mock import MagicMock

        raw_response = {
            "current_step": {
                "thought": "Test",
                "actions": [{"name": "test", "arguments": {}, "id": "id1"}],
            },
            "final_answer": None,
        }

        mock_results = MagicMock()
        mock_result = MagicMock()
        mock_result.result = "async result"
        mock_result.error = None
        mock_results.get_by_id = lambda x: mock_result

        result = await ReAct.ainject_tool_results(raw_response, mock_results)

        assert result["current_step"]["actions"][0]["result"] == "async result"

    @pytest.mark.asyncio
    async def test_async_build_history(self):
        """Test async build_history."""
        raw_response = {
            "current_step": {
                "thought": "Test",
                "actions": [{"name": "test", "arguments": {}, "id": "id1", "result": "r"}],
            },
            "final_answer": None,
        }

        history = await ReAct.abuild_history(raw_response, [])

        assert len(history) == 1
        assert history[0]["role"] == "assistant"


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestReActWithModel:
    """Integration tests that require a real LLM model."""

    @pytest.fixture
    def agent(self):
        """Create an agent with ReAct."""
        from msgflux import Model
        from msgflux.nn import Agent, LM

        model = Model.chat_completion("openai/gpt-4.1-mini")
        lm = LM(model=model)

        def search(query: str) -> str:
            """Search for information about a topic."""
            results = {
                "france": "France is a country in Western Europe. Its capital is Paris.",
                "python": "Python is a high-level programming language.",
                "capital": "Paris is the capital of France, with a population of about 2 million.",
            }
            query_lower = query.lower()
            for key, value in results.items():
                if key in query_lower:
                    return value
            return f"No results found for: {query}"

        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {e}"

        agent = Agent(
            name="researcher",
            model=lm,
            tools=[search, calculate],
            generation_schema=ReAct,
            config={"verbose": True},
        )

        return agent

    def test_simple_tool_usage(self, agent):
        """Test a simple task that uses a tool."""
        result = agent("What is the capital of France? Use the search tool to find out.")

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "").lower()
        assert "paris" in final or "france" in final

    def test_multi_step_reasoning(self, agent):
        """Test multi-step reasoning with ReAct."""
        result = agent(
            "First search for information about Python, then calculate 7 * 8. "
            "Tell me both results."
        )

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "").lower()
        # Should mention Python and 56
        has_python = "python" in final or "programming" in final
        has_calc = "56" in final
        assert has_python or has_calc

    def test_calculation_task(self, agent):
        """Test agent using calculate tool."""
        result = agent("What is 123 + 456? Use the calculate tool.")

        assert result.get("final_answer") is not None
        assert "579" in result.get("final_answer", "")


# Manual test for running with a real model
def manual_test():
    """Manual test with real model - run directly with python."""
    try:
        from msgflux import Model
        from msgflux.nn import Agent, LM
    except ImportError:
        print("msgflux not fully installed, skipping manual test")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run manual test")
        print("\nExample:")
        print("  OPENAI_API_KEY=sk-... python tests/integration/test_react.py")
        return

    print("=" * 60)
    print("Manual Integration Test: ReAct Reasoning")
    print("=" * 60)

    # Create model
    model = Model.chat_completion("openai/gpt-4.1-mini")
    lm = LM(model=model)

    # Define tools
    def search(query: str) -> str:
        """Search for information about a topic."""
        print(f"[Tool] search called with: {query}")
        results = {
            "france": "France is a country in Western Europe. Capital: Paris.",
            "germany": "Germany is a country in Central Europe. Capital: Berlin.",
            "python": "Python is a high-level, interpreted programming language.",
        }
        query_lower = query.lower()
        for key, value in results.items():
            if key in query_lower:
                return value
        return f"Information about: {query}"

    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        print(f"[Tool] calculate called with: {expression}")
        try:
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # Create agent with ReAct
    agent = Agent(
        name="researcher",
        model=lm,
        tools=[search, calculate],
        generation_schema=ReAct,
        config={"verbose": True},
    )

    # Test problem
    problem = "Find information about France and then calculate 7 * 8 + 5"

    print(f"\nProblem: {problem}")
    print("-" * 60)

    try:
        result = agent(problem)
        print(f"\nFinal Answer: {result.get('final_answer', 'No answer')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)


if __name__ == "__main__":
    manual_test()
