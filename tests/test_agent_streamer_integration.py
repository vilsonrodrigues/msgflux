"""Integration tests for AgentStreamer with mock streaming model."""

from typing import Any, AsyncIterator, Dict, Iterator, List
from unittest.mock import MagicMock

import pytest

from msgflux.models.streaming import StreamChunk
from msgflux.nn.modules.agent_streamer import AgentStreamer


class MockStreamingModel:
    """Mock model that simulates streaming responses with tool calls."""

    def __init__(self, responses: List[List[str]]):
        """Initialize mock model.

        Args:
            responses: List of response sequences. Each sequence is a list of
                       strings (tokens) to stream. Multiple sequences are used
                       for multi-turn conversations (after tool execution).
        """
        self.responses = responses
        self.call_count = 0
        self.model_id = "mock-model"

    def stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        prefilling: str = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream tokens from the current response sequence."""
        if self.call_count >= len(self.responses):
            # No more responses, return empty
            return

        tokens = self.responses[self.call_count]
        self.call_count += 1

        for token in tokens:
            yield StreamChunk(type="text", content=token)

    async def astream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        prefilling: str = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Async version of stream."""
        if self.call_count >= len(self.responses):
            return

        tokens = self.responses[self.call_count]
        self.call_count += 1

        for token in tokens:
            yield StreamChunk(type="text", content=token)


def create_tool_xml(name: str, params: Dict[str, Any]) -> str:
    """Helper to create tool call XML."""
    param_xml = ""
    for pname, pvalue in params.items():
        param_xml += f'<parameter name="{pname}">{pvalue}</parameter>'
    return f'<function_calls><invoke name="{name}">{param_xml}</invoke></function_calls>'  # noqa: E501


class TestAgentStreamerIntegration:
    """Integration tests for AgentStreamer."""

    def test_text_only_response(self):
        """Test streaming a response with no tool calls."""
        # Model returns simple text in tokens
        mock_model = MockStreamingModel([
            ["Hello", ", ", "how", " can", " I", " help", " you", "?"]
        ])

        # Create mock LM wrapper
        mock_lm = MagicMock()
        mock_lm.model = mock_model

        # Create agent with minimal setup
        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None
        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Mock _prepare_task to return minimal state
        agent._prepare_task = MagicMock(return_value={
            "model_state": [{"role": "user", "content": "Hi"}],
            "vars": {},
            "model_preference": None,
        })

        # Mock _prepare_model_execution
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [{"role": "user", "content": "Hi"}],
            "system_prompt": "You are helpful.",
            "prefilling": None,
        })

        # Collect events
        events = list(agent.stream("Hi"))

        # Check we got text events and done
        text_events = [e for e in events if e.type == "text"]
        done_events = [e for e in events if e.type == "done"]

        assert len(text_events) == 8
        assert text_events[0].content == "Hello"
        assert text_events[-1].content == "?"
        assert len(done_events) == 1

        # Verify full text
        full_text = "".join(e.content for e in text_events)
        assert full_text == "Hello, how can I help you?"

    def test_text_with_single_tool_call(self):
        """Test streaming response with text followed by tool call."""
        # Model first explains, then calls tool
        tool_xml = create_tool_xml("calculator", {"expression": "2+2"})
        mock_model = MockStreamingModel([
            # First response: text + tool call
            ["Let me ", "calculate ", "that. ", tool_xml],
            # Second response after tool result: final answer
            ["The ", "answer ", "is ", "4."]
        ])

        # Setup mock tool
        def calculator(expression: str) -> str:
            """Calculate expression."""
            return str(eval(expression))

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None

        # Setup tool library mock
        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    }
                }
            }
        }]

        # Mock tool execution
        mock_tool_result = MagicMock()
        mock_tool_result.tool_calls = [MagicMock(result="4", error=None)]
        mock_tool_result.return_directly = False
        agent.tool_library.return_value = mock_tool_result

        agent._prepare_task = MagicMock(return_value={
            "model_state": [{"role": "user", "content": "What is 2+2?"}],
            "vars": {},
            "model_preference": None,
        })

        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "system_prompt": "You are helpful.",
            "prefilling": None,
        })

        # Collect events
        events = list(agent.stream("What is 2+2?"))

        # Check event types
        text_events = [e for e in events if e.type == "text"]
        tool_start_events = [e for e in events if e.type == "tool_start"]
        tool_result_events = [e for e in events if e.type == "tool_result"]
        done_events = [e for e in events if e.type == "done"]

        # Should have text, tool_start, tool_result, more text, done
        assert len(tool_start_events) == 1
        assert tool_start_events[0].tool_name == "calculator"
        assert tool_start_events[0].tool_params == {"expression": "2+2"}

        assert len(tool_result_events) == 1
        assert tool_result_events[0].tool_result == "4"

        assert len(done_events) == 1

        # Verify text includes both parts
        all_text = "".join(e.content for e in text_events)
        assert "Let me calculate that." in all_text
        assert "The answer is 4." in all_text

    def test_callbacks_are_called(self):
        """Test that callbacks are invoked correctly."""
        tool_xml = create_tool_xml("search", {"query": "python"})
        mock_model = MockStreamingModel([
            ["Searching", "...", tool_xml],
            ["Found ", "results."]
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        # Track callback invocations
        text_calls = []
        tool_start_calls = []
        tool_result_calls = []

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = lambda t: text_calls.append(t)
        agent.on_reasoning = None
        agent.on_tool_start = lambda id, n, p: tool_start_calls.append((n, p))
        agent.on_tool_result = lambda id, r: tool_result_calls.append(r)
        agent.on_error = None

        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = [{
            "type": "function",
            "function": {"name": "search", "parameters": {"properties": {}}}
        }]

        mock_tool_result = MagicMock()
        mock_tool_result.tool_calls = [MagicMock(result="results", error=None)]
        mock_tool_result.return_directly = False
        agent.tool_library.return_value = mock_tool_result

        agent._prepare_task = MagicMock(return_value={
            "model_state": [],
            "vars": {},
            "model_preference": None,
        })
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        # Run streaming
        list(agent.stream("search python"))

        # Verify callbacks
        assert "Searching" in text_calls
        assert "..." in text_calls
        assert ("search", {"query": "python"}) in tool_start_calls
        assert "results" in tool_result_calls

    def test_multiple_tool_calls_in_sequence(self):
        """Test multiple tool calls in a single response."""
        # Model calls two tools
        tool1_xml = create_tool_xml("get_weather", {"city": "NYC"})
        tool2_xml = create_tool_xml("get_time", {"timezone": "EST"})
        mock_model = MockStreamingModel([
            ["Let me check ", tool1_xml, tool2_xml],
            ["NYC is 72F ", "and time is 3pm."]
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None

        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Mock tool responses for each call
        mock_results = [
            MagicMock(tool_calls=[MagicMock(result="72F", error=None)],
                      return_directly=False),
            MagicMock(tool_calls=[MagicMock(result="3pm", error=None)],
                      return_directly=False),
        ]
        agent.tool_library.side_effect = mock_results

        agent._prepare_task = MagicMock(return_value={
            "model_state": [],
            "vars": {},
            "model_preference": None,
        })
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        events = list(agent.stream("weather and time?"))

        tool_starts = [e for e in events if e.type == "tool_start"]
        tool_results = [e for e in events if e.type == "tool_result"]

        assert len(tool_starts) == 2
        assert tool_starts[0].tool_name == "get_weather"
        assert tool_starts[1].tool_name == "get_time"

        assert len(tool_results) == 2

    def test_return_directly_stops_loop(self):
        """Test that return_directly=True stops the agent loop."""
        tool_xml = create_tool_xml("final_answer", {"answer": "42"})
        mock_model = MockStreamingModel([
            ["The answer is ", tool_xml],
            # This should NOT be reached
            ["This should not appear"]
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None

        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Tool returns with return_directly=True
        mock_tool_result = MagicMock()
        mock_tool_result.tool_calls = [MagicMock(result="42", error=None)]
        mock_tool_result.return_directly = True
        agent.tool_library.return_value = mock_tool_result

        agent._prepare_task = MagicMock(return_value={
            "model_state": [],
            "vars": {},
            "model_preference": None,
        })
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        events = list(agent.stream("answer?"))

        # Should have stopped after tool result
        text_events = [e for e in events if e.type == "text"]
        full_text = "".join(e.content for e in text_events)

        assert "The answer is " in full_text
        assert "This should not appear" not in full_text

    def test_max_iterations_limit(self):
        """Test that max_iterations prevents infinite loops."""
        # Model always returns a tool call
        tool_xml = create_tool_xml("loop", {"n": "1"})
        mock_model = MockStreamingModel([
            [tool_xml] for _ in range(20)  # More than max_iterations
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 3  # Limit to 3
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None

        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        mock_tool_result = MagicMock()
        mock_tool_result.tool_calls = [MagicMock(result="ok", error=None)]
        mock_tool_result.return_directly = False
        agent.tool_library.return_value = mock_tool_result

        agent._prepare_task = MagicMock(return_value={
            "model_state": [],
            "vars": {},
            "model_preference": None,
        })
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        events = list(agent.stream("loop"))

        # Should hit max iterations error
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Max iterations" in error_events[0].error

        # Should have exactly 3 tool calls
        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 3

    def test_error_handling(self):
        """Test that errors are caught and emitted as error events."""
        mock_model = MockStreamingModel([["test"]])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        error_callback_calls = []

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = lambda e: error_callback_calls.append(e)
        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Make _prepare_task raise an error
        agent._prepare_task = MagicMock(side_effect=ValueError("Test error"))

        events = list(agent.stream("test"))

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "Test error" in error_events[0].error

        # Callback should have been called
        assert len(error_callback_calls) == 1


@pytest.mark.asyncio
class TestAgentStreamerAsyncIntegration:
    """Async integration tests for AgentStreamer."""

    async def test_async_text_only(self):
        """Test async streaming with no tool calls."""
        mock_model = MockStreamingModel([
            ["Hello", " ", "async", " ", "world", "!"]
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None
        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Mock async prepare task
        async def mock_aprepare_task(*args, **kwargs):
            return {
                "model_state": [],
                "vars": {},
                "model_preference": None,
            }

        agent._aprepare_task = mock_aprepare_task
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        events = []
        async for event in agent.astream("Hi"):
            events.append(event)

        text_events = [e for e in events if e.type == "text"]
        full_text = "".join(e.content for e in text_events)
        assert full_text == "Hello async world!"

    async def test_async_with_tool_call(self):
        """Test async streaming with tool call."""
        tool_xml = create_tool_xml("async_tool", {"param": "value"})
        mock_model = MockStreamingModel([
            ["Calling ", tool_xml],
            ["Done!"]
        ])

        mock_lm = MagicMock()
        mock_lm.model = mock_model

        agent = AgentStreamer.__new__(AgentStreamer)
        agent.lm = mock_lm
        agent.prefilling = None
        agent.max_iterations = 10
        agent.on_text = None
        agent.on_reasoning = None
        agent.on_tool_start = None
        agent.on_tool_result = None
        agent.on_error = None

        agent.tool_library = MagicMock()
        agent.tool_library.get_tool_json_schemas.return_value = []

        # Mock async tool execution
        async def mock_acall(*args, **kwargs):
            result = MagicMock()
            result.tool_calls = [MagicMock(result="async_result", error=None)]
            result.return_directly = False
            return result

        agent.tool_library.acall = mock_acall

        async def mock_aprepare_task(*args, **kwargs):
            return {
                "model_state": [],
                "vars": {},
                "model_preference": None,
            }

        agent._aprepare_task = mock_aprepare_task
        agent._prepare_model_execution = MagicMock(return_value={
            "messages": [],
            "system_prompt": None,
            "prefilling": None,
        })

        events = []
        async for event in agent.astream("test"):
            events.append(event)

        tool_starts = [e for e in events if e.type == "tool_start"]
        assert len(tool_starts) == 1
        assert tool_starts[0].tool_name == "async_tool"


class TestBuildToolsSystemPrompt:
    """Test the _build_tools_system_prompt method."""

    def test_builds_correct_prompt(self):
        """Test that tools prompt is built correctly."""
        agent = AgentStreamer.__new__(AgentStreamer)

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        }
                    }
                }
            }
        ]

        prompt = agent._build_tools_system_prompt(tool_schemas)

        assert "search" in prompt
        assert "Search the web" in prompt
        assert "query" in prompt
        assert "function_calls" in prompt
        assert "invoke" in prompt
        assert "parameter" in prompt
