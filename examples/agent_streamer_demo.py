"""Demo: AgentStreamer with Mock Streaming Model.

This example demonstrates how AgentStreamer processes streaming tokens
and executes tool calls in real-time.

Run with: python examples/agent_streamer_demo.py
"""

import time
from typing import Any, Dict, Iterator, List

from msgflux.models.streaming import StreamChunk
from msgflux.nn.modules.agent_streamer import AgentStreamer

# =============================================================================
# Mock Streaming Model
# =============================================================================


class MockStreamingModel:
    """Mock model that simulates streaming responses with delays.

    This simulates how a real LLM streams tokens one by one.
    """

    def __init__(self, responses: List[List[str]], delay: float = 0.05):
        """Initialize mock model.

        Args:
            responses: List of response sequences. Each sequence is a list of
                       strings (tokens) to stream.
            delay: Delay between tokens in seconds (simulates generation time).
        """
        self.responses = responses
        self.delay = delay
        self.call_count = 0
        self.model_id = "mock-streaming-model"

    def stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = None,
        prefilling: str = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        """Stream tokens with simulated delay."""
        if self.call_count >= len(self.responses):
            return

        tokens = self.responses[self.call_count]
        self.call_count += 1

        for token in tokens:
            time.sleep(self.delay)  # Simulate generation time
            yield StreamChunk(type="text", content=token)


# =============================================================================
# Tool Definitions
# =============================================================================


def calculator(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2")

    Returns:
        The result as a string
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city

    Returns:
        Weather information
    """
    # Mock weather data
    weather_data = {
        "new york": "72°F, Sunny",
        "london": "18°C, Cloudy",
        "tokyo": "25°C, Clear",
        "paris": "20°C, Rainy",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


def search(query: str) -> str:
    """Search for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    # Mock search results
    return f"Found 3 results for '{query}': [Result 1], [Result 2], [Result 3]"


# =============================================================================
# Helper to create tool XML
# =============================================================================


def create_tool_xml(name: str, params: Dict[str, Any]) -> str:
    """Create XML for a tool call."""
    param_xml = ""
    for pname, pvalue in params.items():
        param_xml += f'<parameter name="{pname}">{pvalue}</parameter>'
    return (
        f'<function_calls><invoke name="{name}">'
        f'{param_xml}</invoke></function_calls>'
    )


# =============================================================================
# Demo Scenarios
# =============================================================================


def demo_simple_text():
    """Demo 1: Simple text response without tools."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Text Response (No Tools)")
    print("=" * 60 + "\n")

    # Model just returns text
    mock_model = MockStreamingModel([
        ["Hello", "! ", "I'm ", "your ", "AI ", "assistant", ". ",
         "How ", "can ", "I ", "help ", "you ", "today", "?"]
    ])

    # Create a minimal agent setup
    from unittest.mock import MagicMock

    mock_lm = MagicMock()
    mock_lm.model = mock_model

    agent = AgentStreamer.__new__(AgentStreamer)
    agent.lm = mock_lm
    agent.prefilling = None
    agent.max_iterations = 10
    agent.on_text = lambda t: print(t, end="", flush=True)
    agent.on_reasoning = None
    agent.on_tool_start = None
    agent.on_tool_result = None
    agent.on_error = lambda e: print(f"\n[ERROR] {e}")
    agent.tool_library = MagicMock()
    agent.tool_library.get_tool_json_schemas.return_value = []

    agent._prepare_task = MagicMock(return_value={
        "model_state": [{"role": "user", "content": "Hi!"}],
        "vars": {},
        "model_preference": None,
    })
    agent._prepare_model_execution = MagicMock(return_value={
        "messages": [{"role": "user", "content": "Hi!"}],
        "system_prompt": "You are helpful.",
        "prefilling": None,
    })

    print("User: Hi!")
    print("\nAssistant: ", end="")

    for event in agent.stream("Hi!"):
        if event.type == "done":
            print("\n")


def demo_calculator():
    """Demo 2: Using calculator tool."""
    print("\n" + "=" * 60)
    print("Demo 2: Calculator Tool")
    print("=" * 60 + "\n")

    # Model explains, calls calculator, then gives final answer
    calc_xml = create_tool_xml("calculator", {"expression": "15 * 7 + 23"})

    mock_model = MockStreamingModel([
        ["Let ", "me ", "calculate ", "that ", "for ", "you", ".\n\n", calc_xml],
        ["\nThe ", "result ", "of ", "15 × 7 + 23 ", "is ", "**128**", "."]
    ], delay=0.03)

    from unittest.mock import MagicMock

    mock_lm = MagicMock()
    mock_lm.model = mock_model

    agent = AgentStreamer.__new__(AgentStreamer)
    agent.lm = mock_lm
    agent.prefilling = None
    agent.max_iterations = 10
    agent.on_text = lambda t: print(t, end="", flush=True)
    agent.on_reasoning = None
    agent.on_tool_start = lambda id, name, params: print(
        f"\n\n🔧 [Calling {name}({params})]", flush=True
    )
    agent.on_tool_result = lambda id, result: print(
        f"📤 [Result: {result}]", flush=True
    )
    agent.on_error = lambda e: print(f"\n[ERROR] {e}")

    agent.tool_library = MagicMock()
    agent.tool_library.get_tool_json_schemas.return_value = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate math expressions",
            "parameters": {"properties": {"expression": {"type": "string"}}}
        }
    }]

    # Mock tool execution - actually run calculator
    def execute_tool(tool_callings, model_state, vars):
        tool_id, tool_name, tool_params = tool_callings[0]
        if tool_name == "calculator":
            result = calculator(tool_params.get("expression", ""))
        else:
            result = "Unknown tool"

        mock_result = MagicMock()
        mock_result.tool_calls = [MagicMock(result=result, error=None)]
        mock_result.return_directly = False
        return mock_result

    agent.tool_library.side_effect = execute_tool

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

    print("User: What is 15 * 7 + 23?")
    print("\nAssistant: ", end="")

    for event in agent.stream("What is 15 * 7 + 23?"):
        if event.type == "done":
            print("\n")


def demo_multiple_tools():
    """Demo 3: Multiple tool calls."""
    print("\n" + "=" * 60)
    print("Demo 3: Multiple Tools (Weather + Search)")
    print("=" * 60 + "\n")

    weather_xml = create_tool_xml("get_weather", {"city": "Tokyo"})
    search_xml = create_tool_xml("search", {"query": "Tokyo tourist attractions"})

    mock_model = MockStreamingModel([
        ["I'll ", "help ", "you ", "plan ", "your ", "trip ", "to ", "Tokyo", "!\n\n",
         "First, ", "let ", "me ", "check ", "the ", "weather", ":\n", weather_xml,
         "\n\nNow ", "let ", "me ", "find ", "attractions", ":\n", search_xml],
        ["\n\n**Tokyo Trip Summary:**\n",
         "- Weather: ", "25°C, Clear ", "- perfect ", "for ", "sightseeing", "!\n",
         "- Top ", "attractions ", "to ", "visit ", "are ", "available", ".\n\n",
         "Have ", "a ", "great ", "trip", "! 🗼"]
    ], delay=0.02)

    from unittest.mock import MagicMock

    mock_lm = MagicMock()
    mock_lm.model = mock_model

    agent = AgentStreamer.__new__(AgentStreamer)
    agent.lm = mock_lm
    agent.prefilling = None
    agent.max_iterations = 10
    agent.on_text = lambda t: print(t, end="", flush=True)
    agent.on_reasoning = None
    agent.on_tool_start = lambda id, name, params: print(
        f"\n🔧 [{name}({params})]", flush=True
    )
    agent.on_tool_result = lambda id, result: print(
        f"📤 [{result}]\n", flush=True
    )
    agent.on_error = lambda e: print(f"\n[ERROR] {e}")

    agent.tool_library = MagicMock()
    agent.tool_library.get_tool_json_schemas.return_value = []

    # Track which tool to execute
    tool_calls_made = []

    def execute_tool(tool_callings, model_state, vars):
        tool_id, tool_name, tool_params = tool_callings[0]
        tool_calls_made.append(tool_name)

        if tool_name == "get_weather":
            result = get_weather(tool_params.get("city", ""))
        elif tool_name == "search":
            result = search(tool_params.get("query", ""))
        else:
            result = "Unknown tool"

        mock_result = MagicMock()
        mock_result.tool_calls = [MagicMock(result=result, error=None)]
        mock_result.return_directly = False
        return mock_result

    agent.tool_library.side_effect = execute_tool

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

    print("User: I'm planning a trip to Tokyo. What's the weather and attractions?")
    print("\nAssistant: ", end="")

    for event in agent.stream("trip to Tokyo"):
        if event.type == "done":
            print("\n")


def demo_return_directly():
    """Demo 4: Tool with return_directly (final answer tool)."""
    print("\n" + "=" * 60)
    print("Demo 4: Return Directly (Final Answer Tool)")
    print("=" * 60 + "\n")

    final_xml = create_tool_xml(
        "final_answer", {"answer": "The meaning of life is 42."}
    )

    mock_model = MockStreamingModel([
        [
            "After ", "careful ", "analysis", ", ", "here ", "is ", "my ",
            "answer", ":\n\n", final_xml
        ],
        # This should NOT be reached
        ["This ", "text ", "should ", "never ", "appear", "!"]
    ], delay=0.03)

    from unittest.mock import MagicMock

    mock_lm = MagicMock()
    mock_lm.model = mock_model

    agent = AgentStreamer.__new__(AgentStreamer)
    agent.lm = mock_lm
    agent.prefilling = None
    agent.max_iterations = 10
    agent.on_text = lambda t: print(t, end="", flush=True)
    agent.on_reasoning = None
    agent.on_tool_start = lambda id, name, params: print(
        "\n\n🎯 [Final Answer Tool Called]", flush=True
    )
    agent.on_tool_result = lambda id, result: print(
        f"✅ Answer: {result}\n", flush=True
    )
    agent.on_error = lambda e: print(f"\n[ERROR] {e}")

    agent.tool_library = MagicMock()
    agent.tool_library.get_tool_json_schemas.return_value = []

    def execute_tool(tool_callings, model_state, vars):
        tool_id, tool_name, tool_params = tool_callings[0]
        result = tool_params.get("answer", "No answer")

        mock_result = MagicMock()
        mock_result.tool_calls = [MagicMock(result=result, error=None)]
        mock_result.return_directly = True  # This stops the loop!
        return mock_result

    agent.tool_library.side_effect = execute_tool

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

    print("User: What is the meaning of life?")
    print("\nAssistant: ", end="")

    for event in agent.stream("meaning of life"):
        if event.type == "done":
            print("\n[Stream ended - return_directly stopped the loop]")
            print()


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    print("\n🚀 AgentStreamer Demo with Mock Streaming Model")
    print("=" * 60)
    print("This demo shows how AgentStreamer processes streaming")
    print("tokens and executes tool calls in real-time.")
    print("=" * 60)

    demo_simple_text()
    demo_calculator()
    demo_multiple_tools()
    demo_return_directly()

    print("\n✨ All demos completed!")
    print("=" * 60 + "\n")
