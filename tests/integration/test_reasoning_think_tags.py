"""Integration tests: reasoning <think> tags with OpenAI and Groq.

Validates:
1. ChatBlock helpers produce correct <think> tag format
2. ToolCallAggregator embeds reasoning in tool_call message (not separate)
3. OpenAI accepts <think> tags in multi-turn conversation
4. Groq gpt-oss reasoning model works end-to-end
5. nn.Agent tool calls work with reasoning models
"""

import os
import unicodedata

import pytest

import msgflux as mf

mf.load_dotenv()

from msgflux.chat_messages import ChatMessages
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.chat import ChatBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

skip_no_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
skip_no_groq = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
)


def normalize(text: str) -> str:
    """Remove unicode spaces, commas, etc. for numeric comparison."""
    return unicodedata.normalize("NFKD", str(text)).replace(",", "").replace(" ", "")


def calculate(expression: str) -> str:
    """Perform a math calculation and return the result.

    Args:
        expression: Math expression to evaluate (e.g. '2 + 2', '15 * 37')
    """
    try:
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ===========================================================================
# Unit: ChatBlock helpers
# ===========================================================================


class TestChatBlockThinkTags:
    def test_assist_reasoning_embeds_think_tags(self):
        msg = ChatBlock.assist_reasoning("The capital is Paris.", "User asked about France")
        assert msg["role"] == "assistant"
        assert msg["content"].startswith("<think>")
        assert "</think>" in msg["content"]
        assert "The capital is Paris." in msg["content"]

    def test_assist_tool_calls_with_reasoning(self):
        tool_calls = [ChatBlock.tool_call("call_1", "get_weather", '{"city": "SP"}')]
        msg = ChatBlock.assist_tool_calls(tool_calls, reasoning="Need weather data")
        assert msg["role"] == "assistant"
        assert msg["content"] == "<think>Need weather data</think>"
        assert len(msg["tool_calls"]) == 1

    def test_assist_tool_calls_without_reasoning(self):
        tool_calls = [ChatBlock.tool_call("call_2", "search", '{"q": "test"}')]
        msg = ChatBlock.assist_tool_calls(tool_calls)
        assert msg["role"] == "assistant"
        assert "content" not in msg
        assert len(msg["tool_calls"]) == 1


# ===========================================================================
# Unit: ToolCallAggregator
# ===========================================================================


class TestToolCallAggregatorThinkTags:
    def test_with_reasoning_embeds_in_single_message(self):
        agg = ToolCallAggregator(reasoning="I should use the tool")
        agg.process(0, "call_abc", "get_weather", '{"city": "Natal"}')
        agg.insert_results({"call_abc": "25°C sunny"})
        messages = agg.get_messages()

        # Must be 2 messages: assistant (tool_calls + <think>) + tool result
        # NOT 3 (no separate reasoning assistant message)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert "<think>" in messages[0]["content"]
        assert "tool_calls" in messages[0]
        assert messages[1]["role"] == "tool"

    def test_without_reasoning_no_content_field(self):
        agg = ToolCallAggregator()
        agg.process(0, "call_def", "search", '{"q": "test"}')
        agg.insert_results({"call_def": "result"})
        messages = agg.get_messages()

        assert len(messages) == 2
        assert "content" not in messages[0]


# ===========================================================================
# Integration: OpenAI
# ===========================================================================


@skip_no_openai
class TestOpenAIThinkTags:
    """Test that OpenAI accepts <think> tags in conversation history."""

    @pytest.fixture(scope="class")
    def model(self):
        return mf.Model.chat_completion(
            "openai/gpt-5.4-nano",
            max_tokens=256,
        )

    def test_multi_turn_with_think_tags(self, model):
        """OpenAI should accept assistant messages containing <think> tags."""
        messages = [
            ChatBlock.user("What is 10 + 20?"),
            ChatBlock.assist_reasoning("30", "10 + 20 = 30"),
            ChatBlock.user("Multiply that by 3."),
        ]
        response = model(messages=messages)
        assert response.response_type == "text_generation"
        assert "90" in normalize(str(response.data))

    def test_tool_call_with_think_tags_in_history(self, model):
        """OpenAI should accept tool call messages with <think> in content."""
        # Build a synthetic tool-call history with <think> reasoning
        tool_calls = [ChatBlock.tool_call("call_001", "calculate", '{"expression": "5*6"}')]
        messages = [
            ChatBlock.user("Calculate 5 * 6 using the calculate tool."),
            ChatBlock.assist_tool_calls(tool_calls, reasoning="Need to multiply"),
            ChatBlock.tool("call_001", "30"),
            ChatBlock.user("What was the result?"),
        ]
        response = model(messages=messages)
        assert response.response_type == "text_generation"
        assert "30" in str(response.data)

    def test_chat_messages_container(self, model):
        """ChatMessages container should work with the model."""
        cm = ChatMessages()
        cm.add_user("What is the capital of Brazil?")
        response = model(messages=cm)
        assert response.response_type == "text_generation"
        assert "Brasília" in str(response.data) or "Brasilia" in str(response.data)


# ===========================================================================
# Integration: Groq gpt-oss (reasoning model)
# ===========================================================================


@skip_no_groq
class TestGroqReasoningThinkTags:
    """Test reasoning with Groq gpt-oss model and <think> tag handling."""

    @pytest.fixture(scope="class")
    def model(self):
        return mf.Model.chat_completion(
            "groq/openai/gpt-oss-20b",
            reasoning_effort="low",
            return_reasoning=True,
            max_tokens=1024,
        )

    def test_simple_completion_with_reasoning(self, model):
        """Model should return reasoning content."""
        response = model(
            messages=[ChatBlock.user("What is 15 * 37?")],
            system_prompt="Answer with just the number.",
        )
        assert response.response_type in ("text_generation", "reasoning_text_generation")
        assert "555" in normalize(str(response.data))

    def test_multi_turn_with_think_tags(self, model):
        """Groq should accept <think> tags in assistant messages."""
        messages = [
            ChatBlock.user("What is 15 * 37?"),
            ChatBlock.assist_reasoning("555", "15*37 = 15*30 + 15*7 = 450+105 = 555"),
            ChatBlock.user("Multiply that result by 2."),
        ]
        response = model(
            messages=messages,
            system_prompt="Answer with just the number.",
        )
        assert response.response_type in ("text_generation", "reasoning_text_generation")
        assert "1110" in normalize(str(response.data))

    def test_tool_call_with_reasoning(self, model):
        """Tool call should embed reasoning in <think> tags, not as separate message."""
        tool_schema = {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a math calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
        response = model(
            messages=[ChatBlock.user("Use the calculate tool to compute 123 * 456.")],
            system_prompt="You must use the calculate tool.",
            tool_schemas=[tool_schema],
        )
        assert response.response_type == "tool_call"
        agg = response.data
        calls = agg.get_calls()
        assert len(calls) >= 1

        # Simulate tool execution and check message format
        agg.insert_results({calls[0][0]: "56088"})
        messages = agg.get_messages()

        # Must be exactly 2 messages (assistant + tool), not 3
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert "tool_calls" in messages[0]

        if agg.reasoning:
            assert "<think>" in messages[0].get("content", "")


# ===========================================================================
# Integration: nn.Agent end-to-end
# ===========================================================================


@skip_no_groq
class TestAgentToolCallsWithReasoning:
    """End-to-end Agent test with tool use and reasoning model."""

    @pytest.fixture(scope="class")
    def agent(self):
        import msgflux.nn as nn

        agent_model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-20b",
            reasoning_effort="low",
            return_reasoning=True,
            max_tokens=1024,
        )

        class MathAgent(nn.Agent):
            model = agent_model
            tools = [calculate]
            system_message = "You are a math assistant. Always use the calculate tool."
            config = {"verbose": True}

        return MathAgent()

    def test_agent_tool_call(self, agent):
        """Agent should complete a tool call loop with reasoning."""
        response = agent("What is 789 * 321?")
        assert "253269" in normalize(str(response))

    def test_agent_follow_up(self, agent):
        """Agent should handle sequential requests."""
        response = agent("What is 100 + 200?")
        assert "300" in normalize(str(response))


@skip_no_openai
class TestAgentOpenAI:
    """End-to-end Agent test with OpenAI (non-reasoning model)."""

    @pytest.fixture(scope="class")
    def agent(self):
        import msgflux.nn as nn

        agent_model = mf.Model.chat_completion(
            "openai/gpt-5.4-nano",
            max_tokens=256,
        )

        class MathAgent(nn.Agent):
            model = agent_model
            tools = [calculate]
            system_message = "You are a math assistant. Always use the calculate tool."

        return MathAgent()

    def test_agent_tool_call(self, agent):
        """Agent should complete a tool call loop."""
        response = agent("What is 42 * 58?")
        assert "2436" in normalize(str(response))
