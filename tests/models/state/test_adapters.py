"""Tests for message adapters."""

from unittest.mock import Mock

import msgspec

from msgflux.models.state.adapters import get_adapter
from msgflux.models.state.adapters.openai import OpenAIChatAdapter
from msgflux.models.state.adapters.registry import list_adapters, register_adapter
from msgflux.models.state.adapters.vllm import VLLMAdapter
from msgflux.models.state.types import (
    AudioContent,
    ChatMessage,
    FileContent,
    ImageContent,
    Reasoning,
    Role,
    TextContent,
    ToolCall,
    ToolResult,
    VideoContent,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)


class TestOpenAIChatAdapter:
    """Tests for OpenAIChatAdapter."""

    def test_convert_simple_text_message(self):
        adapter = OpenAIChatAdapter()
        messages = [user_message("Hello world")]

        result = adapter.to_provider_format(messages)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello world"

    def test_convert_with_system_prompt(self):
        adapter = OpenAIChatAdapter()
        messages = [user_message("Hello")]

        result = adapter.to_provider_format(messages, system_prompt="You are helpful")

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"
        assert result["messages"][1]["role"] == "user"

    def test_convert_system_message(self):
        adapter = OpenAIChatAdapter()
        messages = [system_message("You are helpful"), user_message("Hello")]

        result = adapter.to_provider_format(messages)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_convert_assistant_message_with_content(self):
        adapter = OpenAIChatAdapter()
        messages = [assistant_message("Hi there!")]

        result = adapter.to_provider_format(messages)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["content"] == "Hi there!"

    def test_convert_assistant_message_with_reasoning(self):
        adapter = OpenAIChatAdapter()
        messages = [
            assistant_message(
                "The answer is 42",
                reasoning=Reasoning(content="Let me think about this..."),
            )
        ]

        result = adapter.to_provider_format(messages)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["reasoning_content"] == "Let me think about this..."
        assert msg["content"] == "The answer is 42"

    def test_convert_assistant_message_with_tool_calls(self):
        adapter = OpenAIChatAdapter()
        messages = [
            assistant_message(
                tool_calls=[
                    ToolCall(id="call_1", name="calculator", arguments={"expr": "2+2"})
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1

        tool_call = msg["tool_calls"][0]
        assert tool_call["id"] == "call_1"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "calculator"
        assert tool_call["function"]["arguments"] == '{"expr":"2+2"}'

    def test_convert_assistant_message_with_only_tool_calls_no_content(self):
        adapter = OpenAIChatAdapter()
        messages = [
            assistant_message(
                tool_calls=[
                    ToolCall(id="call_1", name="calculator", arguments={"x": 1})
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert "content" not in msg or msg["content"] == ""
        assert "tool_calls" in msg

    def test_convert_assistant_message_with_no_content_and_no_tool_calls(self):
        adapter = OpenAIChatAdapter()
        messages = [
            ChatMessage(role=Role.ASSISTANT, content=None)
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == ""

    def test_convert_tool_result_message(self):
        adapter = OpenAIChatAdapter()
        messages = [
            tool_message(call_id="call_1", content="Result: 42")
        ]

        result = adapter.to_provider_format(messages)

        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == "Result: 42"

    def test_convert_tool_message_without_result_returns_none(self):
        adapter = OpenAIChatAdapter()
        # Create a message with TOOL role but no tool_result
        msg = ChatMessage(role=Role.TOOL, content=[TextContent(text="Invalid")])

        result = adapter._convert_message(msg)

        assert result is None

    def test_convert_unknown_role_returns_none(self):
        adapter = OpenAIChatAdapter()
        # Test with a hypothetical unknown role by patching
        from msgflux.models.state.types import Role as RoleEnum

        # Create a message struct that would have an unhandled role
        # Since we can't add new enum values, we test the None return path
        # by verifying TOOL without tool_result returns None
        msg = ChatMessage(role=Role.TOOL, content=[TextContent(text="Test")])

        result = adapter._convert_message(msg)
        assert result is None

    def test_convert_image_content_with_url(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    TextContent(text="What's in this image?"),
                    ImageContent(url="https://example.com/image.png"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2

        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"
        assert msg["content"][1]["image_url"]["url"] == "https://example.com/image.png"

    def test_convert_image_content_with_base64(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    ImageContent(base64="abc123", media_type="image/jpeg"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_convert_image_content_with_detail(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    ImageContent(url="https://example.com/img.png", detail="high"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["image_url"]["detail"] == "high"

    def test_convert_audio_content_with_base64(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(base64="audio_data_base64", format="mp3"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["data"] == "audio_data_base64"
        assert content[0]["input_audio"]["format"] == "mp3"

    def test_convert_audio_content_with_url(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(url="https://example.com/audio.mp3"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["type"] == "audio_url"
        assert content[0]["audio_url"]["url"] == "https://example.com/audio.mp3"

    def test_convert_audio_content_default_format(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(base64="audio_data"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["input_audio"]["format"] == "wav"

    def test_convert_audio_content_empty(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0] == {}

    def test_convert_video_content_with_url(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    VideoContent(url="https://example.com/video.mp4"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"] == "https://example.com/video.mp4"

    def test_convert_video_content_with_base64(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    VideoContent(base64="video_data", media_type="video/webm"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"] == "data:video/webm;base64,video_data"

    def test_convert_video_content_default_media_type(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    VideoContent(base64="video_data"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["video_url"]["url"] == "data:video/mp4;base64,video_data"

    def test_convert_file_content(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    FileContent(filename="document.pdf", data="file_data_base64"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "file"
        assert content[0]["file"]["filename"] == "document.pdf"
        assert content[0]["file"]["file_data"] == "file_data_base64"

    def test_convert_mixed_content_types(self):
        adapter = OpenAIChatAdapter()
        messages = [
            user_message(
                content=[
                    TextContent(text="Check these files:"),
                    ImageContent(url="https://example.com/img.png"),
                    AudioContent(url="https://example.com/audio.mp3"),
                    VideoContent(url="https://example.com/video.mp4"),
                    FileContent(filename="doc.pdf", data="pdf_data"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 5
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "audio_url"
        assert content[3]["type"] == "video_url"
        assert content[4]["type"] == "file"

    def test_convert_empty_content(self):
        adapter = OpenAIChatAdapter()
        messages = [user_message(content=[])]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        assert msg["content"] == ""

    def test_convert_none_content(self):
        adapter = OpenAIChatAdapter()
        messages = [ChatMessage(role=Role.USER, content=None)]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        assert msg["content"] == ""

    def test_from_provider_response_with_choices(self):
        adapter = OpenAIChatAdapter()

        # Mock OpenAI response with choices
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Hello from assistant"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        result = adapter.from_provider_response(mock_response)

        assert result.role == Role.ASSISTANT
        assert len(result.content) == 1
        assert result.content[0].text == "Hello from assistant"

    def test_from_provider_response_direct_message(self):
        adapter = OpenAIChatAdapter()

        # Mock direct message object (without choices attribute)
        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = "Direct response"
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message, model="gpt-4")

        assert result.role == Role.ASSISTANT
        assert result.content[0].text == "Direct response"
        assert result.metadata.model == "gpt-4"

    def test_from_provider_response_with_reasoning_content(self):
        adapter = OpenAIChatAdapter()

        mock_message = Mock(spec=["content", "reasoning_content", "tool_calls"])
        mock_message.content = "Answer"
        mock_message.reasoning_content = "Let me think..."
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message)

        assert result.reasoning is not None
        assert result.reasoning.content == "Let me think..."

    def test_from_provider_response_with_reasoning_attr(self):
        adapter = OpenAIChatAdapter()

        mock_message = Mock(spec=["content", "reasoning", "tool_calls"])
        mock_message.content = "Answer"
        mock_message.reasoning = "Reasoning text"
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message)

        assert result.reasoning is not None
        assert result.reasoning.content == "Reasoning text"

    def test_from_provider_response_with_thinking_attr(self):
        adapter = OpenAIChatAdapter()

        mock_message = Mock(spec=["content", "thinking", "tool_calls"])
        mock_message.content = "Answer"
        mock_message.thinking = "Thinking text"
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message)

        assert result.reasoning is not None
        assert result.reasoning.content == "Thinking text"

    def test_from_provider_response_with_tool_calls(self):
        adapter = OpenAIChatAdapter()

        # Mock tool calls
        mock_tc = Mock()
        mock_tc.id = "call_123"
        mock_tc.function = Mock()
        mock_tc.function.name = "calculator"
        mock_tc.function.arguments = '{"x": 10, "y": 20}'

        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        result = adapter.from_provider_response(mock_message)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"x": 10, "y": 20}

    def test_from_provider_response_with_tool_calls_dict_arguments(self):
        adapter = OpenAIChatAdapter()

        # Mock tool calls with dict arguments (not JSON string)
        mock_tc = Mock()
        mock_tc.id = "call_456"
        mock_tc.function = Mock()
        mock_tc.function.name = "search"
        mock_tc.function.arguments = {"query": "test"}

        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        result = adapter.from_provider_response(mock_message)

        assert result.tool_calls[0].arguments == {"query": "test"}

    def test_from_provider_response_with_invalid_tool_call_json(self):
        adapter = OpenAIChatAdapter()

        # Mock tool call with invalid JSON
        mock_tc = Mock()
        mock_tc.id = "call_789"
        mock_tc.function = Mock()
        mock_tc.function.name = "broken"
        mock_tc.function.arguments = "not valid json{"

        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = None
        mock_message.tool_calls = [mock_tc]

        result = adapter.from_provider_response(mock_message)

        # Should default to empty dict on parse error
        assert result.tool_calls[0].arguments == {}

    def test_from_provider_response_with_usage(self):
        adapter = OpenAIChatAdapter()

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Hello"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_response.usage = mock_usage

        result = adapter.from_provider_response(mock_response)

        assert result.metadata.usage is not None
        assert result.metadata.usage["input"] == 100
        assert result.metadata.usage["output"] == 50

    def test_from_provider_response_without_usage(self):
        adapter = OpenAIChatAdapter()

        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message)

        assert result.metadata.usage is None

    def test_from_provider_response_empty_content(self):
        adapter = OpenAIChatAdapter()

        mock_message = Mock(spec=["content", "tool_calls"])
        mock_message.content = None
        mock_message.tool_calls = None

        result = adapter.from_provider_response(mock_message)

        assert result.content is None


class TestVLLMAdapter:
    """Tests for VLLMAdapter."""

    def test_inherits_from_openai_adapter(self):
        adapter = VLLMAdapter()
        assert isinstance(adapter, OpenAIChatAdapter)

    def test_convert_audio_with_url_prioritized(self):
        adapter = VLLMAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(url="https://example.com/audio.mp3"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["type"] == "audio_url"
        assert content[0]["audio_url"]["url"] == "https://example.com/audio.mp3"

    def test_convert_audio_with_base64_fallback(self):
        adapter = VLLMAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(base64="audio_data", format="wav"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0]["type"] == "input_audio"
        assert content[0]["input_audio"]["data"] == "audio_data"
        assert content[0]["input_audio"]["format"] == "wav"

    def test_convert_audio_with_both_url_and_base64(self):
        adapter = VLLMAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(
                        url="https://example.com/audio.mp3",
                        base64="audio_data",
                        format="mp3",
                    ),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        # URL should be prioritized in vLLM
        assert content[0]["type"] == "audio_url"
        assert content[0]["audio_url"]["url"] == "https://example.com/audio.mp3"

    def test_convert_audio_empty_returns_empty_dict(self):
        adapter = VLLMAdapter()
        messages = [
            user_message(
                content=[
                    AudioContent(),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert content[0] == {}

    def test_inherits_other_content_conversions(self):
        adapter = VLLMAdapter()
        messages = [
            user_message(
                content=[
                    TextContent(text="Hello"),
                    ImageContent(url="https://example.com/img.png"),
                    VideoContent(url="https://example.com/video.mp4"),
                ]
            )
        ]

        result = adapter.to_provider_format(messages)

        msg = result["messages"][0]
        content = msg["content"]
        assert len(content) == 3
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "video_url"


class TestAdapterRegistry:
    """Tests for adapter registry."""

    def test_get_adapter_openai_chat(self):
        adapter = get_adapter("openai-chat")
        assert isinstance(adapter, OpenAIChatAdapter)

    def test_get_adapter_vllm(self):
        adapter = get_adapter("vllm")
        assert isinstance(adapter, VLLMAdapter)

    def test_get_adapter_case_insensitive(self):
        adapter1 = get_adapter("OpenAI-Chat")
        adapter2 = get_adapter("VLLM")

        assert isinstance(adapter1, OpenAIChatAdapter)
        assert isinstance(adapter2, VLLMAdapter)

    def test_get_adapter_unknown_raises_error(self):
        try:
            get_adapter("unknown-adapter")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown adapter: unknown-adapter" in str(e)
            assert "Available:" in str(e)

    def test_list_adapters(self):
        adapters = list_adapters()

        assert "openai-chat" in adapters
        assert "vllm" in adapters
        assert isinstance(adapters, list)

    def test_register_custom_adapter(self):
        # Create a custom adapter class
        class CustomAdapter(OpenAIChatAdapter):
            pass

        # Register it
        register_adapter("custom", CustomAdapter)

        # Verify it's registered
        adapters = list_adapters()
        assert "custom" in adapters

        # Verify we can get it
        adapter = get_adapter("custom")
        assert isinstance(adapter, CustomAdapter)

    def test_register_adapter_case_insensitive(self):
        class AnotherAdapter(OpenAIChatAdapter):
            pass

        register_adapter("MyAdapter", AnotherAdapter)

        # Should be stored lowercase
        adapter = get_adapter("myadapter")
        assert isinstance(adapter, AnotherAdapter)
