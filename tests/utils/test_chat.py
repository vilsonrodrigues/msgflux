import msgspec
import pytest

from msgflux.types.content import (
    AudioContent,
    FileContent,
    ImageContent,
    TextContent,
    VideoContent,
)
from msgflux.utils.chat import (
    ChatBlock,
    adapt_messages_for_vllm_audio,
    clean_docstring,
    generate_json_schema,
    generate_tool_json_schema,
    hint_to_schema,
    parse_docstring_args,
    response_format_from_msgspec_struct,
)


class TestChatBlock:
    # Test new ModelState content type methods
    def test_text_block(self):
        """Test text content block creation."""
        result = ChatBlock.text("hello world")
        assert isinstance(result, TextContent)
        assert result.text == "hello world"

    def test_image_block(self):
        """Test image content block creation."""
        result = ChatBlock.image(
            url="http://example.com/image.jpg",
            detail="high"
        )
        assert isinstance(result, ImageContent)
        assert result.url == "http://example.com/image.jpg"
        assert result.detail == "high"

    def test_image_block_base64(self):
        """Test image content block with base64."""
        result = ChatBlock.image(
            base64="base64data",
            media_type="image/jpeg"
        )
        assert isinstance(result, ImageContent)
        assert result.base64 == "base64data"
        assert result.media_type == "image/jpeg"

    def test_audio_block(self):
        """Test audio content block creation."""
        result = ChatBlock.audio(
            url="http://example.com/audio.mp3",
            format="mp3"
        )
        assert isinstance(result, AudioContent)
        assert result.url == "http://example.com/audio.mp3"
        assert result.format == "mp3"

    def test_video_block(self):
        """Test video content block creation."""
        result = ChatBlock.video(
            url="http://example.com/video.mp4",
            media_type="video/mp4"
        )
        assert isinstance(result, VideoContent)
        assert result.url == "http://example.com/video.mp4"
        assert result.media_type == "video/mp4"

    def test_file_block(self):
        """Test file content block creation."""
        result = ChatBlock.file(
            filename="document.pdf",
            data="base64data",
            media_type="application/pdf"
        )
        assert isinstance(result, FileContent)
        assert result.filename == "document.pdf"
        assert result.data == "base64data"
        assert result.media_type == "application/pdf"

    # Test legacy dict-based methods
    def test_user_message(self):
        assert ChatBlock.user("hello") == {"role": "user", "content": "hello"}

    def test_assist_message(self):
        assert ChatBlock.assist("world") == {"role": "assistant", "content": "world"}

    def test_system_message(self):
        assert ChatBlock.system("system prompt") == {
            "role": "system",
            "content": "system prompt",
        }

    def test_tool_call(self):
        assert ChatBlock.tool_call("123", "my_tool", "{}") == {
            "id": "123",
            "type": "function",
            "function": {"name": "my_tool", "arguments": "{}"},
        }

    def test_assist_tool_calls(self):
        tool_calls = [ChatBlock.tool_call("123", "my_tool", "{}")]
        assert ChatBlock.assist_tool_calls(tool_calls) == {
            "role": "assistant",
            "tool_calls": tool_calls,
        }

    def test_tool_message(self):
        assert ChatBlock.tool("123", "tool output") == {
            "role": "tool",
            "tool_call_id": "123",
            "content": "tool output",
        }


class MyStruct(msgspec.Struct):
    a: int
    b: str


def test_response_format_from_msgspec_struct():
    schema = response_format_from_msgspec_struct(MyStruct)
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["name"] == "mystruct"
    assert schema["json_schema"]["schema"]["properties"]["a"]["type"] == "integer"


def test_hint_to_schema():
    assert hint_to_schema(int) == {"type": "integer"}
    assert hint_to_schema(str) == {"type": "string"}
    assert hint_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}


DOCSTRING = """
My function.

Args:
    a: first param.
    b (int): second param.
"""


def test_clean_docstring():
    assert clean_docstring(DOCSTRING) == "My function."


def test_parse_docstring_args():
    args = parse_docstring_args(DOCSTRING)
    assert args["a"] == "first param."
    assert args["b"] == "second param."


class MyTool:
    def get_module_name(self):
        return "my_tool"

    def get_module_description(self):
        return DOCSTRING

    def get_module_annotations(self):
        return {"a": str, "b": int}


def test_generate_json_schema():
    schema = generate_json_schema(MyTool())
    assert schema["name"] == "my_tool"
    assert schema["description"] == "My function."
    assert schema["parameters"]["properties"]["a"]["description"] == "first param."


def test_generate_tool_json_schema():
    schema = generate_tool_json_schema(MyTool())
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_tool"


def test_adapt_messages_for_vllm_audio():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "base64_data", "format": "mp3"},
                },
            ],
        }
    ]
    adapted = adapt_messages_for_vllm_audio(messages)
    assert adapted[0]["content"][1]["type"] == "audio_url"
    assert (
        "data:audio/mpeg;base64,base64_data"
        in adapted[0]["content"][1]["audio_url"]["url"]
    )
