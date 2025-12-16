"""OpenAI Chat Adapter for message format conversion."""

from typing import Any

import msgspec

from msgflux.models.state.adapters.base import MessageAdapter
from msgflux.models.state.types import (
    AudioContent,
    ChatMessage,
    ContentBlock,
    FileContent,
    ImageContent,
    Reasoning,
    Role,
    TextContent,
    ToolCall,
    VideoContent,
    assistant_message,
)


class OpenAIChatAdapter(MessageAdapter):
    """Adapter for OpenAI chat completions API.

    Also works with OpenAI-compatible APIs: vLLM, Mistral, Groq, Together, etc.
    """

    def to_provider_format(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Convert to OpenAI ChatML format."""
        result = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            converted = self._convert_message(msg)
            if converted:
                result.append(converted)

        return {"messages": result}

    def _convert_message(self, msg: ChatMessage) -> dict[str, Any] | None:
        """Convert single message to OpenAI format."""
        if msg.role == Role.SYSTEM:
            return {"role": "system", "content": msg.text or ""}

        if msg.role == Role.USER:
            content = self._convert_content(msg.content)
            return {"role": "user", "content": content}

        if msg.role == Role.ASSISTANT:
            result: dict[str, Any] = {"role": "assistant"}

            if msg.reasoning:
                result["reasoning_content"] = msg.reasoning.content

            if msg.content:
                result["content"] = self._convert_content(msg.content)
            elif not msg.tool_calls:
                result["content"] = ""

            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments_json()},
                    }
                    for tc in msg.tool_calls
                ]

            return result

        if msg.role == Role.TOOL:
            if msg.tool_result:
                return {
                    "role": "tool",
                    "tool_call_id": msg.tool_result.call_id,
                    "content": msg.tool_result.content,
                }
            return None

        return None

    def _convert_content(
        self,
        content: list[ContentBlock] | None,
    ) -> str | list[dict[str, Any]]:
        """Convert content blocks to OpenAI format."""
        if not content:
            return ""

        if len(content) == 1 and isinstance(content[0], TextContent):
            return content[0].text

        result = []
        for block in content:
            if isinstance(block, TextContent):
                result.append(self._convert_text_content(block))
            elif isinstance(block, ImageContent):
                result.append(self._convert_image_content(block))
            elif isinstance(block, AudioContent):
                result.append(self._convert_audio_content(block))
            elif isinstance(block, VideoContent):
                result.append(self._convert_video_content(block))
            elif isinstance(block, FileContent):
                result.append(self._convert_file_content(block))

        return result if result else ""

    def _convert_text_content(self, block: TextContent) -> dict[str, Any]:
        """Convert text content block."""
        return {"type": "text", "text": block.text}

    def _convert_image_content(self, block: ImageContent) -> dict[str, Any]:
        """Convert image content block."""
        url = block.url
        if block.base64 and not url:
            media_type = block.media_type or "image/png"
            url = f"data:{media_type};base64,{block.base64}"

        img_block: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": url},
        }
        if block.detail:
            img_block["image_url"]["detail"] = block.detail
        return img_block

    def _convert_audio_content(self, block: AudioContent) -> dict[str, Any]:
        """Convert audio content block."""
        if block.base64:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": block.base64,
                    "format": block.format or "wav",
                },
            }
        if block.url:
            return {"type": "audio_url", "audio_url": {"url": block.url}}
        return {}

    def _convert_video_content(self, block: VideoContent) -> dict[str, Any]:
        """Convert video content block."""
        url = block.url
        if block.base64 and not url:
            media_type = block.media_type or "video/mp4"
            url = f"data:{media_type};base64,{block.base64}"
        return {"type": "video_url", "video_url": {"url": url}}

    def _convert_file_content(self, block: FileContent) -> dict[str, Any]:
        """Convert file content block."""
        return {
            "type": "file",
            "file": {"filename": block.filename, "file_data": block.data},
        }

    def from_provider_response(
        self,
        response: Any,
        **kwargs,
    ) -> ChatMessage:
        """Convert OpenAI response to internal ChatMessage."""
        if hasattr(response, "choices"):
            choice = response.choices[0]
            msg = choice.message
        else:
            msg = response

        content = None
        if hasattr(msg, "content") and msg.content:
            content = [TextContent(text=msg.content)]

        reasoning = None
        for attr in ("reasoning_content", "reasoning", "thinking"):
            if hasattr(msg, attr):
                reasoning_text = getattr(msg, attr)
                if reasoning_text:
                    reasoning = Reasoning(content=reasoning_text)
                    break

        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = msgspec.json.decode(args.encode())
                    except Exception:
                        args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input": getattr(response.usage, "prompt_tokens", 0),
                "output": getattr(response.usage, "completion_tokens", 0),
            }

        return assistant_message(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            model=kwargs.get("model"),
            usage=usage,
        )
