"""Factory functions for creating messages."""

from msgflux.models.state.types.chat import ChatMessage, MessageMetadata, Reasoning
from msgflux.models.state.types.enums import Role
from msgflux.models.state.types.tool import ToolCall, ToolResult
from msgflux.types.content import ContentBlock, TextContent


def user_message(
    content: str | list[ContentBlock],
    index: int = 0,
    **metadata_kwargs,
) -> ChatMessage:
    """Create a user message."""
    if isinstance(content, str):
        content = [TextContent(text=content)]
    return ChatMessage(
        index=index,
        role=Role.USER,
        content=content,
        metadata=MessageMetadata(**metadata_kwargs),
    )


def assistant_message(
    content: str | list[ContentBlock] | None = None,
    reasoning: str | Reasoning | None = None,
    tool_calls: list[ToolCall] | None = None,
    index: int = 0,
    **metadata_kwargs,
) -> ChatMessage:
    """Create an assistant message."""
    if isinstance(content, str):
        content = [TextContent(text=content)]
    if isinstance(reasoning, str):
        reasoning = Reasoning(content=reasoning)
    return ChatMessage(
        index=index,
        role=Role.ASSISTANT,
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        metadata=MessageMetadata(**metadata_kwargs),
    )


def system_message(
    content: str,
    index: int = 0,
    **metadata_kwargs,
) -> ChatMessage:
    """Create a system message."""
    return ChatMessage(
        index=index,
        role=Role.SYSTEM,
        content=[TextContent(text=content)],
        metadata=MessageMetadata(**metadata_kwargs),
    )


def tool_message(
    call_id: str,
    content: str,
    name: str | None = None,
    is_error: bool = False,
    index: int = 0,
    **metadata_kwargs,
) -> ChatMessage:
    """Create a tool result message."""
    return ChatMessage(
        index=index,
        role=Role.TOOL,
        tool_result=ToolResult(
            call_id=call_id, content=content, name=name, is_error=is_error
        ),
        metadata=MessageMetadata(**metadata_kwargs),
    )
