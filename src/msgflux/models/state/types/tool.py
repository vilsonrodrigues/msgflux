"""Tool call and result types."""

from typing import Any

import msgspec


class ToolCall(msgspec.Struct, kw_only=True):
    """A tool/function call made by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any]

    def arguments_json(self) -> str:
        """Get arguments as JSON string."""
        return msgspec.json.encode(self.arguments).decode()


class ToolResult(msgspec.Struct, kw_only=True):
    """Result of a tool call execution."""

    call_id: str
    content: str
    is_error: bool = False
    name: str | None = None
