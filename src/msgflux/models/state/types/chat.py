"""Core message types for ModelState."""

import hashlib
import time
from typing import Any

import msgspec

from msgflux.models.state.types.enums import LifecycleType, Role
from msgflux.models.state.types.tool import ToolCall, ToolResult
from msgflux.types.content import ContentBlock, TextContent


class Reasoning(msgspec.Struct, kw_only=True):
    """Reasoning/thinking content from the model."""

    content: str
    budget_tokens: int | None = None
    summary: str | None = None


class MessageMetadata(msgspec.Struct, kw_only=True):
    """Metadata attached to a message."""

    timestamp: float = msgspec.field(default_factory=time.time)

    model: str | None = None
    usage: dict[str, int] | None = None
    custom: dict[str, Any] = msgspec.field(default_factory=dict)

    lifecycle: LifecycleType = LifecycleType.PERMANENT
    ttl_turns: int | None = None
    scope_id: str | None = None
    importance: float = 1.0
    summarizable_group: str | None = None


class ChatMessage(msgspec.Struct, kw_only=True):
    """A single message in the conversation.

    Named ChatMessage to avoid conflict with msgflux.message.Message.
    """

    index: int = 0
    role: Role
    content: list[ContentBlock] | None = None
    reasoning: Reasoning | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    metadata: MessageMetadata = msgspec.field(default_factory=MessageMetadata)

    @property
    def hash(self) -> str:
        """Compute content hash for version control."""
        data = msgspec.json.encode(
            {
                "role": self.role.value,
                "content": (
                    [msgspec.to_builtins(c) for c in self.content]
                    if self.content
                    else None
                ),
                "reasoning": (
                    msgspec.to_builtins(self.reasoning) if self.reasoning else None
                ),
                "tool_calls": (
                    [msgspec.to_builtins(t) for t in self.tool_calls]
                    if self.tool_calls
                    else None
                ),
                "tool_result": (
                    msgspec.to_builtins(self.tool_result) if self.tool_result else None
                ),
            }
        )
        return hashlib.sha256(data).hexdigest()[:12]

    @property
    def text(self) -> str | None:
        """Get text content as a single string."""
        if not self.content:
            return None
        texts = [c.text for c in self.content if isinstance(c, TextContent)]
        return "\n".join(texts) if texts else None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None

    @property
    def is_ephemeral(self) -> bool:
        return self.metadata.lifecycle in (
            LifecycleType.EPHEMERAL_TURNS,
            LifecycleType.EPHEMERAL_SCOPE,
        )

    def with_index(self, index: int) -> "ChatMessage":
        """Create a copy with updated index."""
        return msgspec.structs.replace(self, index=index)

    def with_lifecycle(
        self,
        lifecycle: LifecycleType,
        ttl_turns: int | None = None,
        scope_id: str | None = None,
        importance: float = 1.0,
    ) -> "ChatMessage":
        """Create a copy with updated lifecycle settings."""
        new_metadata = msgspec.structs.replace(
            self.metadata,
            lifecycle=lifecycle,
            ttl_turns=ttl_turns,
            scope_id=scope_id,
            importance=importance,
        )
        return msgspec.structs.replace(self, metadata=new_metadata)
