"""Policy configuration and result types."""

from typing import Any

import msgspec

from msgflux.models.state.types import ChatMessage


class Policy(msgspec.Struct, kw_only=True):
    """Unified policy configuration.

    Example:
        # Sliding window
        policy = Policy(type="sliding_window", max_messages=50)

        # Position-based
        policy = Policy(
            type="position",
            preserve_first_pct=0.1,
            preserve_last_pct=0.2,
        )

        # Lifecycle-based
        policy = Policy(type="lifecycle")
    """

    type: str = "sliding_window"

    max_messages: int | None = None
    max_tokens: int | None = None

    preserve_system: bool = True
    preserve_first_n: int = 0
    preserve_last_n: int = 3
    preserve_first_pct: float = 0.0
    preserve_last_pct: float = 0.0

    min_importance: float = 0.9
    summarize_threshold: int = 5
    summarize_tool_loops: bool = False


class PolicyResult(msgspec.Struct, kw_only=True):
    """Result of applying a policy to messages."""

    messages: list[ChatMessage]
    removed: list[ChatMessage] = msgspec.field(default_factory=list)
    summarized: list[ChatMessage] = msgspec.field(default_factory=list)
    summary: str | None = None
    stats: dict[str, Any] = msgspec.field(default_factory=dict)
