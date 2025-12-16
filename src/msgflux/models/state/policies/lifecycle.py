"""Lifecycle-based compaction policy."""

from collections.abc import Callable

import msgspec

from msgflux.models.state.policies.base import CompactionPolicy
from msgflux.models.state.policies.types import Policy, PolicyResult
from msgflux.models.state.types import ChatMessage, LifecycleType, Role, TextContent


class LifecyclePolicy(CompactionPolicy):
    """Compaction based on message lifecycle."""

    def __init__(
        self,
        config: Policy | None = None,
        summarizer: Callable[[list[ChatMessage]], str] | None = None,
        current_turn: int = 0,
        current_scope: str | None = None,
    ):
        super().__init__(config, summarizer)
        self.current_turn = current_turn
        self.current_scope = current_scope

    def advance_turn(self):
        """Advance the turn counter."""
        self.current_turn += 1

    def set_scope(self, scope_id: str | None):
        """Set current scope."""
        self.current_scope = scope_id

    def apply(self, messages: list[ChatMessage]) -> PolicyResult:
        kept = []
        removed = []
        summarizable = []

        for msg in messages:
            lifecycle = msg.metadata.lifecycle

            if lifecycle == LifecycleType.EPHEMERAL_TURNS:
                ttl = msg.metadata.ttl_turns
                if ttl is not None and ttl <= 0:
                    removed.append(msg)
                    continue

            if lifecycle == LifecycleType.EPHEMERAL_SCOPE:
                if msg.metadata.scope_id != self.current_scope:
                    removed.append(msg)
                    continue

            if lifecycle == LifecycleType.SUMMARIZABLE:
                summarizable.append(msg)

            kept.append(msg)

        summary = None
        summarized = []
        if (
            summarizable
            and self.summarizer
            and len(summarizable) >= self.config.summarize_threshold
        ):
            summary = self.summarizer(summarizable)
            summarized = summarizable

            kept = [
                m for m in kept if m.metadata.lifecycle != LifecycleType.SUMMARIZABLE
            ]

            summary_msg = ChatMessage(
                index=len(kept),
                role=Role.ASSISTANT,
                content=[TextContent(text=f"[Summary]\n{summary}")],
            )
            kept.append(summary_msg)

        result = [msg.with_index(i) for i, msg in enumerate(kept)]

        return PolicyResult(
            messages=result,
            removed=removed,
            summarized=summarized,
            summary=summary,
            stats={
                "removed_count": len(removed),
                "summarized_count": len(summarized),
                "kept_count": len(result),
            },
        )

    def decrement_ttls(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Decrement TTL on ephemeral_turns messages."""
        result = []
        for msg in messages:
            if msg.metadata.lifecycle == LifecycleType.EPHEMERAL_TURNS:
                if msg.metadata.ttl_turns is not None and msg.metadata.ttl_turns > 0:
                    new_metadata = msgspec.structs.replace(
                        msg.metadata, ttl_turns=msg.metadata.ttl_turns - 1
                    )
                    msg = msgspec.structs.replace(msg, metadata=new_metadata)
            result.append(msg)
        return result
