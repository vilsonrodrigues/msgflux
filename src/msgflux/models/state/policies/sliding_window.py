"""Sliding window compaction policy."""

from msgflux.models.state.policies.base import CompactionPolicy
from msgflux.models.state.policies.types import PolicyResult
from msgflux.models.state.types import ChatMessage, Role, TextContent


class SlidingWindowPolicy(CompactionPolicy):
    """Simple sliding window - keep last N messages."""

    def apply(self, messages: list[ChatMessage]) -> PolicyResult:
        if not self.config.max_messages:
            return PolicyResult(messages=messages, stats={"compacted": False})

        target = self.config.max_messages
        if len(messages) <= target:
            return PolicyResult(messages=messages, stats={"compacted": False})

        system_msgs = []
        other_msgs = []

        for msg in messages:
            if self.config.preserve_system and msg.role == Role.SYSTEM:
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        to_keep = max(target - len(system_msgs), 0)

        removed = other_msgs[:-to_keep] if to_keep > 0 else other_msgs
        kept = other_msgs[-to_keep:] if to_keep > 0 else []

        summary = None
        summarized = []
        if (
            removed
            and self.summarizer
            and len(removed) >= self.config.summarize_threshold
        ):
            summary = self.summarizer(removed)
            summarized = removed

            summary_msg = ChatMessage(
                index=0,
                role=Role.ASSISTANT,
                content=[TextContent(text=f"[Earlier Context]\n{summary}")],
            )
            system_msgs.append(summary_msg)

        result = system_msgs + kept
        result = [msg.with_index(i) for i, msg in enumerate(result)]

        return PolicyResult(
            messages=result,
            removed=removed if not summarized else [],
            summarized=summarized,
            summary=summary,
            stats={
                "compacted": True,
                "removed_count": len(removed),
                "system_kept": len(system_msgs),
            },
        )
