"""Position-based compaction policy."""

from msgflux.models.state.policies.base import CompactionPolicy
from msgflux.models.state.policies.types import PolicyResult
from msgflux.models.state.types import ChatMessage, Role, TextContent


class PositionBasedPolicy(CompactionPolicy):
    """Compaction based on message position.

    Keeps first X%, summarizes middle Y%, keeps last Z%.
    """

    def apply(self, messages: list[ChatMessage]) -> PolicyResult:
        total = len(messages)
        if total == 0:
            return PolicyResult(messages=[])

        first_pct = self.config.preserve_first_pct
        last_pct = self.config.preserve_last_pct

        first_boundary = max(self.config.preserve_first_n, int(total * first_pct))
        last_boundary = total - max(
            self.config.preserve_last_n, int(total * last_pct)
        )

        if first_boundary >= last_boundary:
            return PolicyResult(messages=messages, stats={"compacted": False})

        first_region = messages[:first_boundary]
        middle_region = messages[first_boundary:last_boundary]
        last_region = messages[last_boundary:]

        result_messages = list(first_region)
        summarized = []
        summary = None

        if (
            middle_region
            and self.summarizer
            and len(middle_region) >= self.config.summarize_threshold
        ):
            summary = self.summarizer(middle_region)
            summarized = list(middle_region)

            summary_msg = ChatMessage(
                index=len(result_messages),
                role=Role.ASSISTANT,
                content=[TextContent(text=f"[Context Summary]\n{summary}")],
            )
            result_messages.append(summary_msg)
        else:
            result_messages.extend(middle_region)

        for msg in last_region:
            result_messages.append(msg.with_index(len(result_messages)))

        return PolicyResult(
            messages=result_messages,
            summarized=summarized,
            summary=summary,
            stats={
                "compacted": bool(summarized),
                "first_kept": len(first_region),
                "middle_summarized": len(summarized),
                "last_kept": len(last_region),
            },
        )
