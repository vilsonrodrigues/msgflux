"""Importance-based compaction policy."""

from msgflux.models.state.policies.base import CompactionPolicy
from msgflux.models.state.policies.types import PolicyResult
from msgflux.models.state.types import ChatMessage


class ImportancePolicy(CompactionPolicy):
    """Compaction based on message importance."""

    def apply(self, messages: list[ChatMessage]) -> PolicyResult:
        if not self.config.max_messages:
            return PolicyResult(messages=messages, stats={"compacted": False})

        target = self.config.max_messages
        if len(messages) <= target:
            return PolicyResult(messages=messages, stats={"compacted": False})

        protected = []
        candidates = []
        total = len(messages)

        for i, msg in enumerate(messages):
            if self._is_protected(msg, i, total):
                protected.append((i, msg))
            else:
                candidates.append((i, msg))

        to_remove = len(messages) - target
        removed = []

        if to_remove > 0 and candidates:
            candidates.sort(key=lambda x: x[1].metadata.importance)

            num_to_remove = min(to_remove, len(candidates))
            removed.extend(candidates[i][1] for i in range(num_to_remove))
            candidates = candidates[num_to_remove:]

        all_msgs = protected + candidates
        all_msgs.sort(key=lambda x: x[0])

        result = [msg.with_index(i) for i, (_, msg) in enumerate(all_msgs)]

        return PolicyResult(
            messages=result,
            removed=removed,
            stats={
                "compacted": len(removed) > 0,
                "removed_count": len(removed),
                "protected_count": len(protected),
            },
        )
