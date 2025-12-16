"""Base compaction policy class."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from msgflux.models.state.policies.types import Policy, PolicyResult
from msgflux.models.state.types import ChatMessage, Role


class CompactionPolicy(ABC):
    """Base class for compaction policies."""

    def __init__(
        self,
        config: Policy | None = None,
        summarizer: Callable[[list[ChatMessage]], str] | None = None,
    ):
        self.config = config or Policy()
        self.summarizer = summarizer

    @abstractmethod
    def apply(self, messages: list[ChatMessage]) -> PolicyResult:
        """Apply the policy to messages."""
        ...

    def needs_compaction(
        self,
        messages: list[ChatMessage],
        token_count: int = 0,
    ) -> bool:
        """Check if compaction is needed."""
        return (
            (self.config.max_messages and len(messages) > self.config.max_messages)
            or (self.config.max_tokens and token_count > self.config.max_tokens)
        )

    def _is_protected(self, msg: ChatMessage, index: int, total: int) -> bool:
        """Check if a message is protected from removal."""
        if self.config.preserve_system and msg.role == Role.SYSTEM:
            return True

        if msg.metadata.importance >= self.config.min_importance:
            return True

        if index < self.config.preserve_first_n:
            return True

        if index >= total - self.config.preserve_last_n:
            return True

        if self.config.preserve_first_pct > 0:
            first_boundary = int(total * self.config.preserve_first_pct)
            if index < first_boundary:
                return True

        if self.config.preserve_last_pct > 0:
            last_boundary = int(total * (1 - self.config.preserve_last_pct))
            if index >= last_boundary:
                return True

        return False
