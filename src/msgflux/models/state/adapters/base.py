"""Base adapter interface for provider message conversion."""

from abc import ABC, abstractmethod
from typing import Any

from msgflux.models.state.types import ChatMessage


class MessageAdapter(ABC):
    """Base class for provider message adapters."""

    @abstractmethod
    def to_provider_format(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Convert internal messages to provider format."""
        ...

    @abstractmethod
    def from_provider_response(
        self,
        response: Any,
        **kwargs,
    ) -> ChatMessage:
        """Convert provider response to internal ChatMessage."""
        ...
