"""Adapter registry for managing available message adapters."""

from msgflux.models.state.adapters.base import MessageAdapter
from msgflux.models.state.adapters.openai import OpenAIChatAdapter
from msgflux.models.state.adapters.vllm import VLLMAdapter

_ADAPTERS: dict[str, type] = {
    "openai-chat": OpenAIChatAdapter,
    "vllm": VLLMAdapter,
}


def get_adapter(name: str) -> MessageAdapter:
    """Get adapter by name.

    Args:
        name: Adapter name (openai-chat, vllm).

    Returns:
        Adapter instance.

    Raises:
        ValueError: If adapter not found.
    """
    adapter_cls = _ADAPTERS.get(name.lower())
    if adapter_cls is None:
        available = list(_ADAPTERS.keys())
        raise ValueError(f"Unknown adapter: {name}. Available: {available}")
    return adapter_cls()


def register_adapter(name: str, adapter_cls: type) -> None:
    """Register a custom adapter.

    Args:
        name: Adapter name.
        adapter_cls: Adapter class.
    """
    _ADAPTERS[name.lower()] = adapter_cls


def list_adapters() -> list[str]:
    """List available adapters."""
    return list(_ADAPTERS.keys())
