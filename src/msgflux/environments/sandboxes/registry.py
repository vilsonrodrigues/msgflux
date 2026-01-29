"""Registry for sandbox providers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.environments.sandboxes.base import BaseSandbox

sandbox_registry = {}  # sandbox_registry[sandbox_type][provider] = cls


def register_sandbox(cls: "type[BaseSandbox]"):
    """Decorator to register a sandbox provider.

    Args:
        cls:
            The sandbox class to register.

    Returns:
        The registered class.

    Raises:
        ValueError:
            If the class does not define sandbox_type and provider.
    """
    sandbox_type = getattr(cls, "sandbox_type", None)
    provider = getattr(cls, "provider", None)

    if not sandbox_type or not provider:
        raise ValueError(f"{cls.__name__} must define `sandbox_type` and `provider`.")

    sandbox_registry.setdefault(sandbox_type, {})[provider] = cls
    return cls
