"""Sandbox providers."""

from msgflux.environments.sandboxes.providers.mock import MockSandbox

__all__ = [
    "MockSandbox",
]

if __all__ != sorted(__all__):
    raise RuntimeError("__all__ must be sorted")
