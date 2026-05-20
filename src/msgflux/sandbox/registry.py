from typing import TYPE_CHECKING

from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.sandbox.base import BaseSandbox

sandbox_registry = AutoloadRegistry("msgflux.sandbox.providers")


def register_sandbox(cls: type["BaseSandbox"]):
    sandbox_type = getattr(cls, "sandbox_type", None)
    provider = getattr(cls, "provider", None)

    if not sandbox_type or not provider:
        raise ValueError(f"{cls.__name__} must define `sandbox_type` and `provider`.")

    sandbox_registry.setdefault(sandbox_type, {})[provider] = cls
    return cls
