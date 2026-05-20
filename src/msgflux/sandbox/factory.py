from __future__ import annotations

from typing import Any

from msgflux.sandbox.base import BaseSandbox, BaseShellSandbox
from msgflux.sandbox.registry import sandbox_registry


class Sandbox:
    """Factory for sandboxed runtime components."""

    @staticmethod
    def code(identifier: str = "python/local") -> BaseSandbox:
        sandbox_type, provider = Sandbox._sandbox_path_parser(identifier)
        return Sandbox._create_sandbox(sandbox_type, provider)

    @staticmethod
    def python(provider: str = "local") -> BaseSandbox:
        return Sandbox._create_sandbox("python", provider)

    @staticmethod
    def shell(provider: str = "just-bash", **kwargs: Any) -> BaseShellSandbox:
        sandbox = Sandbox._create_sandbox("shell", provider, **kwargs)
        if not isinstance(sandbox, BaseShellSandbox):
            raise TypeError(f"Sandbox `shell/{provider}` is not a shell sandbox.")
        return sandbox

    @staticmethod
    def providers() -> dict[str, list[str]]:
        return {k: list(v.keys()) for k, v in sandbox_registry.items()}

    @staticmethod
    def sandbox_types() -> list[str]:
        return list(sandbox_registry.keys())

    @staticmethod
    def _sandbox_path_parser(identifier: str) -> tuple[str, str]:
        if "/" not in identifier:
            return identifier, "local"
        sandbox_type, provider = identifier.split("/", 1)
        return sandbox_type, provider

    @staticmethod
    def _get_sandbox_class(sandbox_type: str, provider: str) -> type[BaseSandbox]:
        if sandbox_type not in sandbox_registry:
            raise ValueError(f"Sandbox type `{sandbox_type}` is not supported")
        if provider not in sandbox_registry[sandbox_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for sandbox `{sandbox_type}`"
            )
        return sandbox_registry[sandbox_type][provider]

    @staticmethod
    def _create_sandbox(
        sandbox_type: str,
        provider: str,
        **kwargs: Any,
    ) -> BaseSandbox:
        sandbox_cls = Sandbox._get_sandbox_class(sandbox_type, provider)
        return sandbox_cls(**kwargs)
