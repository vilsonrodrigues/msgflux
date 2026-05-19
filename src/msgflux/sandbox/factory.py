from __future__ import annotations

from typing import Any

from msgflux.sandbox.base import BaseSandbox, BaseShellSandbox, LocalPythonSandbox


class Sandbox:
    """Factory for sandboxed runtime components."""

    @staticmethod
    def code(identifier: str = "python/local") -> BaseSandbox:
        if identifier in {"python", "python/local"}:
            return LocalPythonSandbox()
        if identifier == "python/monty":
            return _load_monty_sandbox()
        raise ValueError(f"Unknown code sandbox: {identifier}")

    @staticmethod
    def python(provider: str = "local") -> BaseSandbox:
        return Sandbox.code(f"python/{provider}")

    @staticmethod
    def shell(provider: str = "just-bash", **kwargs: Any) -> BaseShellSandbox:
        if provider in {"just-bash", "local"}:
            return _load_just_bash_sandbox(**kwargs)
        raise ValueError(f"Unknown shell sandbox: {provider}")


def _load_monty_sandbox() -> BaseSandbox:
    try:
        from msgflux.sandbox.providers.monty import MontySandbox  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "`python/monty` sandbox requires the optional Monty provider."
        ) from exc
    return MontySandbox()


def _load_just_bash_sandbox(**kwargs: Any) -> BaseShellSandbox:
    try:
        from msgflux.sandbox.providers.just_bash import (  # noqa: PLC0415
            JustBashSandbox,
        )
    except ImportError as exc:
        raise ImportError(
            "`shell/just-bash` sandbox requires the optional shell provider. "
            "Install it with `msgflux[shell]`."
        ) from exc
    return JustBashSandbox(**kwargs)
