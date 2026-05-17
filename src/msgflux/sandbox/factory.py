from __future__ import annotations

from msgflux.sandbox.base import BaseSandbox, LocalPythonSandbox


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
    def shell(provider: str = "local") -> BaseSandbox:
        return Sandbox.code(f"shell/{provider}")


def _load_monty_sandbox() -> BaseSandbox:
    try:
        from msgflux.sandbox.providers.monty import MontySandbox  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "`python/monty` sandbox requires the optional Monty provider."
        ) from exc
    return MontySandbox()
