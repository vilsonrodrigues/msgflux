"""Sandbox module for secure code execution."""

from typing import Any, Dict, List, Type

from msgflux.environments.sandboxes import providers as _providers  # noqa: F401
from msgflux.environments.sandboxes.base import BasePythonSandbox, BaseSandbox
from msgflux.environments.sandboxes.registry import register_sandbox, sandbox_registry
from msgflux.environments.sandboxes.response import ExecutionResult


class Sandbox:
    """Factory class for creating sandbox instances."""

    @classmethod
    def providers(cls) -> Dict[str, List[str]]:
        """Get available providers by sandbox type.

        Returns:
            Dictionary mapping sandbox types to lists of provider names.
        """
        return {k: list(v.keys()) for k, v in sandbox_registry.items()}

    @classmethod
    def sandbox_types(cls) -> List[str]:
        """Get available sandbox types.

        Returns:
            List of sandbox type names.
        """
        return list(sandbox_registry.keys())

    @classmethod
    def _get_sandbox_class(cls, sandbox_type: str, provider: str) -> Type[BaseSandbox]:
        """Get sandbox class by type and provider.

        Args:
            sandbox_type:
                The type of sandbox (e.g., "python").
            provider:
                The provider name (e.g., "mock", "deno_pyodide").

        Returns:
            The sandbox class.

        Raises:
            ValueError:
                If sandbox type or provider is not registered.
        """
        if sandbox_type not in sandbox_registry:
            available = list(sandbox_registry.keys())
            raise ValueError(
                f"Sandbox type `{sandbox_type}` is not supported. "
                f"Available types: {available}"
            )
        if provider not in sandbox_registry[sandbox_type]:
            available = list(sandbox_registry[sandbox_type].keys())
            raise ValueError(
                f"Provider `{provider}` not registered for type `{sandbox_type}`. "
                f"Available providers: {available}"
            )
        return sandbox_registry[sandbox_type][provider]

    @classmethod
    def _create_sandbox(
        cls, sandbox_type: str, provider: str, **kwargs: Any
    ) -> BaseSandbox:
        """Create a sandbox instance.

        Args:
            sandbox_type:
                The type of sandbox.
            provider:
                The provider name.
            **kwargs:
                Provider-specific arguments.

        Returns:
            A sandbox instance.
        """
        sandbox_cls = cls._get_sandbox_class(sandbox_type, provider)
        return sandbox_cls(**kwargs)

    @classmethod
    def python(cls, provider: str = "mock", **kwargs: Any) -> BasePythonSandbox:
        """Create a Python sandbox.

        Args:
            provider:
                Sandbox provider to use. Available:
                - "mock": Mock sandbox for testing (no external deps)
                - "deno_pyodide": Secure sandbox using Deno + Pyodide (future)
            **kwargs:
                Provider-specific arguments.

        Returns:
            A Python sandbox instance.

        Example:
            >>> sandbox = Sandbox.python(provider="mock")
            >>> result = sandbox("print('Hello')")
            >>> print(result.output)
            Hello
        """
        return cls._create_sandbox("python", provider, **kwargs)


__all__ = [
    "BasePythonSandbox",
    "BaseSandbox",
    "ExecutionResult",
    "Sandbox",
    "register_sandbox",
]

if __all__ != sorted(__all__):
    raise RuntimeError("__all__ must be sorted")
