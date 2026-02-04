"""Registry for code environment providers."""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from msgflux.environments.code.base import BaseCodeEnvironment

# Registry: environment_registry[environment_type][provider] = cls
environment_registry: Dict[str, Dict[str, type]] = {}


def register_environment(cls: "type[BaseCodeEnvironment]"):
    """Decorator to register an environment provider.

    Args:
        cls:
            The environment class to register.

    Returns:
        The registered class.

    Raises:
        ValueError:
            If the class does not define environment_type and provider.
    """
    environment_type = getattr(cls, "environment_type", None)
    provider = getattr(cls, "provider", None)

    if not environment_type or not provider:
        raise ValueError(
            f"{cls.__name__} must define `environment_type` and `provider`."
        )

    environment_registry.setdefault(environment_type, {})[provider] = cls
    return cls


class Environments:
    """Factory for creating code environments.

    This class provides a unified interface for creating different types
    of code execution environments.

    Example:
        >>> from msgflux.environments import Environments
        >>> env = Environments.code("python/deno_pyodide")
        >>> result = env("print('Hello, World!')")
    """

    @staticmethod
    def code(
        environment_id: str = "python",
        **kwargs: Any,
    ) -> "BaseCodeEnvironment":
        """Create a code execution environment.

        Args:
            environment_id:
                Environment identifier in format "type/provider" or just "type".
                Examples: "python/deno_pyodide", "python"
            **kwargs:
                Additional arguments passed to the environment constructor.

        Returns:
            An instance of the requested environment.

        Raises:
            ValueError:
                If the environment type or provider is not registered.

        Example:
            >>> env = Environments.code("python/deno_pyodide")
            >>> env = Environments.code("python")  # uses default provider
            >>> env = Environments.code("python/deno_pyodide", timeout=30.0)
        """
        # Parse environment_id: "type/provider" or just "type"
        if "/" in environment_id:
            environment_type, provider = environment_id.split("/", 1)
        else:
            environment_type = environment_id
            provider = None

        if environment_type not in environment_registry:
            available = list(environment_registry.keys())
            raise ValueError(
                f"Unknown environment type: '{environment_type}'. "
                f"Available types: {available}"
            )

        providers = environment_registry[environment_type]

        if provider is None:
            # Use first registered provider
            provider = next(iter(providers.keys()))

        if provider not in providers:
            available = [f"{environment_type}/{p}" for p in providers.keys()]
            raise ValueError(
                f"Unknown provider '{provider}' for environment type "
                f"'{environment_type}'. Available: {available}"
            )

        cls = providers[provider]
        return cls(**kwargs)

    @staticmethod
    def list_types() -> list:
        """List available environment types.

        Returns:
            List of registered environment type names.
        """
        return list(environment_registry.keys())

    @staticmethod
    def list_providers(environment_type: str) -> list:
        """List available providers for an environment type.

        Args:
            environment_type:
                The environment type to query.

        Returns:
            List of registered provider names.
        """
        if environment_type not in environment_registry:
            return []
        return list(environment_registry[environment_type].keys())
