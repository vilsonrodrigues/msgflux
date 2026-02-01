"""Base class for all environment implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseEnvironment(ABC):
    """Abstract base class for all environments.

    An environment provides isolated execution of actions with support for
    tool injection and variable management. This is the root class for all
    environment types (code, browser, terminal, etc.).
    """

    # Class-level attributes to be defined by subclasses
    environment_type: str = "base"
    name: str = "environment"

    @abstractmethod
    def __call__(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an action in the environment.

        Args:
            action:
                The action to execute.
            timeout:
                Optional execution timeout in seconds.
            vars:
                Optional dictionary of variables to inject.

        Returns:
            Result of the action execution.
        """
        raise NotImplementedError

    @abstractmethod
    async def acall(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute an action asynchronously.

        Args:
            action:
                The action to execute.
            timeout:
                Optional execution timeout in seconds.
            vars:
                Optional dictionary of variables to inject.

        Returns:
            Result of the action execution.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset environment state."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the environment and release resources."""
        raise NotImplementedError

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit with cleanup."""
        self.shutdown()
