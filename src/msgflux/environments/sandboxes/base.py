"""Base classes for sandbox implementations."""

import asyncio
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Union

from msgflux._private.client import BaseClient

if TYPE_CHECKING:
    from msgflux.environments.sandboxes.response import ExecutionResult

# Type alias for tool functions
ToolFunction = Callable[..., Any]


class BaseSandbox(BaseClient):
    """Base class for all sandbox implementations."""

    msgflux_type = "sandbox"
    to_ignore = ["_process", "_lock", "_owner_thread", "_tools"]

    def instance_type(self) -> Mapping[str, str]:
        """Return instance type metadata.

        Returns:
            Dictionary with sandbox_type.
        """
        return {"sandbox_type": self.sandbox_type}

    @abstractmethod
    def _initialize(self):
        """Initialize the sandbox runtime.

        This method is called during the deserialization process to ensure
        that the sandbox is properly initialized after its state has been
        restored.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Execute code in the sandbox.

        Args:
            code:
                The code to execute.
            timeout:
                Optional execution timeout in seconds.
            variables:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with output, errors, and variables.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    async def acall(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Execute code in the sandbox asynchronously.

        This method runs the synchronous execution in a thread pool
        to avoid blocking the event loop.

        Args:
            code:
                The code to execute.
            timeout:
                Optional execution timeout in seconds.
            variables:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self(code, timeout=timeout, variables=variables),
        )

    @abstractmethod
    def mount_file(self, path: str, content: Union[str, bytes]) -> None:
        """Mount a file into the sandbox filesystem.

        Args:
            path:
                The virtual path for the file in the sandbox.
            content:
                The file content as string or bytes.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def get_variable(self, name: str) -> Any:
        """Retrieve a variable from the sandbox.

        Args:
            name:
                The name of the variable to retrieve.

        Returns:
            The value of the variable, or None if not found.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox.

        Args:
            name:
                The name of the variable.
            value:
                The value to set.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset sandbox state.

        Clears all variables and mounted files, returning the sandbox
        to its initial state.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the sandbox.

        Releases all resources and terminates the sandbox process.

        Raises:
            NotImplementedError:
                If the method is not implemented by the subclass.
        """
        raise NotImplementedError

    def register_tool(self, name: str, func: ToolFunction) -> None:
        """Register a tool function that can be called from inside the sandbox.

        Tools are Python functions that execute on the host and can be called
        from code running inside the sandbox. This enables the sandbox to
        interact with external systems (LLMs, APIs, databases, etc.) while
        maintaining isolation.

        Args:
            name:
                The name to use for the tool inside the sandbox.
            func:
                The Python function to register. Must be callable and its
                return value must be JSON-serializable.

        Example:
            >>> def search(query: str) -> str:
            ...     return f"Results for: {query}"
            >>> sandbox.register_tool("search", search)
            >>> sandbox("result = search('python tutorial')")
        """
        if not hasattr(self, "_tools"):
            self._tools: Dict[str, ToolFunction] = {}
        self._tools[name] = func

    def register_tools(self, tools: Dict[str, ToolFunction]) -> None:
        """Register multiple tools at once.

        Args:
            tools:
                Dictionary mapping tool names to functions.

        Example:
            >>> sandbox.register_tools({
            ...     "search": search_function,
            ...     "llm": llm_function,
            ... })
        """
        for name, func in tools.items():
            self.register_tool(name, func)

    @property
    def tools(self) -> Dict[str, ToolFunction]:
        """Get registered tools.

        Returns:
            Dictionary of registered tool names to functions.
        """
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit with cleanup."""
        self.shutdown()


class BasePythonSandbox(BaseSandbox):
    """Base class for Python-specific sandboxes."""

    sandbox_type = "python"
    name = "execute_code"  # Default name for tool identification

    def install_package(self, package: str) -> bool:
        """Install a Python package in the sandbox.

        Args:
            package:
                The package name to install.

        Returns:
            True if installation succeeded, False otherwise.

        Raises:
            NotImplementedError:
                If package installation is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support package installation."
        )

    def list_packages(self) -> List[str]:
        """List installed packages in the sandbox.

        Returns:
            List of installed package names.

        Raises:
            NotImplementedError:
                If package listing is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support package listing."
        )
