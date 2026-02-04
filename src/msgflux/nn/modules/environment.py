"""Environment module for executing actions in isolated environments."""

from typing import Any, Callable, Dict, Optional

from msgflux.environments.base import BaseEnvironment
from msgflux.environments.code.response import ExecutionResult
from msgflux.nn.modules.module import Module


class Environment(Module):
    r"""Environment for executing actions in isolated environments.

    This module wraps an environment (e.g., code, browser, terminal) and provides
    a consistent interface for execution with type-specific parameter handling.

    For code environments, tools can be provided at initialization or per-execution.
    Tools passed at execution time override init tools with the same name.

    Example:
        >>> from msgflux.nn import Environment
        >>> from msgflux.environments import Environments
        >>>
        >>> # Create code environment with default tools
        >>> def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> env = Environment(
        ...     environment=Environments.code("python/deno_pyodide"),
        ...     tools={"search": search}
        ... )
        >>>
        >>> # Execute code - init tools are available
        >>> result = env("data = search('python')\nprint(data)")
        >>> print(result.output)  # "Results for: python"
        >>>
        >>> # Pass additional/override tools at execution time
        >>> result = env(
        ...     "x = add(1, 2)",
        ...     tools={"add": lambda a, b: a + b}
        ... )
    """

    def __init__(
        self,
        environment: BaseEnvironment,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        """Initialize the Environment module.

        Args:
            environment:
                An environment instance for execution (e.g., DenoPyodideSandbox).
                The environment should have `name` and `environment_type` attributes.
            tools:
                Optional dictionary mapping tool names to callables.
                Only applicable for code environments.
                These tools will be available for all executions unless
                overridden by tools passed to forward/acall.

        Example:
            >>> from msgflux.environments import Environments
            >>> env = Environment(
            ...     environment=Environments.code("python/deno_pyodide"),
            ...     tools={"search": search_fn, "calculate": calc_fn}
            ... )
        """
        super().__init__()
        self._environment = environment
        self._tools: Dict[str, Callable] = tools or {}

    @property
    def environment(self) -> BaseEnvironment:
        """Get the underlying environment instance."""
        return self._environment

    @property
    def environment_type(self) -> str:
        """Get the environment type (e.g., 'python', 'browser')."""
        return self._environment.environment_type

    @property
    def name(self) -> str:
        """Get the environment name (e.g., 'execute_code')."""
        return self._environment.name

    @property
    def tools(self) -> Dict[str, Callable]:
        """Get the default tools configured at initialization (code environments only)."""
        return self._tools.copy()

    def forward(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        r"""Execute an action in the environment.

        Args:
            action:
                The action to execute.
            tools:
                Optional dictionary of tools (code environments only).
            vars:
                Optional dictionary of variables (code environments only).
            **kwargs:
                Additional parameters for specific environment types.

        Returns:
            Result from the environment execution.
        """
        if self.environment_type == "python":
            return self._execute_code(action, tools=tools, vars=vars)
        else:
            return self._environment(action, **kwargs)

    async def aforward(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        """Execute an action asynchronously in the environment.

        Args:
            action:
                The action to execute.
            tools:
                Optional dictionary of tools (code environments only).
            vars:
                Optional dictionary of variables (code environments only).
            **kwargs:
                Additional parameters for specific environment types.

        Returns:
            Result from the environment execution.
        """
        if self.environment_type == "python":
            return await self._aexecute_code(action, tools=tools, vars=vars)
        else:
            return await self._environment.acall(action, **kwargs)

    def _execute_code(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute code in a code environment.

        Args:
            action:
                The code to execute.
            tools:
                Optional dictionary of tools to make available.
            vars:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        merged_tools = self._merge_tools(tools)
        return self._environment(action, vars=vars, tools=merged_tools)

    async def _aexecute_code(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute code asynchronously in a code environment.

        Args:
            action:
                The code to execute.
            tools:
                Optional dictionary of tools to make available.
            vars:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        merged_tools = self._merge_tools(tools)
        return await self._environment.acall(action, vars=vars, tools=merged_tools)

    def _merge_tools(
        self, tools: Optional[Dict[str, Callable[..., Any]]]
    ) -> Optional[Dict[str, Callable[..., Any]]]:
        """Merge init tools with execution tools.

        Execution tools override init tools with the same name.

        Args:
            tools: Tools passed at execution time.

        Returns:
            Merged tools dictionary, or None if no tools.
        """
        if not self._tools and not tools:
            return None

        if not tools:
            return self._tools

        if not self._tools:
            return tools

        return {**self._tools, **tools}

    def reset(self) -> None:
        """Reset the environment state."""
        self._environment.reset()

    def shutdown(self) -> None:
        """Shutdown the environment and release resources."""
        self._environment.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit with cleanup."""
        self.shutdown()
