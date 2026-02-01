"""Environment module for code execution in isolated environments."""

from typing import Any, Callable, Dict, Optional

from msgflux.environments.base import BaseEnvironment
from msgflux.environments.code.response import ExecutionResult
from msgflux.nn.modules.module import Module


class Environment(Module):
    r"""Environment for secure code execution with tool injection.

    This module wraps a code environment (e.g., DenoPyodideSandbox) and provides
    a consistent interface for executing code with dynamically injected tools.

    The Environment follows the pattern used in DSPy's PythonInterpreter,
    allowing code to call host-side functions (tools) during execution.

    Tools can be provided at initialization (available for all executions) or
    passed per-execution (merged with init tools). Tools passed at execution
    time override init tools with the same name.

    Example:
        >>> from msgflux.nn import Environment
        >>> from msgflux.environments import Environments
        >>>
        >>> # Create environment with default tools
        >>> def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> env = Environment(
        ...     environment=Environments.code("python"),
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

    For RL and agent workflows:
        >>> # Tools can be passed dynamically from ToolLibrary
        >>> tool_funcs = {t.name: t.callable for t in tool_library}
        >>> result = await env.acall(action, tools=tool_funcs)
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
                A code environment instance for execution (e.g., DenoPyodideSandbox).
                The environment should have a `name` attribute that identifies it
                as a tool (e.g., "execute_code", "python_interpreter").
            tools:
                Optional dictionary mapping tool names to callables.
                These tools will be available for all executions unless
                overridden by tools passed to forward/acall.

        Example:
            >>> from msgflux.environments import Environments
            >>> env = Environment(
            ...     environment=Environments.code("python"),
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
    def name(self) -> str:
        """Get the environment name (e.g., 'execute_code')."""
        return self._environment.name

    @property
    def tools(self) -> Dict[str, Callable]:
        """Get the default tools configured at initialization."""
        return self._tools.copy()

    def forward(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        r"""Execute an action in the environment.

        Args:
            action:
                The action to execute (e.g., Python code).
            tools:
                Optional dictionary mapping tool names to callables.
                These are merged with init tools (overriding on conflict).
            vars:
                Optional dictionary of variables to inject into the
                execution context before running the action.

        Returns:
            ExecutionResult containing:
                - success: Whether execution succeeded
                - output: Captured stdout
                - error: Error message if failed
                - variables: Variables defined during execution
                - return_value: Return value of last expression
                - execution_time_ms: Execution time in milliseconds

        Example:
            >>> result = env("x = add(1, 2)", tools={"add": lambda a, b: a + b})
            >>> print(result.output)  # "3"
        """
        # Merge init tools with execution tools (execution tools override)
        merged_tools = self._merge_tools(tools)

        # Execute action with merged tools
        return self._environment(action, vars=vars, tools=merged_tools)

    async def aforward(
        self,
        action: str,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute an action asynchronously in the environment.

        This is the async version of forward(). The execution runs in a
        thread pool to avoid blocking the event loop.

        Args:
            action:
                The action to execute (e.g., Python code).
            tools:
                Optional dictionary mapping tool names to callables.
                These are merged with init tools (overriding on conflict).
            vars:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with execution details.

        Example:
            >>> async def main():
            ...     result = await env.acall("print('hello')")
            ...     print(result.output)
        """
        # Merge init tools with execution tools (execution tools override)
        merged_tools = self._merge_tools(tools)

        # Execute action asynchronously with merged tools
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

        # Merge: init tools + execution tools (execution overrides)
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

    def __repr__(self) -> str:
        env_type = type(self._environment).__name__
        tool_count = len(self._tools)
        return f"Environment(environment={env_type}, tools={tool_count})"
