"""Environment module for code execution in isolated environments."""

from typing import Any, Callable, Dict, Optional

from msgflux.environments.code.base import BasePythonEnvironment
from msgflux.environments.code.response import ExecutionResult
from msgflux.nn.modules.module import Module


class Environment(Module):
    r"""Environment for secure code execution with optional tool injection.

    This module wraps a code environment (e.g., DenoPyodideSandbox) and provides
    a consistent interface for executing code with dynamically injected tools.

    The Environment follows the pattern used in DSPy's PythonInterpreter,
    allowing code to call host-side functions (tools) during execution.

    Example:
        >>> from msgflux.nn import Environment
        >>> from msgflux.environments import Environments
        >>>
        >>> # Create environment with a code environment
        >>> env = Environment(environment=Environments.code("python"))
        >>>
        >>> # Execute code
        >>> result = env("x = 1 + 2\nprint(x)")
        >>> print(result.output)  # "3"
        >>>
        >>> # Execute with tools
        >>> def search(query: str) -> str:
        ...     return f"Results for: {query}"
        >>>
        >>> result = env(
        ...     "data = search('python')\nprint(data)",
        ...     tools={"search": search}
        ... )

    For RL and agent workflows:
        >>> # Tools can be passed dynamically from ToolLibrary
        >>> tool_funcs = {t.name: t.callable for t in tool_library}
        >>> result = await env.acall(action, tools=tool_funcs)
    """

    def __init__(
        self,
        environment: Optional[BasePythonEnvironment] = None,
    ):
        """Initialize the Environment module.

        Args:
            environment:
                A code environment instance for execution (e.g., DenoPyodideSandbox).
                If None, code execution will raise an error.
                The environment should have a `name` attribute that identifies it
                as a tool (e.g., "execute_code", "python_interpreter").

        Example:
            >>> from msgflux.environments import Environments
            >>> env = Environment(environment=Environments.code("python"))
        """
        super().__init__()
        self._environment = environment
        self._registered_tools: Dict[str, Callable] = {}

    @property
    def environment(self) -> Optional[BasePythonEnvironment]:
        """Get the underlying environment instance."""
        return self._environment

    @property
    def name(self) -> Optional[str]:
        """Get the environment name (e.g., 'execute_code')."""
        if self._environment is not None:
            return self._environment.name
        return None

    @property
    def registered_tools(self) -> Dict[str, Callable]:
        """Get the currently registered tools."""
        return self._registered_tools.copy()

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
                These tools will be available to call from within the code.
                Supports both sync and async callables.
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

        Raises:
            RuntimeError:
                If no environment is configured.

        Example:
            >>> result = env("x = add(1, 2)", tools={"add": lambda a, b: a + b})
            >>> print(result.output)  # "3"
        """
        if self._environment is None:
            raise RuntimeError(
                "No environment configured. "
                "Pass an environment instance to Environment.__init__"
            )

        # Register tools if provided
        if tools:
            self._register_tools(tools)

        # Execute action
        return self._environment(action, vars=vars)

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
            vars:
                Optional dictionary of variables to inject.

        Returns:
            ExecutionResult with execution details.

        Raises:
            RuntimeError:
                If no environment is configured.

        Example:
            >>> async def main():
            ...     result = await env.acall("print('hello')")
            ...     print(result.output)
        """
        if self._environment is None:
            raise RuntimeError(
                "No environment configured. "
                "Pass an environment instance to Environment.__init__"
            )

        # Register tools if provided
        if tools:
            self._register_tools(tools)

        # Execute action asynchronously
        return await self._environment.acall(action, vars=vars)

    def _register_tools(self, tools: Dict[str, Callable[..., Any]]) -> None:
        """Register tools in the environment.

        Only registers tools that haven't been registered yet to avoid
        duplicate registration overhead.

        Args:
            tools:
                Dictionary mapping tool names to callables.
        """
        for name, func in tools.items():
            if name not in self._registered_tools:
                self._environment.register_tool(name, func)
                self._registered_tools[name] = func

    def reset(self) -> None:
        """Reset the environment state.

        Clears all registered tools and resets the environment state.
        """
        if self._environment is not None:
            self._environment.reset()
        self._registered_tools.clear()

    def shutdown(self) -> None:
        """Shutdown the environment and release resources."""
        if self._environment is not None:
            self._environment.shutdown()
        self._registered_tools.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_):
        """Context manager exit with cleanup."""
        self.shutdown()

    def __repr__(self) -> str:
        env_type = type(self._environment).__name__ if self._environment else "None"
        tool_count = len(self._registered_tools)
        return f"Environment(environment={env_type}, tools={tool_count})"
