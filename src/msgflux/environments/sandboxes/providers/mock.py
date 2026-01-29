"""Mock sandbox for testing without external dependencies."""

import asyncio
import concurrent.futures
import io
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Union

from msgflux.environments.sandboxes.base import BasePythonSandbox
from msgflux.environments.sandboxes.registry import register_sandbox
from msgflux.environments.sandboxes.response import ExecutionResult


@register_sandbox
class MockSandbox(BasePythonSandbox):
    """Mock sandbox for testing without external dependencies.

    This sandbox executes code using Python's exec() with restricted
    builtins. It is NOT secure for untrusted code and should only be
    used for testing purposes.
    """

    provider = "mock"

    def __init__(
        self,
        *,
        default_output: Optional[str] = None,
        simulate_errors: bool = False,
        execute_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        allowed_builtins: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        """Initialize mock sandbox.

        Args:
            default_output:
                Default output to return when no output is captured.
            simulate_errors:
                If True, always return errors for testing error handling.
            execute_fn:
                Optional custom function to execute code.
            allowed_builtins:
                Dictionary of allowed builtins. If None, uses safe defaults.
            tools:
                Dictionary of tool name to callable. Tools are Python functions
                that can be called from code running in the sandbox.

        Example:
            >>> def search(query: str) -> str:
            ...     return f"Results for {query}"
            >>> sandbox = MockSandbox(tools={"search": search})
            >>> sandbox("result = search('python')")
        """
        self.default_output = default_output
        self.simulate_errors = simulate_errors
        self.execute_fn = execute_fn
        self.allowed_builtins = allowed_builtins
        self._tools: Dict[str, Callable[..., Any]] = tools or {}
        self._variables: Dict[str, Any] = {}
        self._files: Dict[str, Union[str, bytes]] = {}
        self._call_history: List[tuple] = []
        self._initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize the mock sandbox."""
        self._initialized = True
        if self.allowed_builtins is None:
            self.allowed_builtins = self._get_safe_builtins()

    def _make_sync_wrapper(self, async_func: Callable) -> Callable:
        """Create a sync wrapper for an async function."""

        def wrapper(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, async_func(*args, **kwargs))
                    return future.result()
            except RuntimeError:
                return asyncio.run(async_func(*args, **kwargs))

        return wrapper

    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get a dictionary of safe builtins for mock execution."""
        return {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "True": True,
            "False": False,
            "None": None,
        }

    def __call__(  # noqa: C901
        self,
        code: str,
        *,
        timeout: Optional[float] = None,  # noqa: ARG002
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute code in the mock sandbox.

        Args:
            code:
                Python code to execute.
            timeout:
                Ignored in mock implementation.
            vars:
                Variables to inject before execution.

        Returns:
            ExecutionResult with output and captured variables.
        """
        start_time = time.time()

        if vars:
            self._variables.update(vars)

        self._call_history.append((code, dict(vars) if vars else {}))

        if self.simulate_errors:
            return ExecutionResult(
                success=False,
                error="Simulated error for testing",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        if self.execute_fn is not None:
            try:
                result = self.execute_fn(code, self._variables)
                return ExecutionResult(
                    success=True,
                    output=str(result) if result is not None else None,
                    return_value=result,
                    variables=dict(self._variables),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        try:
            local_vars = dict(self._variables)
            global_vars = {"__builtins__": self.allowed_builtins}

            # Inject tools into global namespace (with async support)
            for tool_name, tool_func in self._tools.items():
                if asyncio.iscoroutinefunction(tool_func):
                    # Wrap async function to make it callable synchronously
                    global_vars[tool_name] = self._make_sync_wrapper(tool_func)
                else:
                    global_vars[tool_name] = tool_func

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, global_vars, local_vars)  # noqa: S102

            for key, value in local_vars.items():
                if not key.startswith("_"):
                    self._variables[key] = value

            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            output = stdout_output
            if stderr_output:
                output = f"{output}\n{stderr_output}" if output else stderr_output

            return ExecutionResult(
                success=True,
                output=output or self.default_output,
                variables={
                    k: v for k, v in self._vars.items() if not k.startswith("_")
                },
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=f"SyntaxError: {e.msg} (line {e.lineno})",
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def acall(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute code in the mock sandbox asynchronously.

        Args:
            code:
                Python code to execute.
            timeout:
                Ignored in mock implementation.
            vars:
                Variables to inject before execution.

        Returns:
            ExecutionResult with output and captured variables.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self(code, timeout=timeout, vars=vars),
        )

    def mount_file(self, path: str, content: Union[str, bytes]) -> None:
        """Mount a virtual file in the mock sandbox.

        Args:
            path:
                Virtual path for the file.
            content:
                File content.
        """
        self._files[path] = content

    def get_variable(self, name: str) -> Any:
        """Get a variable from the sandbox.

        Args:
            name:
                Variable name.

        Returns:
            Variable value or None if not found.
        """
        return self._variables.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox.

        Args:
            name:
                Variable name.
            value:
                Variable value.
        """
        self._variables[name] = value

    def reset(self) -> None:
        """Reset sandbox state."""
        self._variables.clear()
        self._files.clear()
        self._call_history.clear()

    def shutdown(self) -> None:
        """Shutdown the sandbox."""
        self._initialized = False

    def install_package(self, package: str) -> bool:  # noqa: ARG002
        """Mock package installation (always succeeds).

        Args:
            package:
                Package name.

        Returns:
            Always True in mock implementation.
        """
        return True

    def list_packages(self) -> List[str]:
        """List mock packages.

        Returns:
            Empty list in mock implementation.
        """
        return []

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool that can be called from inside the sandbox.

        Args:
            name:
                The name to use for the tool inside the sandbox.
            func:
                The Python function to register.

        Example:
            >>> def my_search(query: str) -> str:
            ...     return f"Results for: {query}"
            >>> sandbox.register_tool("search", my_search)
            >>> sandbox("result = search('python')")
        """
        self._tools[name] = func

    @property
    def call_history(self) -> List[tuple]:
        """Get execution history for testing.

        Returns:
            List of (code, variables) tuples.
        """
        return list(self._call_history)

    @property
    def call_count(self) -> int:
        """Get number of executions.

        Returns:
            Number of times execute was called.
        """
        return len(self._call_history)
