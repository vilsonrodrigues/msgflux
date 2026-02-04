"""Secure Python environment using Deno + Pyodide (WebAssembly)."""

import asyncio
import base64
import concurrent.futures
import inspect
import json
import os
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from msgflux.environments.code.base import BasePythonEnvironment
from msgflux.environments.code.registry import register_environment
from msgflux.environments.code.response import ExecutionResult
from msgflux.environments.exceptions import (
    SandboxConnectionError,
    SandboxNotReadyError,
    SandboxSecurityError,
    SandboxTimeoutError,
    VariableSizeLimitError,
)
from msgflux.envs import envs
from msgflux.logger import logger

# Maximum variable size for Pyodide FFI (100MB)
MAX_VARIABLE_SIZE_BYTES = 100 * 1024 * 1024


@register_environment
class DenoPyodideSandbox(BasePythonEnvironment):
    """Secure Python environment using Deno + Pyodide (WebAssembly).

    This provides high security by running Python code in a WebAssembly
    sandbox within Deno's permission-restricted runtime.

    Security features:
    - WebAssembly memory isolation
    - Deno permission restrictions (no network, no filesystem by default)
    - Timeout enforcement
    - Thread ownership verification
    - Variable size limits (100MB)

    Requires:
    - Deno installed and available in PATH

    Example:
        >>> from msgflux.environments import Environments
        >>> env = Environments.code("python/deno_pyodide")
        >>> result = env("print('Hello!')")
        >>> print(result.output)  # "Hello!"
    """

    provider = "deno_pyodide"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        allow_network: bool = False,
        allow_read: Optional[List[str]] = None,
        allow_write: Optional[List[str]] = None,
        tools: Optional[Dict[str, Any]] = None,
        packages: Optional[List[str]] = None,
    ):
        """Initialize Deno+Pyodide environment.

        Args:
            timeout:
                Default execution timeout in seconds.
            allow_network:
                Allow network access (default: False).
            allow_read:
                List of paths to allow reading (default: None).
            allow_write:
                List of paths to allow writing (default: None).
            tools:
                Dictionary of tool name to callable. Tools are Python functions
                that can be called from inside the environment.
            packages:
                List of Python packages to install via micropip on initialization.
                Packages must be pure Python or have WebAssembly wheels available.

        Raises:
            SandboxConnectionError:
                If Deno is not installed or package installation fails.

        Example:
            >>> env = Environments.code(
            ...     "python",
            ...     packages=["matplotlib", "numpy"],
            ...     tools={"search": search_fn}
            ... )
            >>> env("import matplotlib; print(matplotlib.__version__)")
        """
        self.timeout = timeout
        self.allow_network = allow_network
        self.allow_read = allow_read or []
        self.allow_write = allow_write or []
        self._tools: Dict[str, Any] = tools or {}
        self._packages: List[str] = packages or []

        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._owner_thread: Optional[int] = None
        self._lock = threading.RLock()  # RLock allows reentrant locking
        self._initialized = False
        self._variables: Dict[str, Any] = {}
        self._tools_registered = False
        self._allow_cross_thread = False  # Used for async execution

        self._initialize()

        # Install packages after initialization
        if self._packages:
            self._install_packages()

        # Register tools after initialization
        if self._tools:
            self._register_tools_in_sandbox()

    def _check_deno_installed(self) -> bool:
        """Check if Deno is available in PATH."""
        return shutil.which("deno") is not None

    def _get_runner_path(self) -> Path:
        """Get path to the Pyodide runner script."""
        return Path(__file__).parent / "scripts" / "pyodide_runner.js"

    def _get_deno_cache_dir(self) -> Optional[str]:
        """Get Deno cache directory."""
        # Check DENO_DIR env var first
        deno_dir = os.environ.get("DENO_DIR")
        if deno_dir:
            return deno_dir

        # Default locations
        home = os.environ.get("HOME", os.path.expanduser("~"))
        cache_dir = os.path.join(home, ".cache", "deno")
        if os.path.exists(cache_dir):
            return cache_dir

        # macOS location
        cache_dir = os.path.join(home, "Library", "Caches", "deno")
        if os.path.exists(cache_dir):
            return cache_dir

        return None

    def _build_deno_command(self) -> List[str]:
        """Build the Deno command with appropriate permissions."""
        runner_path = self._get_runner_path()

        cmd = ["deno", "run", "--allow-env"]

        # Always allow reading the runner script and Deno cache
        read_paths = [str(runner_path)]

        # Add Deno cache directory (required for Pyodide WASM files)
        deno_cache = self._get_deno_cache_dir()
        if deno_cache:
            read_paths.append(deno_cache)

        # Add user-specified read paths
        if self.allow_read:
            read_paths.extend(self.allow_read)

        cmd.append(f"--allow-read={','.join(read_paths)}")

        # Write permissions
        write_paths = list(self.allow_write) if self.allow_write else []
        # Add Deno cache write permission if enabled via env var
        if envs.deno_allow_cache_write and deno_cache:
            write_paths.append(deno_cache)
        if write_paths:
            cmd.append(f"--allow-write={','.join(write_paths)}")

        # Network permissions - always allow Pyodide CDN for package downloads
        # User can expand with allow_network=True for full access
        pyodide_hosts = ["cdn.jsdelivr.net", "pypi.org", "files.pythonhosted.org"]
        if self.allow_network:
            cmd.append("--allow-net")
        else:
            cmd.append(f"--allow-net={','.join(pyodide_hosts)}")

        cmd.append(str(runner_path))
        return cmd

    def _initialize(self):
        """Start the Deno subprocess."""
        if not self._check_deno_installed():
            raise SandboxConnectionError(
                "Deno is not installed. Install from https://deno.land "
                "or run: curl -fsSL https://deno.land/install.sh | sh"
            )

        runner_path = self._get_runner_path()
        if not runner_path.exists():
            raise SandboxConnectionError(
                f"Pyodide runner script not found: {runner_path}"
            )

        cmd = self._build_deno_command()
        logger.debug(f"Starting Deno environment: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._owner_thread = threading.current_thread().ident

            # Wait for initialization response
            self._wait_for_init()
            self._initialized = True
            logger.info("Deno+Pyodide environment initialized successfully")

        except Exception as e:
            if self._process:
                self._process.kill()
                self._process = None
            raise SandboxConnectionError(f"Failed to start environment: {e}") from e

    def _wait_for_init(self):
        """Wait for environment initialization."""
        start_time = time.time()
        timeout_seconds = 120.0  # Pyodide download can take a while

        while True:
            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxTimeoutError(
                    timeout_seconds,
                    f"Environment initialization timed out. Stderr: {stderr}",
                )

            wait_time = min(remaining, 5.0)
            ready, _, _ = select.select([self._process.stdout], [], [], wait_time)
            if not ready:
                continue  # Keep waiting

            line = self._process.stdout.readline().strip()
            if not line:
                continue  # Empty line, keep reading

            # Skip non-JSON lines (Pyodide loading messages)
            if not line.startswith("{"):
                logger.debug(f"Pyodide init message: {line}")
                continue

            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Non-JSON line during init: {line}")
                continue

            if "error" in response:
                raise SandboxConnectionError(f"Init error: {response['error']}")

            result = response.get("result", {})
            if result.get("status") == "ready":
                logger.debug(f"Pyodide version: {result.get('version', 'unknown')}")
                return

            # If we got JSON but not "ready", log and continue
            logger.debug(f"Unexpected init response: {response}")

    def _install_packages(self):
        """Install packages specified at initialization."""
        for package in self._packages:
            logger.debug(f"Installing package: {package}")
            success = self.install_package(package)
            if not success:
                raise SandboxConnectionError(
                    f"Failed to install package: {package}. "
                    "Ensure the package is available for Pyodide (pure Python "
                    "or has WebAssembly wheels)."
                )
            logger.info(f"Installed package: {package}")

    def _register_tools_in_sandbox(self):
        """Register all tools in the environment."""
        for name, func in self._tools.items():
            # Get function signature to build parameters
            parameters = []
            try:
                sig = inspect.signature(func)
                for param_name, param in sig.parameters.items():
                    param_info = {"name": param_name}

                    # Get type annotation if available
                    if param.annotation != inspect.Parameter.empty:
                        annotation = param.annotation
                        type_name = getattr(annotation, "__name__", str(annotation))
                        param_info["type"] = type_name

                    # Get default value if available
                    if param.default != inspect.Parameter.empty:
                        param_info["default"] = param.default

                    parameters.append(param_info)
            except (ValueError, TypeError):
                pass  # No signature available

            # Register tool in environment
            response = self._send_request(
                "register_tool",
                {"name": name, "parameters": parameters},
                timeout=10.0,
            )

            if "error" in response:
                logger.warning(f"Failed to register tool {name}: {response['error']}")
            else:
                logger.debug(f"Registered tool: {name}")

        self._tools_registered = True

    def _handle_tool_call(self, tool_name: str, args: list, kwargs: dict) -> Any:
        """Execute a tool call from the environment.

        Supports both sync and async tools. Async tools are executed
        using asyncio.run() or via a thread pool if a loop is already running.
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        func = self._tools[tool_name]

        try:
            # Check if the function is a coroutine function
            if asyncio.iscoroutinefunction(func):
                # Try to get the running loop, otherwise create a new one
                try:
                    asyncio.get_running_loop()
                    # If we're in an async context, run in a thread pool
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, func(*args, **kwargs))
                        result = future.result()
                except RuntimeError:
                    # No running loop, safe to use asyncio.run
                    result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)

            # Ensure result is JSON-serializable
            try:
                json.dumps(result)
            except (TypeError, ValueError):
                result = str(result)

            return result

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    def _check_thread_ownership(self):
        """Verify we're on the owner thread."""
        # Allow cross-thread access when explicitly enabled (e.g., for async execution)
        if self._allow_cross_thread:
            return

        current_thread = threading.current_thread().ident
        if self._owner_thread != current_thread:
            raise SandboxSecurityError(
                f"Environment accessed from wrong thread. "
                f"Owner: {self._owner_thread}, Current: {current_thread}. "
                "Create a new environment instance for each thread."
            )

    def _send_tool_response(
        self,
        tool_call_id: str,
        result: Any = None,
        error: Optional[str] = None,
    ):
        """Send a tool call response back to the environment."""
        if error:
            response = {
                "jsonrpc": "2.0",
                "id": tool_call_id,
                "error": {"code": -32000, "message": error},
            }
        else:
            # Format result with type information
            # JSON-serializable types (dict, list, int, float, bool, None) go as json
            # Strings go as string type
            if isinstance(result, str):
                formatted_result = {"value": result, "type": "string"}
            elif result is None or isinstance(result, (dict, list, int, float, bool)):
                formatted_result = {"value": json.dumps(result), "type": "json"}
            else:
                # Fallback for other types - convert to string
                formatted_result = {"value": str(result), "type": "string"}

            response = {
                "jsonrpc": "2.0",
                "id": tool_call_id,
                "result": formatted_result,
            }

        response_str = json.dumps(response) + "\n"
        self._process.stdin.write(response_str)
        self._process.stdin.flush()

    def _send_request(  # noqa: C901
        self,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to Deno process.

        This method handles bidirectional communication:
        - Sends requests to the environment
        - Handles tool calls from the environment during execution
        - Returns the final response
        """
        if not self._initialized or not self._process:
            raise SandboxNotReadyError("Environment not initialized")

        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise SandboxConnectionError(f"Environment process died. Stderr: {stderr}")

        self._check_thread_ownership()
        timeout = timeout or self.timeout

        with self._lock:
            self._request_id += 1
            expected_id = self._request_id
            request = {
                "jsonrpc": "2.0",
                "id": expected_id,
                "method": method,
                "params": params,
            }

            try:
                request_str = json.dumps(request) + "\n"
                self._process.stdin.write(request_str)
                self._process.stdin.flush()

                start_time = time.time()

                # Read responses, handling tool calls until we get our response
                while True:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise SandboxTimeoutError(timeout)

                    ready, _, _ = select.select(
                        [self._process.stdout], [], [], min(remaining, 1.0)
                    )

                    if not ready:
                        continue

                    response_line = self._process.stdout.readline().strip()
                    if not response_line:
                        continue

                    # Skip non-JSON lines (Pyodide loading messages)
                    if not response_line.startswith("{"):
                        logger.debug(f"Pyodide message: {response_line}")
                        continue

                    response = json.loads(response_line)

                    # Check if this is a tool call request from the environment
                    if response.get("method") == "tool_call":
                        tool_params = response.get("params", {})
                        tool_call_id = response.get("id")
                        tool_name = tool_params.get("tool_name")
                        tool_args = tool_params.get("args", [])
                        tool_kwargs = tool_params.get("kwargs", {})

                        logger.debug(f"Tool call: {tool_name}(args={tool_args})")

                        try:
                            result = self._handle_tool_call(
                                tool_name, tool_args, tool_kwargs
                            )
                            self._send_tool_response(tool_call_id, result=result)
                        except Exception as e:
                            self._send_tool_response(tool_call_id, error=str(e))

                        continue

                    # Check if this is our response
                    if response.get("id") == expected_id:
                        if "error" in response:
                            return {"error": response["error"]}
                        return response.get("result", {})

                    # Log unexpected responses
                    logger.warning(f"Unexpected response: {response}")

            except json.JSONDecodeError as e:
                raise SandboxConnectionError(f"Invalid response JSON: {e}") from e
            except BrokenPipeError as e:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxConnectionError(
                    f"Environment process pipe broken. Stderr: {stderr}"
                ) from e

    def _check_variable_size(self, name: str, value: Any):
        """Check if variable exceeds size limit."""
        try:
            serialized = json.dumps(value)
            size = len(serialized.encode("utf-8"))
            if size > MAX_VARIABLE_SIZE_BYTES:
                raise VariableSizeLimitError(
                    name,
                    size / (1024 * 1024),
                    MAX_VARIABLE_SIZE_BYTES / (1024 * 1024),
                )
        except (TypeError, ValueError):
            pass  # Non-serializable, will be handled by environment

    def __call__(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> ExecutionResult:
        """Execute Python code in the environment.

        Args:
            action:
                Python code to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Variables to inject before execution.
            tools:
                Tools to make available for this execution.
                New tools are registered; existing tools are updated.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        start_time = time.time()

        # Register tools if provided
        if tools:
            for name, func in tools.items():
                if name not in self._tools:
                    self.register_tool(name, func)
                else:
                    # Update existing tool function
                    self._tools[name] = func

        # Inject variables if provided
        if vars:
            for name, value in vars.items():
                self._check_variable_size(name, value)
                self.set_variable(name, value)

        response = self._send_request("execute", {"code": action}, timeout=timeout)

        if "error" in response:
            error_info = response["error"]
            if isinstance(error_info, dict):
                # Error type is in data.type (JSON-RPC format from pyodide_runner.js)
                error_data = error_info.get("data", {})
                if isinstance(error_data, dict):
                    error_type = error_data.get("type", "Error")
                else:
                    error_type = "Error"
                error_message = error_info.get("message", "")
                error_msg = f"{error_type}: {error_message}"
            else:
                error_msg = str(error_info)

            return ExecutionResult(
                success=False,
                error=error_msg,
                output=response.get("output", ""),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Update local variable cache
        if response.get("variables"):
            self._variables.update(response["variables"])

        return ExecutionResult(
            success=response.get("success", True),
            output=response.get("output", ""),
            error=response.get("error"),
            return_value=response.get("return_value"),
            variables=response.get("variables", {}),
            execution_time_ms=response.get("execution_time_ms"),
        )

    async def acall(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> ExecutionResult:
        """Execute Python code in the environment asynchronously.

        This method runs the synchronous execution in a thread pool
        to avoid blocking the event loop. Thread safety is ensured via
        internal locking.

        Args:
            action:
                Python code to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Variables to inject before execution.
            tools:
                Tools to make available for this execution.

        Returns:
            ExecutionResult with output, errors, and variables.

        Example:
            >>> async def main():
            ...     env = Environments.code("python/deno_pyodide")
            ...     result = await env.acall("print('hello')")
            ...     print(result.output)
        """

        def _execute():
            # Temporarily allow cross-thread access for async execution
            # The lock ensures thread safety
            with self._lock:
                original_allow = self._allow_cross_thread
                self._allow_cross_thread = True
                try:
                    return self(action, timeout=timeout, vars=vars, tools=tools)
                finally:
                    self._allow_cross_thread = original_allow

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _execute)

    def mount_file(self, path: str, content: Union[str, bytes]) -> None:
        """Mount a file in the environment virtual filesystem.

        Args:
            path:
                Virtual path for the file in the environment.
            content:
                File content as string or bytes.
        """
        if isinstance(content, str):
            content = content.encode("utf-8")

        encoded = base64.b64encode(content).decode("ascii")
        response = self._send_request("mount_file", {"path": path, "content": encoded})

        if "error" in response:
            raise SandboxConnectionError(f"Failed to mount file: {response['error']}")

    def get_variable(self, name: str) -> Any:
        """Get a variable from the environment.

        Args:
            name:
                Variable name.

        Returns:
            Variable value or None if not found.
        """
        response = self._send_request("get_variable", {"name": name})
        return response.get("value")

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the environment.

        Args:
            name:
                Variable name.
            value:
                Variable value (must be JSON-serializable).
        """
        self._check_variable_size(name, value)
        response = self._send_request("set_variable", {"name": name, "value": value})

        if "error" in response:
            raise SandboxConnectionError(
                f"Failed to set variable: {response['error']}"
            )

        self._variables[name] = value

    def reset(self) -> None:
        """Reset environment state."""
        response = self._send_request("reset", {})

        if "error" in response:
            raise SandboxConnectionError(f"Failed to reset: {response['error']}")

        self._variables.clear()

    def shutdown(self) -> None:
        """Shutdown the environment."""
        if self._process:
            try:
                self._send_request("shutdown", {}, timeout=5.0)
            except Exception as e:
                logger.debug(f"Shutdown request failed (expected): {e}")

            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            except Exception as e:
                logger.debug(f"Process termination error: {e}")

            self._process = None
            self._initialized = False
            logger.debug("Deno environment shutdown complete")

    def install_package(self, package: str) -> bool:
        """Install a Python package via micropip.

        Args:
            package:
                Package name to install.

        Returns:
            True if installation succeeded.
        """
        response = self._send_request(
            "install_package", {"package": package}, timeout=120.0
        )
        return response.get("success", False)

    def list_packages(self) -> List[str]:
        """List installed packages.

        Returns:
            List of installed package names.
        """
        response = self._send_request("list_packages", {})
        return response.get("packages", [])

    def register_tool(self, name: str, func: Callable[..., Any]) -> None:
        """Register a tool that can be called from inside the environment.

        Args:
            name:
                The name to use for the tool inside the environment.
            func:
                The Python function to register.

        Example:
            >>> def my_search(query: str) -> str:
            ...     return f"Results for: {query}"
            >>> env.register_tool("search", my_search)
            >>> env("result = search('python')")
        """
        # Store in local tools dict
        self._tools[name] = func

        # Get function signature for the environment
        parameters = []
        try:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                param_info = {"name": param_name}
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                    type_name = getattr(annotation, "__name__", str(annotation))
                    param_info["type"] = type_name
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                parameters.append(param_info)
        except (ValueError, TypeError):
            pass

        # Register in environment
        response = self._send_request(
            "register_tool",
            {"name": name, "parameters": parameters},
            timeout=10.0,
        )

        if "error" in response:
            error_msg = response["error"]
            raise SandboxConnectionError(f"Failed to register tool: {error_msg}")

        logger.debug(f"Registered tool: {name}")

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:  # noqa: S110
            pass  # Ignore errors during garbage collection
