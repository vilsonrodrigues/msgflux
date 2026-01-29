"""Secure Python sandbox using Deno + Pyodide (WebAssembly)."""

import base64
import json
import select
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from msgflux.environments.exceptions import (
    SandboxConnectionError,
    SandboxNotReadyError,
    SandboxSecurityError,
    SandboxTimeoutError,
    VariableSizeLimitError,
)
from msgflux.environments.sandboxes.base import BasePythonSandbox
from msgflux.environments.sandboxes.registry import register_sandbox
from msgflux.environments.sandboxes.response import ExecutionResult
from msgflux.logger import logger

# Maximum variable size for Pyodide FFI (100MB)
MAX_VARIABLE_SIZE_BYTES = 100 * 1024 * 1024


@register_sandbox
class DenoPyodideSandbox(BasePythonSandbox):
    """Secure Python sandbox using Deno + Pyodide (WebAssembly).

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
    """

    provider = "deno_pyodide"

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        allow_network: bool = False,
        allow_read: Optional[List[str]] = None,
        allow_write: Optional[List[str]] = None,
    ):
        """Initialize Deno+Pyodide sandbox.

        Args:
            timeout:
                Default execution timeout in seconds.
            allow_network:
                Allow network access (default: False).
            allow_read:
                List of paths to allow reading (default: None).
            allow_write:
                List of paths to allow writing (default: None).

        Raises:
            SandboxConnectionError:
                If Deno is not installed.
        """
        self.timeout = timeout
        self.allow_network = allow_network
        self.allow_read = allow_read or []
        self.allow_write = allow_write or []

        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._owner_thread: Optional[int] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._variables: Dict[str, Any] = {}

        self._initialize()

    def _check_deno_installed(self) -> bool:
        """Check if Deno is available in PATH."""
        return shutil.which("deno") is not None

    def _get_runner_path(self) -> Path:
        """Get path to the Pyodide runner script."""
        return Path(__file__).parent / "scripts" / "pyodide_runner.js"

    def _build_deno_command(self) -> List[str]:
        """Build the Deno command with appropriate permissions."""
        runner_path = self._get_runner_path()

        cmd = ["deno", "run", "--allow-env"]

        # Always allow reading the runner script and Deno cache
        read_paths = [str(runner_path)]

        # Add user-specified read paths
        if self.allow_read:
            read_paths.extend(self.allow_read)

        cmd.append(f"--allow-read={','.join(read_paths)}")

        # Write permissions
        if self.allow_write:
            cmd.append(f"--allow-write={','.join(self.allow_write)}")

        # Network permissions
        if self.allow_network:
            cmd.append("--allow-net")

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
        logger.debug(f"Starting Deno sandbox: {' '.join(cmd)}")

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
            logger.info("Deno+Pyodide sandbox initialized successfully")

        except Exception as e:
            if self._process:
                self._process.kill()
                self._process = None
            raise SandboxConnectionError(f"Failed to start sandbox: {e}") from e

    def _wait_for_init(self):
        """Wait for sandbox initialization."""
        try:
            # Read the init response (with timeout)
            ready, _, _ = select.select([self._process.stdout], [], [], 60.0)
            if not ready:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxTimeoutError(
                    60.0, f"Sandbox initialization timed out. Stderr: {stderr}"
                )

            line = self._process.stdout.readline().strip()
            if not line:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxConnectionError(f"Empty init response. Stderr: {stderr}")

            response = json.loads(line)
            if "error" in response:
                raise SandboxConnectionError(f"Init error: {response['error']}")

            result = response.get("result", {})
            if result.get("status") != "ready":
                raise SandboxConnectionError(f"Unexpected init response: {response}")

            logger.debug(f"Pyodide version: {result.get('version', 'unknown')}")

        except json.JSONDecodeError as e:
            raise SandboxConnectionError(f"Invalid init response: {e}") from e

    def _check_thread_ownership(self):
        """Verify we're on the owner thread."""
        current_thread = threading.current_thread().ident
        if self._owner_thread != current_thread:
            raise SandboxSecurityError(
                f"Sandbox accessed from wrong thread. "
                f"Owner: {self._owner_thread}, Current: {current_thread}. "
                "Create a new sandbox instance for each thread."
            )

    def _send_request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to Deno process."""
        if not self._initialized or not self._process:
            raise SandboxNotReadyError("Sandbox not initialized")

        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise SandboxConnectionError(f"Sandbox process died. Stderr: {stderr}")

        self._check_thread_ownership()
        timeout = timeout or self.timeout

        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

            try:
                request_str = json.dumps(request) + "\n"
                self._process.stdin.write(request_str)
                self._process.stdin.flush()

                # Read response with timeout
                ready, _, _ = select.select([self._process.stdout], [], [], timeout)

                if not ready:
                    raise SandboxTimeoutError(timeout)

                response_line = self._process.stdout.readline().strip()
                if not response_line:
                    stderr = self._process.stderr.read() if self._process.stderr else ""
                    raise SandboxConnectionError(f"Empty response. Stderr: {stderr}")

                response = json.loads(response_line)

                # Verify response ID matches
                if response.get("id") != self._request_id:
                    logger.warning(
                        f"Response ID mismatch: expected {self._request_id}, "
                        f"got {response.get('id')}"
                    )

                if "error" in response:
                    return {"error": response["error"]}

                return response.get("result", {})

            except json.JSONDecodeError as e:
                raise SandboxConnectionError(f"Invalid response JSON: {e}") from e
            except BrokenPipeError as e:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise SandboxConnectionError(
                    f"Sandbox process pipe broken. Stderr: {stderr}"
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
            pass  # Non-serializable, will be handled by sandbox

    def __call__(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """Execute Python code in the sandbox.

        Args:
            code:
                Python code to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            variables:
                Variables to inject before execution.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        start_time = time.time()

        # Inject variables if provided
        if variables:
            for name, value in variables.items():
                self._check_variable_size(name, value)
                self.set_variable(name, value)

        response = self._send_request("execute", {"code": code}, timeout=timeout)

        if "error" in response:
            error_info = response["error"]
            if isinstance(error_info, dict):
                error_type = error_info.get("type", "Error")
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

    def mount_file(self, path: str, content: Union[str, bytes]) -> None:
        """Mount a file in the sandbox virtual filesystem.

        Args:
            path:
                Virtual path for the file in the sandbox.
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
        """Get a variable from the sandbox.

        Args:
            name:
                Variable name.

        Returns:
            Variable value or None if not found.
        """
        response = self._send_request("get_variable", {"name": name})
        return response.get("value")

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the sandbox.

        Args:
            name:
                Variable name.
            value:
                Variable value (must be JSON-serializable).
        """
        self._check_variable_size(name, value)
        response = self._send_request("set_variable", {"name": name, "value": value})

        if "error" in response:
            raise SandboxConnectionError(f"Failed to set variable: {response['error']}")

        self._variables[name] = value

    def reset(self) -> None:
        """Reset sandbox state."""
        response = self._send_request("reset", {})

        if "error" in response:
            raise SandboxConnectionError(f"Failed to reset: {response['error']}")

        self._variables.clear()

    def shutdown(self) -> None:
        """Shutdown the sandbox."""
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
            logger.debug("Deno sandbox shutdown complete")

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

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:  # noqa: S110
            pass  # Ignore errors during garbage collection
