"""Agent Sandbox providers for Python and Shell execution via Docker.

This module provides two environment providers for the agent-infra/sandbox:
- AgentSandboxPython: Python code execution via Jupyter kernel
- AgentSandboxShell: Shell command execution

Requires:
    - Docker running with agent-sandbox container
    - pip install msgflux[agent-sandbox]

Example:
    >>> # Start container first:
    >>> # docker run --rm -p 8080:8080 ghcr.io/agent-infra/sandbox:latest

    >>> from msgflux.environments import Environments

    >>> # Python execution
    >>> env = Environments.code("python/agent_sandbox")
    >>> result = env("x = 1 + 1; print(x)")
    >>> print(result.output)  # "2"

    >>> # Shell execution
    >>> env = Environments.code("shell/agent_sandbox")
    >>> result = env("ls -la /workspace")
    >>> print(result.output)
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Union

from msgflux.environments.code.base import BasePythonEnvironment, BaseShellEnvironment
from msgflux.environments.code.registry import register_environment
from msgflux.environments.code.response import ExecutionResult
from msgflux.environments.exceptions import (
    SandboxConnectionError,
    SandboxNotReadyError,
    SandboxTimeoutError,
)
from msgflux.logger import logger


class AgentSandboxBase:
    """Shared connection logic for agent-sandbox providers.

    This base class handles:
    - Connection to agent-sandbox API
    - Session management
    - File mounting

    Both AgentSandboxPython and AgentSandboxShell inherit from this class
    to share the connection logic.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        session_id: Optional[str] = None,
    ):
        """Initialize connection settings for agent-sandbox.

        Args:
            base_url:
                URL of the agent-sandbox API (default: http://localhost:8080).
            timeout:
                Default execution timeout in seconds.
            session_id:
                Optional session ID for state persistence. If not provided,
                a unique session ID will be generated.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session_id = session_id or f"msgflux-{id(self)}"
        self._client = None
        self._async_client = None
        self._initialized = False
        self._tools: Dict[str, Callable] = {}

    def _initialize(self):
        """Initialize connection to agent-sandbox (sync and async clients)."""
        try:
            from agent_sandbox import AsyncSandbox, Sandbox
        except ImportError as e:
            raise SandboxConnectionError(
                "agent-sandbox not installed. Run: pip install msgflux[agent-sandbox]"
            ) from e

        try:
            self._client = Sandbox(base_url=self.base_url)
            self._async_client = AsyncSandbox(base_url=self.base_url)
            self._initialized = True
            logger.info(f"AgentSandbox connected to {self.base_url}")
        except Exception as e:
            raise SandboxConnectionError(
                f"Failed to connect to agent-sandbox at {self.base_url}: {e}"
            ) from e

    def mount_file(self, path: str, content: Union[str, bytes]) -> None:
        """Write file to sandbox filesystem.

        Args:
            path:
                The path in the sandbox filesystem.
            content:
                The file content as string or bytes.

        Raises:
            SandboxNotReadyError: If the sandbox is not initialized.
        """
        if not self._initialized:
            raise SandboxNotReadyError("AgentSandbox not initialized")

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        self._client.file.write(file=path, content=content)
        logger.debug(f"Mounted file: {path}")

    def reset(self) -> None:
        """Reset session by creating a new session ID.

        This effectively starts a fresh session, discarding any previous
        state from the Jupyter kernel or shell.
        """
        self._session_id = f"msgflux-{id(self)}-{time.time()}"
        logger.debug(f"AgentSandbox session reset: {self._session_id}")

    def shutdown(self) -> None:
        """Cleanup resources.

        Disconnects from the agent-sandbox API.
        """
        self._client = None
        self._async_client = None
        self._initialized = False
        logger.debug("AgentSandbox shutdown complete")


@register_environment
class AgentSandboxPython(AgentSandboxBase, BasePythonEnvironment):
    """Python execution via Jupyter kernel in agent-sandbox.

    This provider executes Python code using the Jupyter kernel in the
    agent-sandbox Docker container. It supports:
    - Variable persistence across executions (via session)
    - Package installation via pip
    - File mounting

    Note:
        Tool injection is NOT supported in agent-sandbox. Use the
        DenoPyodideSandbox provider if you need tool injection.

    Example:
        >>> from msgflux.environments import Environments

        >>> env = Environments.code("python/agent_sandbox")
        >>> result = env("x = 1 + 1; print(x)")
        >>> print(result.output)  # "2"

        >>> # Variables persist in the session
        >>> result = env("print(x * 10)")
        >>> print(result.output)  # "20"

        >>> # With variable injection
        >>> result = env("print(sum(numbers))", vars={"numbers": [1, 2, 3]})
        >>> print(result.output)  # "6"
    """

    provider = "agent_sandbox"
    environment_type = "python"
    name = "execute_code"

    def __init__(self, **kwargs):
        """Initialize Python environment.

        Args:
            **kwargs:
                Arguments passed to AgentSandboxBase:
                - base_url: URL of the agent-sandbox API
                - timeout: Default execution timeout
                - session_id: Optional session ID
        """
        AgentSandboxBase.__init__(self, **kwargs)
        self._variables: Dict[str, Any] = {}
        self._jupyter_session_id: Optional[str] = None
        self._initialize()
        self._create_jupyter_session()

    def _create_jupyter_session(self):
        """Create a Jupyter session for code execution."""
        try:
            result = self._client.jupyter.create_session(
                session_id=self._session_id,
                kernel_name="python3",
                cwd="/workspace",
            )
            if result.success:
                self._jupyter_session_id = result.data.session_id
                logger.debug(f"Jupyter session created: {self._jupyter_session_id}")
            else:
                logger.warning(f"Failed to create Jupyter session: {result.message}")
        except Exception as e:
            logger.warning(f"Could not create Jupyter session: {e}")

    def _process_jupyter_outputs(self, outputs: list) -> tuple:
        """Process Jupyter kernel outputs.

        Returns:
            Tuple of (output_parts, error_output) where error_output is None on success.
        """
        output_parts = []
        for output in outputs:
            if output.output_type == "stream" and output.text:
                output_parts.append(output.text.rstrip("\n"))
            elif output.output_type == "execute_result":
                if output.data and "text/plain" in output.data:
                    output_parts.append(output.data["text/plain"])
            elif output.output_type == "error":
                return output_parts, (output.ename, output.evalue)
        return output_parts, None

    def __call__(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,  # noqa: ARG002
    ) -> ExecutionResult:
        """Execute Python code in the Jupyter kernel.

        Args:
            action:
                Python code to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Variables to inject before execution.
            tools:
                NOT SUPPORTED. Kept for interface compatibility.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        if not self._initialized:
            raise SandboxNotReadyError("AgentSandbox not initialized")

        start_time = time.time()
        timeout = timeout or self.timeout

        if vars:
            self._inject_variables(vars)

        try:
            result = self._client.jupyter.execute_code(
                code=action,
                timeout=int(timeout),
                session_id=self._jupyter_session_id,
            )
            execution_time_ms = (time.time() - start_time) * 1000

            # Process outputs to extract error info (even if success=False)
            output_parts, error_info = self._process_jupyter_outputs(
                result.data.outputs
            )

            if error_info:
                return ExecutionResult(
                    success=False,
                    error=f"{error_info[0]}: {error_info[1]}",
                    output="\n".join(output_parts),
                    execution_time_ms=execution_time_ms,
                )

            if not result.success:
                return ExecutionResult(
                    success=False,
                    error=str(getattr(result, "message", "Execution failed")),
                    output="\n".join(output_parts),
                    execution_time_ms=execution_time_ms,
                )

            new_vars = self._extract_variables(action)
            self._variables.update(new_vars)

            return ExecutionResult(
                success=True,
                output="\n".join(output_parts),
                variables=new_vars,
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError as e:
            raise SandboxTimeoutError(timeout) from e

    async def acall(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,  # noqa: ARG002
    ) -> ExecutionResult:
        """Execute Python code asynchronously using native async client.

        Args:
            action:
                Python code to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Variables to inject before execution.
            tools:
                NOT SUPPORTED. Kept for interface compatibility.

        Returns:
            ExecutionResult with output, errors, and variables.
        """
        if not self._initialized:
            raise SandboxNotReadyError("AgentSandbox not initialized")

        start_time = time.time()
        timeout = timeout or self.timeout

        # Ensure we have a Jupyter session
        if not self._jupyter_session_id:
            await self._acreate_jupyter_session()

        if vars:
            await self._ainject_variables(vars)

        try:
            result = await self._async_client.jupyter.execute_code(
                code=action,
                timeout=int(timeout),
                session_id=self._jupyter_session_id,
            )
            execution_time_ms = (time.time() - start_time) * 1000

            # Process outputs to extract error info (even if success=False)
            output_parts, error_info = self._process_jupyter_outputs(
                result.data.outputs
            )

            if error_info:
                return ExecutionResult(
                    success=False,
                    error=f"{error_info[0]}: {error_info[1]}",
                    output="\n".join(output_parts),
                    execution_time_ms=execution_time_ms,
                )

            if not result.success:
                return ExecutionResult(
                    success=False,
                    error=str(getattr(result, "message", "Execution failed")),
                    output="\n".join(output_parts),
                    execution_time_ms=execution_time_ms,
                )

            new_vars = await self._aextract_variables(action)
            self._variables.update(new_vars)

            return ExecutionResult(
                success=True,
                output="\n".join(output_parts),
                variables=new_vars,
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError as e:
            raise SandboxTimeoutError(timeout) from e

    async def _acreate_jupyter_session(self):
        """Create a Jupyter session asynchronously."""
        try:
            result = await self._async_client.jupyter.create_session(
                session_id=self._session_id,
                kernel_name="python3",
                cwd="/workspace",
            )
            if result.success:
                self._jupyter_session_id = result.data.session_id
                logger.debug(
                    f"Jupyter session created (async): {self._jupyter_session_id}"
                )
            else:
                logger.warning(f"Failed to create Jupyter session: {result.message}")
        except Exception as e:
            logger.warning(f"Could not create Jupyter session: {e}")

    async def _ainject_variables(self, vars: Dict[str, Any]):
        """Inject variables into the Jupyter kernel asynchronously."""
        for name, value in vars.items():
            try:
                serialized = json.dumps(value)
                code = f"import json; {name} = json.loads('''{serialized}''')"
                await self._async_client.jupyter.execute_code(
                    code=code,
                    session_id=self._jupyter_session_id,
                    timeout=5,
                )
                self._variables[name] = value
            except Exception as e:
                logger.warning(f"Could not inject variable {name}: {e}")

    async def _aextract_variables(self, code: str) -> Dict[str, Any]:
        """Extract variables defined in the code asynchronously."""
        pattern = r"^(\w+)\s*="
        var_names = set(re.findall(pattern, code, re.MULTILINE))
        if not var_names:
            return {}

        extracted = {}
        for name in var_names:
            try:
                result = await self._async_client.jupyter.execute_code(
                    code=f"import json; print(json.dumps({name}))",
                    session_id=self._jupyter_session_id,
                    timeout=5,
                )
                if result.success and result.data.outputs:
                    for output in result.data.outputs:
                        if output.output_type == "stream" and output.text:
                            extracted[name] = json.loads(output.text.strip())
                            break
            except Exception:  # noqa: S112
                continue
        return extracted

    def _inject_variables(self, vars: Dict[str, Any]):
        """Inject variables into the Jupyter kernel."""
        for name, value in vars.items():
            try:
                serialized = json.dumps(value)
                code = f"import json; {name} = json.loads('''{serialized}''')"
                self._client.jupyter.execute_code(
                    code=code,
                    session_id=self._jupyter_session_id,
                    timeout=5,
                )
                self._variables[name] = value
            except Exception as e:
                logger.warning(f"Could not inject variable {name}: {e}")

    def _extract_variables(self, code: str) -> Dict[str, Any]:
        """Extract variables defined in the code from the Jupyter kernel."""
        # Find variable assignments in the code
        pattern = r"^(\w+)\s*="
        var_names = set(re.findall(pattern, code, re.MULTILINE))
        if not var_names:
            return {}

        extracted = {}
        for name in var_names:
            try:
                result = self._client.jupyter.execute_code(
                    code=f"import json; print(json.dumps({name}))",
                    session_id=self._jupyter_session_id,
                    timeout=5,
                )
                if result.success and result.data.outputs:
                    for output in result.data.outputs:
                        if output.output_type == "stream" and output.text:
                            extracted[name] = json.loads(output.text.strip())
                            break
            except Exception:  # noqa: S112
                # Variable might not be JSON-serializable or doesn't exist
                continue
        return extracted

    def get_variable(self, name: str) -> Any:
        """Get a variable from the Jupyter kernel.

        Args:
            name:
                Variable name.

        Returns:
            Variable value or None if not found.
        """
        try:
            result = self._client.jupyter.execute_code(
                code=f"import json; print(json.dumps({name}))",
                session_id=self._jupyter_session_id,
                timeout=5,
            )
            if result.success and result.data.outputs:
                for output in result.data.outputs:
                    if output.output_type == "stream" and output.text:
                        return json.loads(output.text.strip())
        except Exception:  # noqa: S110
            pass  # Variable not found or not JSON-serializable
        return self._variables.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the Jupyter kernel.

        Args:
            name:
                Variable name.
            value:
                Variable value (must be JSON-serializable).
        """
        serialized = json.dumps(value)
        self._client.jupyter.execute_code(
            code=f"import json; {name} = json.loads('''{serialized}''')",
            session_id=self._jupyter_session_id,
            timeout=5,
        )
        self._variables[name] = value

    def install_package(self, package: str) -> bool:
        """Install a Python package via pip.

        Args:
            package:
                Package name to install.

        Returns:
            True if installation succeeded.
        """
        try:
            result = self._client.shell.exec_command(
                command=f"pip install {package}",
                timeout=120,
            )
            return result.success and result.data.exit_code == 0
        except Exception:
            return False

    def list_packages(self) -> List[str]:
        """List installed packages.

        Returns:
            List of installed package names.
        """
        try:
            result = self._client.shell.exec_command(
                command="pip list --format=freeze",
                timeout=30,
            )
            if result.success and result.data.exit_code == 0:
                lines = result.data.output.strip().split("\n")
                return [line.split("==")[0] for line in lines if line]
        except Exception:  # noqa: S110
            pass  # Failed to list packages
        return []

    def reset(self) -> None:
        """Reset session and clear variables."""
        AgentSandboxBase.reset(self)
        self._variables.clear()
        self._create_jupyter_session()


@register_environment
class AgentSandboxShell(AgentSandboxBase, BaseShellEnvironment):
    """Shell command execution in agent-sandbox.

    This provider executes shell commands in the agent-sandbox Docker
    container. It supports:
    - Environment variables via the `vars` parameter
    - File mounting
    - Session persistence

    Example:
        >>> from msgflux.environments import Environments

        >>> env = Environments.code("shell/agent_sandbox")
        >>> result = env("ls -la /workspace")
        >>> print(result.output)

        >>> # With environment variables
        >>> result = env("echo $MY_VAR", vars={"MY_VAR": "hello"})
        >>> print(result.output)  # "hello"

        >>> # Check exit code
        >>> result = env("exit 1")
        >>> print(result.success)  # False
        >>> print(result.metadata["exit_code"])  # 1
    """

    provider = "agent_sandbox"
    environment_type = "shell"
    name = "execute_command"

    def __init__(self, **kwargs):
        """Initialize Shell environment.

        Args:
            **kwargs:
                Arguments passed to AgentSandboxBase:
                - base_url: URL of the agent-sandbox API
                - timeout: Default execution timeout
                - session_id: Optional session ID
        """
        AgentSandboxBase.__init__(self, **kwargs)
        self._initialize()

    def __call__(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,  # noqa: ARG002
    ) -> ExecutionResult:
        """Execute shell command.

        Args:
            action:
                Shell command to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Environment variables to set for this execution.
                Values are JSON-encoded if not strings.
            tools:
                NOT SUPPORTED. Kept for interface compatibility.

        Returns:
            ExecutionResult with output, errors, and metadata (exit_code).
        """
        if not self._initialized:
            raise SandboxNotReadyError("AgentSandbox not initialized")

        start_time = time.time()
        timeout = timeout or self.timeout

        # Prepend environment variable exports if provided
        command = action
        if vars:
            exports = []
            for k, v in vars.items():
                if isinstance(v, str):
                    exports.append(f'export {k}="{v}"')
                else:
                    exports.append(f"export {k}='{json.dumps(v)}'")
            command = " && ".join([*exports, action])

        try:
            result = self._client.shell.exec_command(
                command=command,
                timeout=float(timeout),
            )

            execution_time_ms = (time.time() - start_time) * 1000

            if not result.success:
                return ExecutionResult(
                    success=False,
                    error=str(getattr(result, "message", "Command failed")),
                    execution_time_ms=execution_time_ms,
                )

            exit_code = result.data.exit_code
            success = exit_code == 0

            return ExecutionResult(
                success=success,
                output=result.data.output,
                error=result.data.output if not success else None,
                metadata={"exit_code": exit_code},
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError as e:
            raise SandboxTimeoutError(timeout) from e

    async def acall(
        self,
        action: str,
        *,
        timeout: Optional[float] = None,
        vars: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,  # noqa: ARG002
    ) -> ExecutionResult:
        """Execute shell command asynchronously using native async client.

        Args:
            action:
                Shell command to execute.
            timeout:
                Execution timeout in seconds (overrides default).
            vars:
                Environment variables to set for this execution.
            tools:
                NOT SUPPORTED. Kept for interface compatibility.

        Returns:
            ExecutionResult with output, errors, and metadata (exit_code).
        """
        if not self._initialized:
            raise SandboxNotReadyError("AgentSandbox not initialized")

        start_time = time.time()
        timeout = timeout or self.timeout

        # Prepend environment variable exports if provided
        command = action
        if vars:
            exports = []
            for k, v in vars.items():
                if isinstance(v, str):
                    exports.append(f'export {k}="{v}"')
                else:
                    exports.append(f"export {k}='{json.dumps(v)}'")
            command = " && ".join([*exports, action])

        try:
            result = await self._async_client.shell.exec_command(
                command=command,
                timeout=float(timeout),
            )

            execution_time_ms = (time.time() - start_time) * 1000

            if not result.success:
                return ExecutionResult(
                    success=False,
                    error=str(getattr(result, "message", "Command failed")),
                    execution_time_ms=execution_time_ms,
                )

            exit_code = result.data.exit_code
            success = exit_code == 0

            return ExecutionResult(
                success=success,
                output=result.data.output,
                error=result.data.output if not success else None,
                metadata={"exit_code": exit_code},
                execution_time_ms=execution_time_ms,
            )

        except TimeoutError as e:
            raise SandboxTimeoutError(timeout) from e
