"""Integration tests for AgentSandbox providers.

Requires:
    - Docker running with agent-sandbox container:
      docker run --rm -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
    - pip install msgflux[agent-sandbox]
"""

import asyncio

import httpx
import pytest

from msgflux.environments import Environments


def is_agent_sandbox_available():
    """Check if agent-sandbox is available."""
    try:
        # The agent-sandbox serves an HTML page at root
        response = httpx.get("http://localhost:8080/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


# Skip all tests if agent-sandbox is not available
pytestmark = pytest.mark.skipif(
    not is_agent_sandbox_available(),
    reason="agent-sandbox container not running on localhost:8080",
)


class TestAgentSandboxPython:
    """Tests for AgentSandboxPython provider (python/agent_sandbox)."""

    @pytest.fixture
    def env(self):
        """Create Python environment for testing."""
        environment = Environments.code("python/agent_sandbox", timeout=60.0)
        yield environment
        environment.shutdown()

    def test_basic_execution(self, env):
        """Test basic Python code execution."""
        result = env("x = 1 + 2; print(x)")

        assert result.success, f"Execution failed: {result.error}"
        assert "3" in result.output

    def test_variable_persistence(self, env):
        """Test that variables persist across executions."""
        env("x = 10")
        env("y = 20")
        result = env("z = x + y; print(z)")

        assert result.success, f"Execution failed: {result.error}"
        assert "30" in result.output

    def test_variable_injection(self, env):
        """Test variable injection via vars parameter."""
        result = env("result = a * b; print(result)", vars={"a": 7, "b": 6})

        assert result.success, f"Execution failed: {result.error}"
        assert "42" in result.output

    def test_complex_data_injection(self, env):
        """Test injection of complex data structures."""
        data = {
            "name": "test",
            "values": [1, 2, 3, 4, 5],
            "nested": {"key": "value"},
        }
        result = env("total = sum(data['values']); print(total)", vars={"data": data})

        assert result.success, f"Execution failed: {result.error}"
        assert "15" in result.output

    def test_error_handling(self, env):
        """Test error handling for undefined variables."""
        result = env("x = undefined_variable")

        assert not result.success
        assert result.error is not None
        assert "NameError" in result.error

    def test_syntax_error(self, env):
        """Test syntax error handling."""
        result = env("if True print('bad')")

        assert not result.success
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_multiline_code(self, env):
        """Test multiline code execution."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
"""
        result = env(code)

        assert result.success, f"Execution failed: {result.error}"
        assert "120" in result.output

    def test_import_module(self, env):
        """Test importing Python modules."""
        result = env("""
import math
result = math.sqrt(16)
print(result)
""")

        assert result.success, f"Execution failed: {result.error}"
        assert "4.0" in result.output

    def test_list_comprehension(self, env):
        """Test list comprehensions."""
        result = env("squares = [x**2 for x in range(5)]; print(squares)")

        assert result.success, f"Execution failed: {result.error}"
        assert "[0, 1, 4, 9, 16]" in result.output

    def test_execution_time_measured(self, env):
        """Test that execution time is measured."""
        result = env("x = sum(range(1000))")

        assert result.success
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    def test_get_variable(self, env):
        """Test getting variable from environment."""
        env("my_var = 42")
        value = env.get_variable("my_var")

        assert value == 42

    def test_set_variable(self, env):
        """Test setting variable in environment."""
        env.set_variable("injected", 100)
        result = env("print(injected)")

        assert result.success
        assert "100" in result.output


class TestAgentSandboxShell:
    """Tests for AgentSandboxShell provider (shell/agent_sandbox)."""

    @pytest.fixture
    def env(self):
        """Create Shell environment for testing."""
        environment = Environments.code("shell/agent_sandbox", timeout=60.0)
        yield environment
        environment.shutdown()

    def test_basic_command(self, env):
        """Test basic shell command execution."""
        result = env("echo 'Hello from Shell!'")

        assert result.success, f"Command failed: {result.error}"
        assert "Hello from Shell!" in result.output

    def test_exit_code_success(self, env):
        """Test exit code on success."""
        result = env("true")

        assert result.success
        assert result.metadata.get("exit_code") == 0

    def test_exit_code_failure(self, env):
        """Test exit code on failure."""
        # Use a command that fails (file not found) instead of 'exit'
        # because 'exit' has special behavior in agent-sandbox
        result = env("cat /nonexistent/file/path")

        assert not result.success
        assert result.metadata.get("exit_code") != 0

    def test_command_with_pipe(self, env):
        """Test command with pipe."""
        result = env("echo 'hello world' | tr 'a-z' 'A-Z'")

        assert result.success, f"Command failed: {result.error}"
        assert "HELLO WORLD" in result.output

    def test_environment_variables(self, env):
        """Test environment variable injection."""
        result = env("echo $MY_VAR", vars={"MY_VAR": "test_value"})

        assert result.success, f"Command failed: {result.error}"
        assert "test_value" in result.output

    def test_working_directory(self, env):
        """Test working directory."""
        result = env("pwd")

        assert result.success, f"Command failed: {result.error}"
        assert result.output.strip() != ""

    def test_list_files(self, env):
        """Test listing files."""
        result = env("ls /")

        assert result.success, f"Command failed: {result.error}"
        # Common directories in a Linux container
        assert any(d in result.output for d in ["bin", "etc", "home", "usr"])

    def test_command_not_found(self, env):
        """Test handling of unknown command."""
        result = env("nonexistent_command_xyz")

        assert not result.success
        assert result.metadata.get("exit_code") != 0

    def test_multiline_script(self, env):
        """Test multiline shell script."""
        script = """
for i in 1 2 3; do
    echo "Number: $i"
done
"""
        result = env(script)

        assert result.success, f"Command failed: {result.error}"
        assert "Number: 1" in result.output
        assert "Number: 2" in result.output
        assert "Number: 3" in result.output

    def test_file_operations(self, env):
        """Test file creation and reading."""
        # Create a file
        result1 = env("echo 'test content' > /tmp/test_file.txt")
        assert result1.success, f"Write failed: {result1.error}"

        # Read the file
        result2 = env("cat /tmp/test_file.txt")
        assert result2.success, f"Read failed: {result2.error}"
        assert "test content" in result2.output

        # Clean up
        env("rm /tmp/test_file.txt")

    def test_execution_time_measured(self, env):
        """Test that execution time is measured."""
        result = env("sleep 0.1")

        assert result.success
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 100  # At least 100ms


class TestAsyncAPI:
    """Tests for async API (acall method)."""

    @pytest.fixture
    def py_env(self):
        """Create Python environment for testing."""
        environment = Environments.code("python/agent_sandbox", timeout=60.0)
        yield environment
        environment.shutdown()

    @pytest.fixture
    def sh_env(self):
        """Create Shell environment for testing."""
        environment = Environments.code("shell/agent_sandbox", timeout=60.0)
        yield environment
        environment.shutdown()

    @pytest.mark.asyncio
    async def test_async_python_basic(self, py_env):
        """Test async Python execution."""
        result = await py_env.acall("x = 1 + 2; print(x)")

        assert result.success, f"Execution failed: {result.error}"
        assert "3" in result.output

    @pytest.mark.asyncio
    async def test_async_python_variable_persistence(self, py_env):
        """Test variable persistence in async mode."""
        await py_env.acall("x = 10")
        result = await py_env.acall("print(x * 5)")

        assert result.success, f"Execution failed: {result.error}"
        assert "50" in result.output

    @pytest.mark.asyncio
    async def test_async_python_variable_injection(self, py_env):
        """Test variable injection in async mode."""
        result = await py_env.acall(
            "print(sum(numbers))", vars={"numbers": [1, 2, 3, 4, 5]}
        )

        assert result.success, f"Execution failed: {result.error}"
        assert "15" in result.output

    @pytest.mark.asyncio
    async def test_async_shell_basic(self, sh_env):
        """Test async Shell execution."""
        result = await sh_env.acall("echo 'Hello Async!'")

        assert result.success, f"Execution failed: {result.error}"
        assert "Hello Async!" in result.output

    @pytest.mark.asyncio
    async def test_async_shell_env_vars(self, sh_env):
        """Test environment variables in async mode."""
        result = await sh_env.acall("echo $MY_VAR", vars={"MY_VAR": "async_test"})

        assert result.success, f"Execution failed: {result.error}"
        assert "async_test" in result.output

    @pytest.mark.asyncio
    async def test_async_concurrent_execution(self):
        """Test concurrent async execution."""
        py_env = Environments.code("python/agent_sandbox", timeout=60.0)
        sh_env = Environments.code("shell/agent_sandbox", timeout=60.0)

        try:
            # Run both in parallel
            py_task = py_env.acall("import time; time.sleep(0.1); print('python')")
            sh_task = sh_env.acall("sleep 0.1 && echo 'shell'")

            py_result, sh_result = await asyncio.gather(py_task, sh_task)

            assert py_result.success, f"Python failed: {py_result.error}"
            assert sh_result.success, f"Shell failed: {sh_result.error}"
            assert "python" in py_result.output
            assert "shell" in sh_result.output
        finally:
            py_env.shutdown()
            sh_env.shutdown()


class TestMixedUsage:
    """Tests for using both Python and Shell environments."""

    def test_python_and_shell_independent(self):
        """Test that Python and Shell environments work independently."""
        py_env = Environments.code("python/agent_sandbox", timeout=60.0)
        sh_env = Environments.code("shell/agent_sandbox", timeout=60.0)

        try:
            # Execute in Python
            py_result = py_env("x = 42; print(x)")
            assert py_result.success
            assert "42" in py_result.output

            # Execute in Shell
            sh_result = sh_env("echo 42")
            assert sh_result.success
            assert "42" in sh_result.output

        finally:
            py_env.shutdown()
            sh_env.shutdown()

    def test_file_sharing_between_environments(self):
        """Test that files can be shared between Python and Shell."""
        py_env = Environments.code("python/agent_sandbox", timeout=60.0)
        sh_env = Environments.code("shell/agent_sandbox", timeout=60.0)

        try:
            # Create file with Python
            py_result = py_env("""
with open('/tmp/shared_file.txt', 'w') as f:
    f.write('Hello from Python!')
print('File created')
""")
            assert py_result.success, f"Python failed: {py_result.error}"

            # Read file with Shell
            sh_result = sh_env("cat /tmp/shared_file.txt")
            assert sh_result.success, f"Shell failed: {sh_result.error}"
            assert "Hello from Python!" in sh_result.output

            # Clean up
            sh_env("rm /tmp/shared_file.txt")

        finally:
            py_env.shutdown()
            sh_env.shutdown()
