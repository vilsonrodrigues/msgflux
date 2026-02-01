"""Integration tests for nn.Environment module."""

import pytest

from msgflux.environments import Environments
from msgflux.environments.code.response import ExecutionResult
from msgflux.nn.modules.environment import Environment


class TestEnvironmentModule:
    """Tests for nn.Environment wrapper module."""

    @pytest.fixture
    def env(self):
        """Create Environment with Deno sandbox."""
        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
            yield env
            env.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_basic_execution(self, env):
        """Test basic code execution via Environment."""
        result = env("x = 1 + 2\nprint(x)")

        assert isinstance(result, ExecutionResult)
        assert result.success
        assert "3" in result.output

    def test_environment_name(self, env):
        """Test environment name property."""
        assert env.name == "execute_code"

    def test_environment_type(self, env):
        """Test environment type property."""
        assert env.environment_type == "python"

    def test_variable_injection(self, env):
        """Test vars injection."""
        result = env("result = a + b", vars={"a": 10, "b": 20})

        assert result.success

    def test_tool_registration(self, env):
        """Test tool registration and execution."""

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        result = env(
            "result = add(5, 3)\nprint(result)",
            tools={"add": add},
        )

        assert result.success
        assert "8" in result.output

    def test_tool_with_vars(self, env):
        """Test tools and vars together."""

        def multiply(x: int, y: int) -> int:
            return x * y

        result = env(
            "result = multiply(factor, 10)\nprint(result)",
            tools={"multiply": multiply},
            vars={"factor": 5},
        )

        assert result.success
        assert "50" in result.output

    def test_init_tools_available(self, env):
        """Test that init tools are available in executions."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        # Create new env with init tools
        from msgflux.environments import Environments
        from msgflux.nn.modules.environment import Environment

        try:
            code_env = Environments.code("python", timeout=60.0)
            env_with_tools = Environment(
                environment=code_env,
                tools={"greet": greet}
            )
            result = env_with_tools("msg = greet('World')\nprint(msg)")
            assert result.success
            assert "Hello, World!" in result.output
            env_with_tools.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_error_handling(self, env):
        """Test error handling."""
        result = env("x = undefined_var")

        assert not result.success
        assert result.error is not None

    def test_reset(self, env):
        """Test environment reset."""
        env("x = 1")
        env.reset()
        # After reset, previous variables should be cleared
        result = env("print(x)")
        assert not result.success  # x should not exist

    def test_context_manager(self):
        """Test context manager usage."""
        try:
            code_env = Environments.code("python", timeout=30.0)
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

        with Environment(environment=code_env) as env:
            result = env("print('hello')")
            assert result.success


class TestEnvironmentsFactory:
    """Tests for Environments factory."""

    def test_list_types(self):
        """Test listing environment types."""
        types = Environments.list_types()
        assert "python" in types

    def test_list_providers(self):
        """Test listing providers for a type."""
        providers = Environments.list_providers("python")
        assert "deno_pyodide" in providers

    def test_invalid_type_raises(self):
        """Test invalid environment type raises."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            Environments.code("invalid_type")

    def test_invalid_provider_raises(self):
        """Test invalid provider raises."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Environments.code("python", provider="invalid")


@pytest.mark.asyncio
class TestEnvironmentAsync:
    """Async tests for nn.Environment."""

    @pytest.fixture
    def env(self):
        """Create Environment with Deno sandbox."""
        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
            yield env
            env.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    async def test_async_execution(self, env):
        """Test async code execution."""
        result = await env.acall("x = 2 ** 10\nprint(x)")

        assert result.success
        assert "1024" in result.output

    async def test_async_with_tools(self, env):
        """Test async execution with tools."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await env.acall(
            "msg = greet('World')\nprint(msg)",
            tools={"greet": greet},
        )

        assert result.success
        assert "Hello, World!" in result.output
