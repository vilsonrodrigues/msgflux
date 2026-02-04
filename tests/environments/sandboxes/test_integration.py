"""Integration tests for code environment providers."""

import pytest

from msgflux.environments import Environments


class TestDenoPyodideSandbox:
    """Tests for DenoPyodideSandbox provider.

    These tests require Deno to be installed.
    """

    @pytest.fixture
    def env(self):
        """Create a Deno environment for testing."""
        try:
            environment = Environments.code("python/deno_pyodide", timeout=60.0)
            yield environment
            environment.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_basic_execution(self, env):
        """Test basic code execution."""
        result = env("x = 1 + 2")

        assert result.success
        assert env.get_variable("x") == 3

    def test_print_capture(self, env):
        """Test stdout capture."""
        result = env("print('Hello from Pyodide!')")

        assert result.success
        assert "Hello from Pyodide!" in result.output

    def test_variable_injection(self, env):
        """Test variable injection."""
        result = env("result = a * b", vars={"a": 7, "b": 6})

        assert result.success
        assert env.get_variable("result") == 42

    def test_complex_variables(self, env):
        """Test injection of complex data structures."""
        data = {
            "name": "test",
            "values": [1, 2, 3, 4, 5],
            "nested": {"key": "value"},
        }
        result = env("total = sum(data['values'])", vars={"data": data})

        assert result.success
        assert env.get_variable("total") == 15

    def test_error_handling(self, env):
        """Test error handling."""
        result = env("x = undefined_variable")

        assert not result.success
        assert "NameError" in result.error

    def test_syntax_error(self, env):
        """Test syntax error handling."""
        result = env("if True print('bad')")

        assert not result.success
        assert "SyntaxError" in result.error

    def test_multiple_executions(self, env):
        """Test state persistence across executions."""
        env("x = 10")
        env("y = 20")
        result = env("z = x + y")

        assert result.success
        assert env.get_variable("z") == 30

    def test_builtin_functions(self, env):
        """Test Python builtin functions."""
        result = env("""
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
result = {
    'sum': sum(numbers),
    'max': max(numbers),
    'min': min(numbers),
    'len': len(numbers),
    'sorted': sorted(numbers),
}
""")

        assert result.success
        result_dict = env.get_variable("result")
        assert result_dict["sum"] == 31
        assert result_dict["max"] == 9
        assert result_dict["min"] == 1
        assert result_dict["len"] == 8
        assert result_dict["sorted"] == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_list_comprehension(self, env):
        """Test list comprehensions."""
        result = env("squares = [x**2 for x in range(10)]")

        assert result.success
        assert env.get_variable("squares") == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    def test_dict_comprehension(self, env):
        """Test dict comprehensions."""
        result = env("mapping = {x: x**2 for x in range(5)}")

        assert result.success
        assert env.get_variable("mapping") == {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

    def test_function_definition(self, env):
        """Test function definition and execution."""
        result = env("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
""")

        assert result.success
        assert env.get_variable("result") == 120

    def test_class_definition(self, env):
        """Test class definition."""
        result = env("""
class Counter:
    def __init__(self, start=0):
        self.value = start

    def increment(self):
        self.value += 1
        return self.value

c = Counter(10)
result = c.increment()
""")

        assert result.success
        assert env.get_variable("result") == 11

    def test_import_builtin_module(self, env):
        """Test importing builtin modules."""
        result = env("""
import math
result = math.sqrt(16)
""")

        assert result.success
        assert env.get_variable("result") == 4.0

    def test_json_module(self, env):
        """Test JSON module."""
        result = env("""
import json
data = {'key': 'value', 'number': 42}
json_str = json.dumps(data)
parsed = json.loads(json_str)
""")

        assert result.success
        assert env.get_variable("parsed") == {"key": "value", "number": 42}

    def test_reset(self, env):
        """Test environment reset."""
        env("x = 42")
        assert env.get_variable("x") == 42

        env.reset()
        assert env.get_variable("x") is None

    def test_execution_time_measured(self, env):
        """Test that execution time is measured."""
        result = env("x = sum(range(1000))")

        assert result.success
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    def test_multiline_code(self, env):
        """Test multiline code execution."""
        code = """
# This is a comment
x = 1
y = 2

# Another comment
z = x + y

# Final result
result = z * 10
"""
        result = env(code)

        assert result.success
        assert env.get_variable("result") == 30


class TestEnvironmentsFactory:
    """Tests for Environments factory class."""

    def test_list_types(self):
        """Test that environment types are listed correctly."""
        types = Environments.list_types()

        # Python type should always be available
        assert "python" in types

    def test_list_providers(self):
        """Test that providers are listed correctly."""
        # Ensure at least deno_pyodide is registered
        providers = Environments.list_providers("python")

        assert "deno_pyodide" in providers

    def test_invalid_provider(self):
        """Test error on invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Environments.code("python/invalid_provider_xyz")

    def test_invalid_environment_type(self):
        """Test error on invalid environment type."""
        with pytest.raises(ValueError, match="Unknown environment type"):
            Environments.code("invalid_type_xyz")
