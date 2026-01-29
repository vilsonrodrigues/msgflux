"""Integration tests for sandbox providers."""

import pytest

from msgflux import Sandbox
from msgflux.environments.sandboxes.response import ExecutionResult


class TestMockSandbox:
    """Tests for MockSandbox provider."""

    def test_basic_execution(self):
        """Test basic code execution."""
        sandbox = Sandbox.python(provider="mock")
        result = sandbox("x = 1 + 2")

        assert result.success
        assert sandbox.get_variable("x") == 3

    def test_print_capture(self):
        """Test stdout capture."""
        sandbox = Sandbox.python(provider="mock")
        result = sandbox("print('Hello, World!')")

        assert result.success
        assert "Hello, World!" in result.output

    def test_variable_injection(self):
        """Test variable injection."""
        sandbox = Sandbox.python(provider="mock")
        result = sandbox("z = a + b", variables={"a": 10, "b": 20})

        assert result.success
        assert sandbox.get_variable("z") == 30

    def test_error_handling(self):
        """Test error handling."""
        sandbox = Sandbox.python(provider="mock")
        result = sandbox("x = undefined_variable")

        assert not result.success
        assert "NameError" in result.error

    def test_syntax_error(self):
        """Test syntax error handling."""
        sandbox = Sandbox.python(provider="mock")
        result = sandbox("if True print('bad')")

        assert not result.success
        assert "SyntaxError" in result.error

    def test_multiple_executions(self):
        """Test state persistence across executions."""
        sandbox = Sandbox.python(provider="mock")

        sandbox("x = 1")
        sandbox("y = 2")
        result = sandbox("z = x + y")

        assert result.success
        assert sandbox.get_variable("z") == 3

    def test_reset(self):
        """Test sandbox reset."""
        sandbox = Sandbox.python(provider="mock")

        sandbox("x = 42")
        assert sandbox.get_variable("x") == 42

        sandbox.reset()
        assert sandbox.get_variable("x") is None

    def test_context_manager(self):
        """Test context manager usage."""
        with Sandbox.python(provider="mock") as sandbox:
            result = sandbox("x = 1")
            assert result.success

    def test_call_history(self):
        """Test call history tracking."""
        sandbox = Sandbox.python(provider="mock")

        sandbox("x = 1")
        sandbox("y = 2", variables={"a": 10})

        assert sandbox.call_count == 2
        assert len(sandbox.call_history) == 2


class TestDenoPyodideSandbox:
    """Tests for DenoPyodideSandbox provider.

    These tests require Deno to be installed.
    """

    @pytest.fixture
    def sandbox(self):
        """Create a Deno sandbox for testing."""
        try:
            sandbox = Sandbox.python(provider="deno_pyodide", timeout=60.0)
            yield sandbox
            sandbox.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_basic_execution(self, sandbox):
        """Test basic code execution."""
        result = sandbox("x = 1 + 2")

        assert result.success
        assert sandbox.get_variable("x") == 3

    def test_print_capture(self, sandbox):
        """Test stdout capture."""
        result = sandbox("print('Hello from Pyodide!')")

        assert result.success
        assert "Hello from Pyodide!" in result.output

    def test_variable_injection(self, sandbox):
        """Test variable injection."""
        result = sandbox("result = a * b", variables={"a": 7, "b": 6})

        assert result.success
        assert sandbox.get_variable("result") == 42

    def test_complex_variables(self, sandbox):
        """Test injection of complex data structures."""
        data = {
            "name": "test",
            "values": [1, 2, 3, 4, 5],
            "nested": {"key": "value"},
        }
        result = sandbox("total = sum(data['values'])", variables={"data": data})

        assert result.success
        assert sandbox.get_variable("total") == 15

    def test_error_handling(self, sandbox):
        """Test error handling."""
        result = sandbox("x = undefined_variable")

        assert not result.success
        assert "NameError" in result.error

    def test_syntax_error(self, sandbox):
        """Test syntax error handling."""
        result = sandbox("if True print('bad')")

        assert not result.success
        assert "SyntaxError" in result.error

    def test_multiple_executions(self, sandbox):
        """Test state persistence across executions."""
        sandbox("x = 10")
        sandbox("y = 20")
        result = sandbox("z = x + y")

        assert result.success
        assert sandbox.get_variable("z") == 30

    def test_builtin_functions(self, sandbox):
        """Test Python builtin functions."""
        result = sandbox("""
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
        result_dict = sandbox.get_variable("result")
        assert result_dict["sum"] == 31
        assert result_dict["max"] == 9
        assert result_dict["min"] == 1
        assert result_dict["len"] == 8
        assert result_dict["sorted"] == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_list_comprehension(self, sandbox):
        """Test list comprehensions."""
        result = sandbox("squares = [x**2 for x in range(10)]")

        assert result.success
        assert sandbox.get_variable("squares") == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    def test_dict_comprehension(self, sandbox):
        """Test dict comprehensions."""
        result = sandbox("mapping = {x: x**2 for x in range(5)}")

        assert result.success
        assert sandbox.get_variable("mapping") == {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

    def test_function_definition(self, sandbox):
        """Test function definition and execution."""
        result = sandbox("""
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
""")

        assert result.success
        assert sandbox.get_variable("result") == 120

    def test_class_definition(self, sandbox):
        """Test class definition."""
        result = sandbox("""
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
        assert sandbox.get_variable("result") == 11

    def test_import_builtin_module(self, sandbox):
        """Test importing builtin modules."""
        result = sandbox("""
import math
result = math.sqrt(16)
""")

        assert result.success
        assert sandbox.get_variable("result") == 4.0

    def test_json_module(self, sandbox):
        """Test JSON module."""
        result = sandbox("""
import json
data = {'key': 'value', 'number': 42}
json_str = json.dumps(data)
parsed = json.loads(json_str)
""")

        assert result.success
        assert sandbox.get_variable("parsed") == {"key": "value", "number": 42}

    def test_reset(self, sandbox):
        """Test sandbox reset."""
        sandbox("x = 42")
        assert sandbox.get_variable("x") == 42

        sandbox.reset()
        assert sandbox.get_variable("x") is None

    def test_execution_time_measured(self, sandbox):
        """Test that execution time is measured."""
        result = sandbox("x = sum(range(1000))")

        assert result.success
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    def test_multiline_code(self, sandbox):
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
        result = sandbox(code)

        assert result.success
        assert sandbox.get_variable("result") == 30


class TestSandboxFactory:
    """Tests for Sandbox factory class."""

    def test_providers_available(self):
        """Test that providers are listed correctly."""
        providers = Sandbox.providers()

        assert "python" in providers
        assert "mock" in providers["python"]
        assert "deno_pyodide" in providers["python"]

    def test_sandbox_types(self):
        """Test sandbox types listing."""
        types = Sandbox.sandbox_types()

        assert "python" in types

    def test_invalid_provider(self):
        """Test error on invalid provider."""
        with pytest.raises(ValueError, match="not registered"):
            Sandbox.python(provider="invalid_provider")

    def test_invalid_sandbox_type(self):
        """Test error on invalid sandbox type."""
        with pytest.raises(ValueError, match="not supported"):
            Sandbox._create_sandbox("invalid_type", "mock")
