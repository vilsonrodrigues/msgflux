"""Integration tests for CodeAct with Python sandbox.

These tests verify the integration between:
- nn.Agent with generation_schema=CodeAct
- nn.Environment wrapping DenoPyodideSandbox
- Tool injection into the sandbox for code to call

To run with a real model:
    OPENAI_API_KEY=... pytest tests/integration/test_codeact_sandbox.py -v

To run the manual test:
    python tests/integration/test_codeact_sandbox.py
"""

import os

import pytest

from msgflux.environments import Environments
from msgflux.generation.control_flow import EnvironmentCall, FlowResult
from msgflux.generation.reasoning.code_act import CodeAct
from msgflux.nn.modules.environment import Environment


class TestCodeActIntegration:
    """Integration tests for CodeAct with sandbox."""

    @pytest.fixture
    def python_env(self):
        """Create a Python environment for testing."""
        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
            yield env
            env.shutdown()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_extract_flow_result_with_code(self):
        """Test that extract_flow_result creates environment_call for code."""
        raw_response = {
            "current_step": {
                "thought": "I need to search for Python documentation",
                "actions": {"code": "result = search('Python docs')\nprint(result)"},
            },
            "final_answer": None,
        }

        flow_result = CodeAct.extract_flow_result(raw_response)

        assert not flow_result.is_complete
        assert flow_result.environment_call is not None
        assert flow_result.environment_call.action == "result = search('Python docs')\nprint(result)"
        assert flow_result.environment_call.inject_vars is True
        assert flow_result.environment_call.inject_tools is True
        assert flow_result.reasoning == "I need to search for Python documentation"

    def test_extract_flow_result_with_final_answer(self):
        """Test that extract_flow_result completes on final_answer."""
        raw_response = {
            "current_step": None,
            "final_answer": "Python is a programming language",
        }

        flow_result = CodeAct.extract_flow_result(raw_response)

        assert flow_result.is_complete
        assert flow_result.final_response == raw_response

    def test_inject_environment_result_success(self):
        """Test injecting successful execution result."""
        raw_response = {
            "current_step": {
                "thought": "Search for info",
                "actions": {"code": "print(search('test'))"},
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "Search results for: test\n", "error": None}
        updated = CodeAct.inject_environment_result(raw_response, result)

        assert updated["current_step"]["actions"]["result"] == "Search results for: test"

    def test_inject_environment_result_error(self):
        """Test injecting error result."""
        raw_response = {
            "current_step": {
                "thought": "Call undefined function",
                "actions": {"code": "result = undefined_tool()"},
            },
            "final_answer": None,
        }

        result = {
            "success": False,
            "output": "",
            "error": "NameError: name 'undefined_tool' is not defined",
        }
        updated = CodeAct.inject_environment_result(raw_response, result)

        assert "Error:" in updated["current_step"]["actions"]["result"]
        assert "NameError" in updated["current_step"]["actions"]["result"]

    def test_environment_executes_code_with_tools(self, python_env):
        """Test that environment executes code that uses injected tools."""

        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        # Simulate what Agent does
        raw_response = {
            "current_step": {
                "thought": "I need to search for Python info",
                "actions": {"code": "result = search('Python')\nprint(result)"},
            },
            "final_answer": None,
        }

        flow_result = CodeAct.extract_flow_result(raw_response)
        assert flow_result.environment_call is not None

        # Execute in environment with tool injected
        env_call = flow_result.environment_call
        exec_result = python_env(env_call.action, tools={"search": search})

        assert exec_result.success
        assert "Results for: Python" in exec_result.output

        # Inject result back
        result_dict = {
            "success": exec_result.success,
            "output": exec_result.output,
            "error": exec_result.error,
        }
        updated = CodeAct.inject_environment_result(raw_response, result_dict)

        assert "Results for: Python" in updated["current_step"]["actions"]["result"]

    def test_environment_with_multiple_tools(self, python_env):
        """Test environment execution with multiple injected tools."""

        def search(query: str) -> str:
            """Search for information."""
            return f"Found: {query}"

        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            return str(eval(expression))

        # Execute code that uses multiple tools
        result = python_env(
            "s = search('math')\nc = calculate('2 + 2')\nprint(f'{s}, {c}')",
            tools={"search": search, "calculate": calculate},
        )

        assert result.success
        assert "Found: math" in result.output
        assert "4" in result.output

    def test_environment_with_vars(self, python_env):
        """Test environment execution with injected variables."""

        def process(data: list) -> int:
            """Process data and return count."""
            return len(data)

        result = python_env(
            "count = process(items)\nprint(f'Processed {count} items')",
            tools={"process": process},
            vars={"items": [1, 2, 3, 4, 5]},
        )

        assert result.success
        assert "Processed 5 items" in result.output

    def test_multi_step_execution_with_tools(self, python_env):
        """Test multiple execution steps with tools and state persistence."""

        def fetch_data(source: str) -> list:
            """Fetch data from source."""
            return [1, 2, 3, 4, 5] if source == "source_a" else [10, 20, 30]

        def aggregate(values: list) -> int:
            """Aggregate values."""
            return sum(values)

        # Step 1: Fetch data
        result1 = python_env(
            "data_a = fetch_data('source_a')\nprint(f'Fetched: {data_a}')",
            tools={"fetch_data": fetch_data, "aggregate": aggregate},
        )
        assert result1.success
        assert "[1, 2, 3, 4, 5]" in result1.output

        # Step 2: Aggregate (state persists)
        result2 = python_env(
            "total = aggregate(data_a)\nprint(f'Total: {total}')",
            tools={"fetch_data": fetch_data, "aggregate": aggregate},
        )
        assert result2.success
        assert "Total: 15" in result2.output

    def test_error_handling_in_tool_call(self, python_env):
        """Test that errors from tools are properly captured."""

        def failing_tool(x: int) -> int:
            """A tool that fails."""
            raise ValueError("Tool failed intentionally")

        result = python_env(
            "result = failing_tool(42)",
            tools={"failing_tool": failing_tool},
        )

        assert not result.success
        assert result.error is not None
        # Note: Deno/Pyodide may serialize tool errors as JsException
        # The important thing is that the error is captured
        error_indicators = ["ValueError", "Tool failed", "JsException", "Error"]
        assert any(indicator in result.error for indicator in error_indicators)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestCodeActWithModel:
    """Tests that require a real LLM model."""

    @pytest.fixture
    def agent(self):
        """Create an agent with CodeAct."""
        from msgflux import Model
        from msgflux.nn import Agent, LM

        model = Model.chat_completion("openai/gpt-4.1-mini")
        lm = LM(model=model)

        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

        def search(query: str) -> str:
            """Search for information about a topic."""
            # Simulated search results
            results = {
                "python": "Python is a high-level programming language.",
                "factorial": "Factorial of n is the product of all positive integers <= n.",
            }
            for key, value in results.items():
                if key in query.lower():
                    return value
            return f"No results found for: {query}"

        def calculate(expression: str) -> str:
            """Calculate a mathematical expression."""
            try:
                return str(eval(expression))
            except Exception as e:
                return f"Error: {e}"

        agent = Agent(
            name="researcher",
            model=lm,
            environment=env,
            tools=[search, calculate],
            generation_schema=CodeAct,
        )

        yield agent
        env.shutdown()

    def test_simple_tool_usage(self, agent):
        """Test a simple task that uses a tool."""
        result = agent("What is Python? Use the search tool to find out.")
        assert result.get("final_answer") is not None
        # Should mention Python in some form
        final = result.get("final_answer", "").lower()
        assert "python" in final or "programming" in final or "language" in final

    def test_calculation_via_code(self, agent):
        """Test agent using calculate tool via code."""
        result = agent("What is 7 * 8? Use the calculate tool.")
        assert result.get("final_answer") is not None
        assert "56" in result.get("final_answer", "")


class TestCodeActBuildHistory:
    """Tests for history building in CodeAct."""

    def test_build_history_first_step(self):
        """Test building history for first step."""
        raw_response = {
            "current_step": {
                "thought": "First",
                "actions": {"code": "print(1)", "result": "1"},
            },
            "final_answer": None,
        }

        messages = []
        result = CodeAct.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_build_history_append(self):
        """Test appending to existing history."""
        import msgspec

        first_response = {
            "current_step": {
                "thought": "First",
                "actions": {"code": "x = search('a')", "result": "result_a"},
            },
            "final_answer": None,
        }
        second_response = {
            "current_step": {
                "thought": "Second",
                "actions": {"code": "y = search('b')", "result": "result_b"},
            },
            "final_answer": None,
        }

        messages = []
        CodeAct.build_history(first_response, messages)
        CodeAct.build_history(second_response, messages)

        # Should have one message with both steps
        assert len(messages) == 1
        content = msgspec.json.decode(messages[0]["content"])
        assert len(content) == 2


# Manual test for running with a real model
def manual_test():
    """Manual test with real model - run directly with python."""
    try:
        from msgflux import Model
        from msgflux.nn import Agent, LM
    except ImportError:
        print("msgflux not fully installed, skipping manual test")
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run manual test")
        print("\nExample:")
        print("  OPENAI_API_KEY=sk-... python tests/integration/test_codeact_sandbox.py")
        return

    print("=" * 60)
    print("Manual Integration Test: CodeAct + Sandbox")
    print("=" * 60)

    # Create model using chat_completion
    model = Model.chat_completion("openai/gpt-4.1-mini")
    lm = LM(model=model)

    # Create environment
    try:
        code_env = Environments.code("python", timeout=60.0)
        env = Environment(environment=code_env)
    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("Make sure Deno is installed: curl -fsSL https://deno.land/install.sh | sh")
        return

    # Define tools
    def search(query: str) -> str:
        """Search for information about a topic."""
        print(f"[Tool] search called with: {query}")
        results = {
            "python": "Python is a high-level, interpreted programming language known for its simplicity.",
            "prime": "Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves.",
            "factorial": "Factorial (n!) is the product of all positive integers less than or equal to n.",
        }
        for key, value in results.items():
            if key in query.lower():
                return value
        return f"Information about: {query}"

    def calculate(expression: str) -> str:
        """Calculate a mathematical expression safely."""
        print(f"[Tool] calculate called with: {expression}")
        try:
            # Simple safe eval for basic math
            allowed_names = {"__builtins__": {}}
            result = eval(expression, allowed_names, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # Create agent with CodeAct
    agent = Agent(
        name="researcher",
        model=lm,
        environment=env,
        tools=[search, calculate],
        generation_schema=CodeAct,
        config={"verbose": True},
    )

    # Test problem
    problem = "Find information about Python and then calculate 7 * 8 + 5"

    print(f"\nProblem: {problem}")
    print("-" * 60)

    try:
        result = agent(problem)
        print(f"\nFinal Answer: {result.get('final_answer', 'No answer')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.shutdown()

    print("=" * 60)


if __name__ == "__main__":
    manual_test()
