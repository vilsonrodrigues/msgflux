"""Integration tests for ProgramOfThought with Python sandbox.

These tests verify the integration between:
- nn.Agent with generation_schema=ProgramOfThought
- nn.Environment wrapping DenoPyodideSandbox
- Tool injection into the sandbox

To run with a real model:
    OPENAI_API_KEY=... pytest tests/integration/test_pot_sandbox.py -v

To run the manual test:
    python tests/integration/test_pot_sandbox.py
"""

import os

import pytest

from msgflux.environments import Environments
from msgflux.generation.control_flow import EnvironmentCall, FlowResult
from msgflux.generation.reasoning.program_of_thought import ProgramOfThought
from msgflux.nn.modules.environment import Environment


class TestProgramOfThoughtIntegration:
    """Integration tests for ProgramOfThought with sandbox."""

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
                "thought": "I need to calculate 2 + 2",
                "code": "result = 2 + 2\nprint(result)",
            },
            "final_answer": None,
        }

        flow_result = ProgramOfThought.extract_flow_result(raw_response)

        assert not flow_result.is_complete
        assert flow_result.environment_call is not None
        assert flow_result.environment_call.action == "result = 2 + 2\nprint(result)"
        assert flow_result.environment_call.inject_vars is True
        assert flow_result.environment_call.inject_tools is True
        assert flow_result.reasoning == "I need to calculate 2 + 2"

    def test_extract_flow_result_with_final_answer(self):
        """Test that extract_flow_result completes on final_answer."""
        raw_response = {
            "current_step": None,
            "final_answer": "The answer is 4",
        }

        flow_result = ProgramOfThought.extract_flow_result(raw_response)

        assert flow_result.is_complete
        assert flow_result.final_response == raw_response

    def test_inject_environment_result_success(self):
        """Test injecting successful execution result."""
        raw_response = {
            "current_step": {
                "thought": "Calculate",
                "code": "print(2 + 2)",
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "4\n", "error": None}
        updated = ProgramOfThought.inject_environment_result(raw_response, result)

        assert updated["current_step"]["result"] == "4"

    def test_inject_environment_result_error(self):
        """Test injecting error result."""
        raw_response = {
            "current_step": {
                "thought": "Calculate",
                "code": "print(undefined)",
            },
            "final_answer": None,
        }

        result = {
            "success": False,
            "output": "",
            "error": "NameError: name 'undefined' is not defined",
        }
        updated = ProgramOfThought.inject_environment_result(raw_response, result)

        assert "Error:" in updated["current_step"]["result"]
        assert "NameError" in updated["current_step"]["result"]

    def test_environment_executes_code(self, python_env):
        """Test that environment executes code from ProgramOfThought."""
        # Simulate what Agent does
        raw_response = {
            "current_step": {
                "thought": "I need to calculate factorial of 5",
                "code": "import math\nresult = math.factorial(5)\nprint(f'5! = {result}')",
            },
            "final_answer": None,
        }

        flow_result = ProgramOfThought.extract_flow_result(raw_response)
        assert flow_result.environment_call is not None

        # Execute in environment
        env_call = flow_result.environment_call
        exec_result = python_env(env_call.action)

        assert exec_result.success
        assert "5! = 120" in exec_result.output

        # Inject result back
        result_dict = {
            "success": exec_result.success,
            "output": exec_result.output,
            "error": exec_result.error,
        }
        updated = ProgramOfThought.inject_environment_result(raw_response, result_dict)

        assert "5! = 120" in updated["current_step"]["result"]

    def test_environment_with_tools(self, python_env):
        """Test environment execution with injected tools."""

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        # Execute code that uses the tool
        result = python_env(
            "result = multiply(7, 6)\nprint(f'7 x 6 = {result}')",
            tools={"multiply": multiply},
        )

        assert result.success
        assert "7 x 6 = 42" in result.output

    def test_environment_with_vars(self, python_env):
        """Test environment execution with injected variables."""
        result = python_env(
            "total = sum(numbers)\nprint(f'Sum: {total}')",
            vars={"numbers": [1, 2, 3, 4, 5]},
        )

        assert result.success
        assert "Sum: 15" in result.output

    def test_multi_step_execution(self, python_env):
        """Test multiple execution steps with state persistence."""
        # Step 1: Define data
        result1 = python_env("data = [1, 2, 3, 4, 5]\nprint('Data defined')")
        assert result1.success

        # Step 2: Process data (state persists)
        result2 = python_env("squared = [x**2 for x in data]\nprint(squared)")
        assert result2.success
        assert "[1, 4, 9, 16, 25]" in result2.output

        # Step 3: Aggregate
        result3 = python_env("total = sum(squared)\nprint(f'Total: {total}')")
        assert result3.success
        assert "Total: 55" in result3.output

    def test_error_handling_in_code(self, python_env):
        """Test that errors are properly captured and formatted."""
        result = python_env("x = undefined_variable")

        assert not result.success
        assert result.error is not None
        assert "NameError" in result.error
        assert "undefined_variable" in result.error


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestProgramOfThoughtWithModel:
    """Tests that require a real LLM model."""

    @pytest.fixture
    def agent(self):
        """Create an agent with ProgramOfThought."""
        from msgflux import Model
        from msgflux.nn import Agent, LM

        model = Model.chat_completion("openai/gpt-4.1-mini")
        lm = LM(model=model)

        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

        agent = Agent(
            name="solver",
            model=lm,
            environment=env,
            generation_schema=ProgramOfThought,
        )

        yield agent
        env.shutdown()

    def test_simple_calculation(self, agent):
        """Test a simple calculation with the agent."""
        result = agent("What is 7 * 8?")
        assert result.get("final_answer") is not None
        assert "56" in result.get("final_answer", "")

    def test_with_tool_injection(self, agent):
        """Test agent with tools available in sandbox."""
        result = agent("Calculate the factorial of 6")
        assert result.get("final_answer") is not None
        assert "720" in result.get("final_answer", "")


class TestProgramOfThoughtBuildHistory:
    """Tests for history building in ProgramOfThought."""

    def test_build_history_first_step(self):
        """Test building history for first step."""
        raw_response = {
            "current_step": {"thought": "First", "code": "print(1)", "result": "1"},
            "final_answer": None,
        }

        messages = []
        result = ProgramOfThought.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_build_history_append(self):
        """Test appending to existing history."""
        import msgspec

        first_response = {
            "current_step": {"thought": "First", "code": "x = 1", "result": ""},
            "final_answer": None,
        }
        second_response = {
            "current_step": {"thought": "Second", "code": "print(x)", "result": "1"},
            "final_answer": None,
        }

        messages = []
        ProgramOfThought.build_history(first_response, messages)
        ProgramOfThought.build_history(second_response, messages)

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
        print("  OPENAI_API_KEY=sk-... python tests/integration/test_pot_sandbox.py")
        return

    print("=" * 60)
    print("Manual Integration Test: ProgramOfThought + Sandbox")
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

    # Create agent with ProgramOfThought
    agent = Agent(
        name="solver",
        model=lm,
        environment=env,
        generation_schema=ProgramOfThought,
    )

    # Test problem
    problem = "What is the sum of the first 10 prime numbers?"

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
