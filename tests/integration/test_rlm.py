"""Integration tests for RLM (Recursive Language Model) with sandbox.

These tests verify the integration between:
- nn.Agent with generation_schema=RLM
- nn.Environment wrapping DenoPyodideSandbox
- make_rlm_tools for sub-LLM queries

To run with a real model:
    OPENAI_API_KEY=... pytest tests/integration/test_rlm.py -v

To run the manual test:
    python tests/integration/test_rlm.py
"""

import os

import msgspec
import pytest

from msgflux.environments import Environments
from msgflux.generation.reasoning.rlm import (
    DEFAULT_QUERY_LLM,
    RLM,
    RLMStep,
    make_rlm_tools,
)
from msgflux.nn.modules.environment import Environment


class TestRLMFlowControl:
    """Tests for RLM FlowControl interface."""

    def test_extract_flow_result_with_final_answer(self):
        """Test that extract_flow_result completes on final_answer."""
        raw_response = {
            "current_step": None,
            "final_answer": "The answer is 42.",
        }

        flow_result = RLM.extract_flow_result(raw_response, {})

        assert flow_result.is_complete
        assert flow_result.final_response == raw_response
        assert flow_result.environment_call is None

    def test_extract_flow_result_with_code(self):
        """Test that extract_flow_result creates environment_call for code."""
        raw_response = {
            "current_step": {
                "reasoning": "I need to explore the data first",
                "code": "print(len(context))",
            },
            "final_answer": None,
        }

        flow_result = RLM.extract_flow_result(raw_response, {})

        assert not flow_result.is_complete
        assert flow_result.environment_call is not None
        assert flow_result.environment_call.action == "print(len(context))"
        assert flow_result.environment_call.inject_vars is True
        assert flow_result.environment_call.inject_tools is True
        assert flow_result.reasoning == "I need to explore the data first"

    def test_extract_flow_result_empty_state(self):
        """Test extract_flow_result with no step and no final_answer."""
        raw_response = {"current_step": None, "final_answer": None}
        flow_result = RLM.extract_flow_result(raw_response, {})

        assert flow_result.is_complete
        assert flow_result.final_response is raw_response

    def test_inject_environment_result_success(self):
        """Test injecting successful execution result."""
        raw_response = {
            "current_step": {
                "reasoning": "Check data length",
                "code": "print(len(context))",
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "1000\n", "error": None, "variables": {}}
        updated = RLM.inject_environment_result(raw_response, result, {})

        assert updated["current_step"]["output"] == "1000"

    def test_inject_environment_result_error(self):
        """Test injecting error result."""
        raw_response = {
            "current_step": {
                "reasoning": "Try to access undefined",
                "code": "print(undefined_var)",
            },
            "final_answer": None,
        }

        result = {
            "success": False,
            "output": "",
            "error": "NameError: name 'undefined_var' is not defined",
            "variables": {},
        }
        updated = RLM.inject_environment_result(raw_response, result, {})

        assert "Error:" in updated["current_step"]["output"]
        assert "NameError" in updated["current_step"]["output"]

    def test_inject_environment_result_no_output(self):
        """Test injecting result with no output."""
        raw_response = {
            "current_step": {
                "reasoning": "Assign variable",
                "code": "x = 42",
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "", "error": None, "variables": {}}
        updated = RLM.inject_environment_result(raw_response, result, {})

        assert updated["current_step"]["output"] == "(no output)"


class TestRLMBuildHistory:
    """Tests for history building in RLM."""

    def test_build_history_first_step(self):
        """Test building history for first step."""
        raw_response = {
            "current_step": {
                "reasoning": "First",
                "code": "print(1)",
                "output": "1",
            },
            "final_answer": None,
        }

        messages = []
        result = RLM.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        content = msgspec.json.decode(result[0]["content"])
        assert len(content) == 1

    def test_build_history_append(self):
        """Test appending to existing history."""
        first_response = {
            "current_step": {
                "reasoning": "First",
                "code": "x = 1",
                "output": "(no output)",
            },
            "final_answer": None,
        }
        second_response = {
            "current_step": {
                "reasoning": "Second",
                "code": "print(x + 1)",
                "output": "2",
            },
            "final_answer": None,
        }

        messages = []
        RLM.build_history(first_response, messages)
        RLM.build_history(second_response, messages)

        # Should have one message with both steps
        assert len(messages) == 1
        content = msgspec.json.decode(messages[0]["content"])
        assert len(content) == 2


class TestRLMStructures:
    """Tests for RLM data structures."""

    def test_rlm_step_structure(self):
        """Test RLMStep structure."""
        step = RLMStep(
            reasoning="I need to analyze the data",
            code="print(data[:100])",
        )

        assert step.reasoning == "I need to analyze the data"
        assert step.code == "print(data[:100])"

    def test_rlm_class_attributes(self):
        """Test RLM class attributes."""
        assert "Python REPL" in RLM.system_message
        assert "EXPLORE FIRST" in RLM.system_message
        assert "environment" in RLM.tools_template.lower()

    def test_rlm_structure(self):
        """Test RLM main structure."""
        rlm = RLM(
            current_step=RLMStep(
                reasoning="Testing",
                code="print(1)",
            ),
            final_answer=None,
        )

        assert rlm.current_step is not None
        assert rlm.current_step.reasoning == "Testing"
        assert rlm.final_answer is None


class TestMakeRLMTools:
    """Tests for make_rlm_tools factory."""

    def test_default_query_llm_config(self):
        """Test DEFAULT_QUERY_LLM has required keys."""
        assert "name" in DEFAULT_QUERY_LLM
        assert "signature" in DEFAULT_QUERY_LLM
        assert "templates" in DEFAULT_QUERY_LLM
        assert DEFAULT_QUERY_LLM["name"] == "query_llm"
        assert "response" in DEFAULT_QUERY_LLM["templates"]


class TestRLMWithEnvironment:
    """Integration tests with sandbox environment."""

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

    def test_environment_executes_code(self, python_env):
        """Test that environment executes RLM-style code."""
        # Simulate RLM code execution
        raw_response = {
            "current_step": {
                "reasoning": "Check the data",
                "code": "data = [1, 2, 3, 4, 5]\nprint(f'Length: {len(data)}')",
            },
            "final_answer": None,
        }

        flow_result = RLM.extract_flow_result(raw_response)
        assert flow_result.environment_call is not None

        # Execute in environment
        exec_result = python_env(flow_result.environment_call.action)

        assert exec_result.success
        assert "Length: 5" in exec_result.output

    def test_environment_with_vars(self, python_env):
        """Test environment with injected variables."""
        # Execute code that uses injected vars
        result = python_env(
            "total = sum(numbers)\nprint(f'Sum: {total}')",
            vars={"numbers": [10, 20, 30]},
        )

        assert result.success
        assert "Sum: 60" in result.output

    def test_environment_with_tools(self, python_env):
        """Test environment with injected tools."""

        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        result = python_env(
            "result = search('Python')\nprint(result)",
            tools={"search": search},
        )

        assert result.success
        assert "Results for: Python" in result.output

    def test_multi_step_with_state_persistence(self, python_env):
        """Test multiple steps with state persistence."""
        # Step 1: Initialize
        result1 = python_env("data = {'count': 0}")
        assert result1.success

        # Step 2: Modify (state persists)
        result2 = python_env("data['count'] += 10\nprint(data)")
        assert result2.success
        assert "'count': 10" in result2.output

        # Step 3: Use accumulated state
        result3 = python_env("data['count'] *= 2\nprint(f'Final: {data[\"count\"]}')")
        assert result3.success
        assert "Final: 20" in result3.output


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestRLMWithModel:
    """Integration tests that require a real LLM model."""

    @pytest.fixture
    def agent(self):
        """Create an agent with RLM and query_llm tools."""
        from msgflux import Model
        from msgflux.nn import Agent, LM

        model = Model.chat_completion("openai/gpt-4.1-mini")
        lm = LM(model=model)

        try:
            code_env = Environments.code("python", timeout=60.0)
            env = Environment(environment=code_env)
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

        # Create sub-LLM tools
        rlm_tools = make_rlm_tools(lm)

        agent = Agent(
            name="analyzer",
            model=lm,
            environment=env,
            tools=[*rlm_tools.values()],
            generation_schema=RLM,
            config={"verbose": True},
        )

        yield agent
        env.shutdown()

    def test_simple_code_execution(self, agent):
        """Test a simple task with code execution."""
        result = agent(
            "Calculate the sum of numbers from 1 to 10 using Python code.",
            vars={"numbers": list(range(1, 11))},
        )

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "")
        # Should mention 55 (sum of 1-10)
        assert "55" in final

    def test_query_llm_for_semantics(self, agent):
        """Test using query_llm for semantic analysis."""
        result = agent(
            "Use the query_llm tool to ask what Python is, then summarize the answer.",
        )

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "").lower()
        # Should mention Python or programming
        assert "python" in final or "programming" in final or "language" in final

    def test_query_llm_with_context(self, agent):
        """Test using query_llm with context from vars."""
        context = """
        The company was founded in 2020. It has 50 employees.
        The main product is a software platform for data analysis.
        Revenue last year was $5 million.
        """

        result = agent(
            "Use query_llm to extract the key facts from the context variable. "
            "Pass the context variable as the second argument to query_llm.",
            vars={"context": context},
        )

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "").lower()
        # Should extract some facts
        has_facts = any(
            x in final for x in ["2020", "50", "employees", "5 million", "data"]
        )
        assert has_facts

    def test_batched_query_llm(self, agent):
        """Test using batched_query_llm for multiple queries."""
        result = agent(
            "Use batched_query_llm to ask these questions in parallel: "
            "['What is Python?', 'What is JavaScript?']. "
            "Then summarize both answers.",
        )

        assert result.get("final_answer") is not None
        final = result.get("final_answer", "").lower()
        # Should mention both languages
        has_python = "python" in final
        has_js = "javascript" in final or "js" in final
        assert has_python or has_js


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
        print("  OPENAI_API_KEY=sk-... python tests/integration/test_rlm.py")
        return

    print("=" * 60)
    print("Manual Integration Test: RLM with Sub-LLM Queries")
    print("=" * 60)

    # Create model
    model = Model.chat_completion("openai/gpt-4.1-mini")
    lm = LM(model=model)

    # Create environment
    try:
        code_env = Environments.code("python", timeout=60.0)
        env = Environment(environment=code_env)
    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("Make sure Deno is installed")
        return

    # Create sub-LLM tools
    rlm_tools = make_rlm_tools(lm)
    print(f"Created tools: {list(rlm_tools.keys())}")

    # Create agent with RLM
    agent = Agent(
        name="researcher",
        model=lm,
        environment=env,
        tools=[*rlm_tools.values()],
        generation_schema=RLM,
        config={"verbose": True},
    )

    # Test problem using query_llm
    problem = (
        "I have some context about a company. Use query_llm to extract the key facts. "
        "The context is in the 'context' variable."
    )
    context = """
    Acme Corp was founded in 2015 by John Smith.
    The company has 200 employees and is headquartered in San Francisco.
    Main products: cloud storage and data analytics platform.
    Annual revenue: $50 million.
    """

    print(f"\nProblem: {problem}")
    print(f"Context: {context[:100]}...")
    print("-" * 60)

    try:
        result = agent(problem, vars={"context": context})
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
