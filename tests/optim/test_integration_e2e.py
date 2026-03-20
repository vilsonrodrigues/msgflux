"""Integration tests for optimizers using real OpenAI API calls.

Requires OPENAI_API_KEY in .env file.
Run with: uv run pytest tests/optim/test_integration_e2e.py -v -s
"""

import pytest

from msgflux import Example, load_dotenv
from msgflux.context import trace_context
from msgflux.models import Model
from msgflux.nn import Agent, Module, Sequential
from msgflux.optim import (
    COPRO,
    GEPA,
    SIMBA,
    BootstrapFewShot,
    Evaluate,
    EvalResult,
    LabeledFewShot,
    MIPROv2,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "openai/gpt-4.1-mini"


def exact_match(example: Example, prediction) -> float:
    """Simple metric: 1.0 if prediction contains the expected label."""
    if prediction is None:
        return 0.0
    pred_str = str(prediction).strip().lower()
    label_str = str(example.labels).strip().lower()
    return 1.0 if label_str in pred_str else 0.0


def make_trainset():
    """Small QA trainset for testing."""
    return [
        Example(inputs="What is the capital of France?", labels="Paris"),
        Example(inputs="What is the capital of Germany?", labels="Berlin"),
        Example(inputs="What is the capital of Italy?", labels="Rome"),
        Example(inputs="What is the capital of Spain?", labels="Madrid"),
        Example(inputs="What is the capital of Portugal?", labels="Lisbon"),
        Example(inputs="What is the capital of Japan?", labels="Tokyo"),
        Example(inputs="What is the capital of Brazil?", labels="Brasilia"),
        Example(inputs="What is the capital of Argentina?", labels="Buenos Aires"),
    ]


def make_model():
    return Model.chat_completion(MODEL_ID, max_tokens=50, temperature=0.0)


def make_student():
    return Agent(
        name="qa_agent",
        model=make_model(),
        system_message="You are a geography expert.",
        instructions="Answer the question with just the city name.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTraceContextE2E:
    """Test that trace_context captures Agent I/O with real API calls."""

    def test_trace_captures_agent_turn(self):
        agent = make_student()

        with trace_context() as trace:
            result = agent("What is the capital of France?")

        assert len(trace) == 1
        agent_ref, turn_record = trace[0]
        assert agent_ref is agent
        assert turn_record is not None
        assert "Paris" in str(result)
        print(f"\n[trace] Agent output: {result}")
        print(f"[trace] Turn record keys: {list(turn_record.keys())}")


class TestEvaluateE2E:
    """Test Evaluate with real API calls."""

    def test_evaluate_on_small_set(self):
        student = make_student()
        trainset = make_trainset()[:3]

        evaluator = Evaluate(devset=trainset, metric=exact_match)
        result = evaluator(student)

        assert isinstance(result, EvalResult)
        assert 0.0 <= result.score <= 1.0
        print(f"\n[evaluate] Score on 3 examples: {result.score:.2f}")


class TestLabeledFewShotE2E:
    """Test LabeledFewShot with real API calls."""

    def test_compile_and_evaluate(self):
        student = make_student()
        trainset = make_trainset()

        optimizer = LabeledFewShot(k=4)
        compiled = optimizer.compile(student, trainset=trainset)

        # Verify demos assigned
        agents = list(compiled.named_agents())
        assert len(agents) == 1
        _, agent = agents[0]
        assert len(agent.optimized_examples) == 4
        assert compiled._compiled

        # Evaluate compiled program
        evaluator = Evaluate(devset=trainset[:3], metric=exact_match)
        result = evaluator(compiled)

        assert 0.0 <= result.score <= 1.0
        print(f"\n[LabeledFewShot] Demos: {len(agent.optimized_examples)}")
        print(f"[LabeledFewShot] Score: {result.score:.2f}")


class TestBootstrapFewShotE2E:
    """Test BootstrapFewShot with real API calls."""

    def test_compile_and_evaluate(self):
        student = make_student()
        trainset = make_trainset()[:4]

        optimizer = BootstrapFewShot(
            metric=exact_match,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            max_rounds=1,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        assert compiled._compiled

        # Evaluate
        evaluator = Evaluate(devset=trainset, metric=exact_match)
        result = evaluator(compiled)

        agents = list(compiled.named_agents())
        _, agent = agents[0]
        print(f"\n[Bootstrap] Demos: {len(agent.optimized_examples)}")
        print(f"[Bootstrap] Score: {result.score:.2f}")


class TestCOPROE2E:
    """Test COPRO with real API calls."""

    def test_compile_and_evaluate(self):
        student = make_student()
        trainset = make_trainset()[:4]
        prompt_model = make_model()

        # Use a simple callable wrapper for prompt_model
        def prompt_fn(prompt_text):
            temp_agent = Agent(
                name="prompt_generator",
                model=prompt_model,
                system_message="You generate system prompts.",
            )
            return temp_agent(prompt_text)

        optimizer = COPRO(
            metric=exact_match,
            prompt_model=prompt_fn,
            breadth=3,
            depth=2,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        assert compiled._compiled

        # Check optimized_system_prompt was set
        agents = list(compiled.named_agents())
        _, agent = agents[0]
        print(f"\n[COPRO] Optimized prompt: {agent.optimized_system_prompt.data[:100]}...")

        # Evaluate
        evaluator = Evaluate(devset=trainset, metric=exact_match)
        result = evaluator(compiled)
        print(f"[COPRO] Score: {result.score:.2f}")


class TestSIMBAE2E:
    """Test SIMBA with real API calls."""

    def test_compile_and_evaluate(self):
        student = make_student()
        # SIMBA needs trainset >= bsize, use small values
        trainset = make_trainset()

        optimizer = SIMBA(
            metric=exact_match,
            bsize=4,
            num_candidates=2,
            max_steps=2,
            max_demos=2,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        assert compiled._compiled

        # Evaluate
        evaluator = Evaluate(devset=trainset[:4], metric=exact_match)
        result = evaluator(compiled)

        agents = list(compiled.named_agents())
        _, agent = agents[0]
        print(f"\n[SIMBA] Demos: {len(agent.optimized_examples)}")
        if agent.optimized_system_prompt.data:
            print(f"[SIMBA] Optimized prompt: {agent.optimized_system_prompt.data[:100]}...")
        print(f"[SIMBA] Score: {result.score:.2f}")


class TestMIPROv2E2E:
    """Test MIPROv2 with real API calls + Optuna."""

    def test_compile_and_evaluate(self):
        student = make_student()
        trainset = make_trainset()
        prompt_model = make_model()

        def prompt_fn(prompt_text):
            temp_agent = Agent(
                name="proposer",
                model=prompt_model,
                system_message="You generate system prompts.",
            )
            return temp_agent(prompt_text)

        optimizer = MIPROv2(
            metric=exact_match,
            prompt_model=prompt_fn,
            auto="light",
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        assert compiled._compiled

        agents = list(compiled.named_agents())
        _, agent = agents[0]
        if agent.optimized_system_prompt.data:
            print(f"\n[MIPROv2] Optimized prompt: {agent.optimized_system_prompt.data[:100]}...")
        print(f"[MIPROv2] Demos: {len(agent.optimized_examples)}")

        evaluator = Evaluate(devset=trainset[:4], metric=exact_match)
        result = evaluator(compiled)
        print(f"[MIPROv2] Score: {result.score:.2f}")


class TestGEPAE2E:
    """Test GEPA with real API calls (built-in loop, no gepa package)."""

    def test_compile_and_evaluate(self):
        student = make_student()
        trainset = make_trainset()[:4]
        prompt_model = make_model()

        def reflection_fn(prompt_text):
            temp_agent = Agent(
                name="reflector",
                model=prompt_model,
                system_message="You optimize system prompts.",
            )
            return temp_agent(prompt_text)

        optimizer = GEPA(
            metric=exact_match,
            reflection_lm=reflection_fn,
            max_iterations=2,
            num_candidates=2,
        )
        compiled = optimizer.compile(student, trainset=trainset)

        assert compiled._compiled

        agents = list(compiled.named_agents())
        _, agent = agents[0]
        if agent.optimized_system_prompt.data:
            print(f"\n[GEPA] Optimized prompt: {agent.optimized_system_prompt.data[:100]}...")

        evaluator = Evaluate(devset=trainset, metric=exact_match)
        result = evaluator(compiled)
        print(f"[GEPA] Score: {result.score:.2f}")


class TestMultiAgentPipeline:
    """Test optimizers on a multi-agent pipeline (Sequential)."""

    def test_labeled_fewshot_multi_agent(self):
        model = make_model()

        class QAPipeline(Module):
            def __init__(self):
                super().__init__()
                self.thinker = Agent(
                    name="thinker",
                    model=model,
                    system_message="You reason about geography questions.",
                    instructions="Think step by step about the answer. Output your reasoning.",
                )
                self.answerer = Agent(
                    name="answerer",
                    model=model,
                    system_message="You answer geography questions concisely.",
                    instructions="Given the reasoning, provide just the city name.",
                )

            def forward(self, message=None, **kwargs):
                reasoning = self.thinker(message)
                answer = self.answerer(
                    f"Question: {message}\nReasoning: {reasoning}\nAnswer with just the city name:"
                )
                return answer

        student = QAPipeline()
        trainset = make_trainset()[:4]

        optimizer = LabeledFewShot(k=2)
        compiled = optimizer.compile(student, trainset=trainset)

        # Verify both agents got demos
        agents = list(compiled.named_agents())
        assert len(agents) == 2
        for name, agent in agents:
            assert len(agent.optimized_examples) == 2
            print(f"\n[MultiAgent] {name}: {len(agent.optimized_examples)} demos")

        # Evaluate
        evaluator = Evaluate(devset=trainset[:2], metric=exact_match)
        result = evaluator(compiled)
        print(f"[MultiAgent] Score: {result.score:.2f}")
