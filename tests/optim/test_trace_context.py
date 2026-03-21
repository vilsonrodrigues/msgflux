"""Tests for trace context and Module optimizer helpers."""

import pytest
from unittest.mock import Mock

from msgflux.context import trace_context, get_trace
from msgflux.examples import Example
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module


@pytest.fixture
def mock_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


class TestTraceContext:
    def test_trace_context_yields_empty_list(self):
        with trace_context() as trace:
            assert trace == []

    def test_get_trace_returns_none_outside_context(self):
        assert get_trace() is None

    def test_get_trace_returns_list_inside_context(self):
        with trace_context() as trace:
            assert get_trace() is trace

    def test_nested_trace_contexts_are_isolated(self):
        with trace_context() as outer:
            outer.append("outer_item")
            with trace_context() as inner:
                inner.append("inner_item")
                assert get_trace() is inner
                assert len(inner) == 1
            assert get_trace() is outer
            assert len(outer) == 1

    def test_trace_context_restores_none_after_exit(self):
        with trace_context():
            assert get_trace() is not None
        assert get_trace() is None


class TestModuleHelpers:
    def test_deepcopy(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        clone = agent.deepcopy()

        assert clone is not agent
        assert clone.get_module_name() == agent.get_module_name()

    def test_reset_copy_clears_state(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        agent.optimized_examples = [Example(inputs="q", labels="a")]
        agent.optimized_system_prompt.data = "optimized"
        agent.compile_(optimizer="test")

        reset = agent.reset_copy()

        assert reset.optimized_examples == []
        assert reset.optimized_system_prompt.data is None
        assert reset._compiled is False

    def test_agents_returns_agent_list(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        agents = agent.agents()
        assert len(agents) >= 1
        assert agent.__class__.__name__ == "Agent"

    def test_named_agents(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        named = list(agent.named_agents())
        assert len(named) >= 1
        name, mod = named[0]
        assert isinstance(name, str)


class TestOptimizedSystemPrompt:
    def test_default_is_none(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        assert agent.optimized_system_prompt.data is None

    def test_get_optimized_prompts_for_root_agent(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        agent.optimized_system_prompt.data = "You are a concise helper."

        assert agent.get_optimized_prompts() == {"qa": "You are a concise helper."}

    def test_get_optimized_prompts_for_composed_module(self, mock_model):
        class Pipeline(Module):
            def __init__(self):
                super().__init__()
                self.classifier = Agent(name="classifier", model=mock_model)
                self.reviewer = Agent(name="reviewer", model=mock_model)

        program = Pipeline()
        program.classifier.optimized_system_prompt.data = "Classify precisely."
        program.reviewer.optimized_system_prompt.data = "Review carefully."

        assert program.get_optimized_prompts() == {
            "classifier": "Classify precisely.",
            "reviewer": "Review carefully.",
        }

    def test_get_optimized_prompts_uses_paths_for_duplicate_agent_names(
        self, mock_model
    ):
        class Pipeline(Module):
            def __init__(self):
                super().__init__()
                self.left = Agent(name="shared", model=mock_model)
                self.right = Agent(name="shared", model=mock_model)

        program = Pipeline()
        program.left.optimized_system_prompt.data = "Left prompt."
        program.right.optimized_system_prompt.data = "Right prompt."

        assert program.get_optimized_prompts() == {
            "left": "Left prompt.",
            "right": "Right prompt.",
        }

    def test_optimized_prompt_takes_precedence(self, mock_model):
        agent = Agent(
            name="qa",
            model=mock_model,
            system_message="I am a helper",
            instructions="Be concise",
        )
        agent.optimized_system_prompt.data = "You are a concise helper."

        prompt = agent.get_system_prompt()
        assert "You are a concise helper." in prompt
        # fragments should NOT appear
        assert "I am a helper" not in prompt

    def test_expected_output_immune_to_optimization(self, mock_model):
        """expected_output must always render, even with optimized_system_prompt."""
        agent = Agent(
            name="qa",
            model=mock_model,
            system_message="I am a helper",
            instructions="Be concise",
            expected_output="Return JSON.",
        )
        agent.optimized_system_prompt.data = "You are a concise helper."

        prompt = agent.get_system_prompt()
        # optimized_system_prompt replaces system_message + instructions
        assert "You are a concise helper." in prompt
        assert "I am a helper" not in prompt
        assert "Be concise" not in prompt
        # expected_output is immune — always present
        assert "Return JSON." in prompt

    def test_fragments_used_when_no_optimized(self, mock_model):
        agent = Agent(
            name="qa",
            model=mock_model,
            system_message="I am a helper",
        )
        prompt = agent.get_system_prompt()
        assert "I am a helper" in prompt


class TestOptimizedExamples:
    def test_default_empty(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        assert agent.optimized_examples == []

    def test_optimized_examples_appear_in_system_prompt(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        agent.optimized_examples = [
            Example(inputs="2+2", labels="4"),
        ]
        prompt = agent.get_system_prompt()
        assert "2+2" in prompt
        assert "4" in prompt

    def test_optimized_examples_with_optimized_prompt(self, mock_model):
        agent = Agent(name="qa", model=mock_model)
        agent.optimized_system_prompt.data = "You are a calculator."
        agent.optimized_examples = [Example(inputs="1+1", labels="2")]

        prompt = agent.get_system_prompt()
        assert "You are a calculator." in prompt
        assert "1+1" in prompt

    def test_optimized_examples_override_dev_examples(self, mock_model):
        """optimized_examples should replace developer-provided examples."""
        agent = Agent(
            name="qa",
            model=mock_model,
            examples="Q: old_q\nA: old_a",
        )
        agent.optimized_examples = [Example(inputs="new_q", labels="new_a")]

        prompt = agent.get_system_prompt()
        # optimized_examples take precedence
        assert "new_q" in prompt
        # dev examples should NOT appear
        assert "old_q" not in prompt
