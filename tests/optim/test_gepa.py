"""Tests for the GEPA optimizer."""

from unittest.mock import Mock

import pytest

pytest.importorskip("gepa")

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, Module
from msgflux.optim import GEPA
from msgflux.optim.gepa.gepa import (
    _apply_tool_descriptions,
    _collect_tool_descriptions,
)


def _make_agent(*, name: str = "qa_agent", response_text: str = "Wrong") -> Agent:
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    response = ModelResponse()
    response.data = response_text
    response.response_type = "text_generation"

    mock_model.return_value = response

    agent = Agent(
        name=name,
        model=mock_model,
        system_message="You are a geography expert.",
        instructions="Answer with just the city name.",
    )
    agent.generator.forward = Mock(return_value=response)
    return agent


def _feedback_metric(
    gold: Example,
    pred,
    trace=None,
    pred_name: str | None = None,
    pred_trace=None,
):
    score = 0.0
    if pred is not None and str(gold.labels).lower() in str(pred).lower():
        score = 1.0
    return {"score": score, "feedback": "Return only the exact city name."}


def test_metric_requires_feedback_signature():
    def bad_metric(example, prediction):
        return 0.0

    with pytest.raises(TypeError, match="five arguments"):
        GEPA(
            metric=bad_metric,
            instruction_proposer=lambda **_: {},
            max_metric_calls=1,
        )


def test_compile_with_instruction_proposer_uses_agent_name_for_component_feedback():
    student = _make_agent()
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]
    metric_calls = []
    proposer_calls = []

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        metric_calls.append(
            {
                "gold": gold,
                "pred": pred,
                "trace": trace,
                "pred_name": pred_name,
                "pred_trace": pred_trace,
            }
        )
        return {"score": 0.0, "feedback": "Return only the exact city name."}

    def instruction_proposer(candidate, reflective_dataset, components_to_update):
        proposer_calls.append(
            {
                "candidate": candidate,
                "reflective_dataset": reflective_dataset,
                "components_to_update": components_to_update,
            }
        )
        return {
            name: candidate[name] + "\nReturn only the exact city name."
            for name in components_to_update
        }

    optimizer = GEPA(
        metric=metric,
        instruction_proposer=instruction_proposer,
        max_metric_calls=4,
        track_stats=True,
        skip_perfect_score=False,
    )

    compiled = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert compiled.compiled is True
    assert compiled.get_compile_info()["optimizer"] == "GEPA"
    assert hasattr(compiled, "detailed_results")
    assert proposer_calls
    assert proposer_calls[0]["components_to_update"] == ["qa_agent"]
    assert list(proposer_calls[0]["candidate"]) == ["qa_agent"]
    assert (
        proposer_calls[0]["reflective_dataset"]["qa_agent"][0]["Generated Outputs"]
        == "Wrong"
    )

    component_calls = [call for call in metric_calls if call["pred_name"] == "qa_agent"]
    assert component_calls
    assert component_calls[0]["trace"] is not None
    assert component_calls[0]["pred_trace"] is not None
    assert component_calls[0]["pred_trace"][0][0] == "qa_agent"


def test_compile_with_reflection_lm_uses_gepa_instruction_signature():
    student = _make_agent()
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]
    prompts = []

    def reflection_lm(prompt):
        prompts.append(prompt)
        return "```\nAlways answer with only the city name.\n```"

    optimizer = GEPA(
        metric=_feedback_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=4,
        skip_perfect_score=False,
    )

    compiled = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert compiled.compiled is True
    assert prompts
    assert "You are a geography expert." in prompts[0]
    assert "Return only the exact city name." in prompts[0]


def test_compile_with_model_response_reflection_lm():
    student = _make_agent()
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]
    prompts = []

    def reflection_lm(prompt):
        prompts.append(prompt)
        response = ModelResponse()
        response.data = "```\nAlways answer with only the city name.\n```"
        response.response_type = "text_generation"
        return response

    optimizer = GEPA(
        metric=_feedback_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=4,
        skip_perfect_score=False,
    )

    compiled = optimizer.compile(student, trainset=trainset, valset=trainset)

    assert compiled.compiled is True
    assert prompts
    assert "You are a geography expert." in prompts[0]
    assert "Return only the exact city name." in prompts[0]


def test_compile_requires_unique_agent_names():
    class DuplicateNamesProgram(Module):
        def __init__(self):
            super().__init__()
            self.first = _make_agent(name="shared_agent")
            self.second = _make_agent(name="shared_agent")

        def forward(self, message=None, **kwargs):
            return self.first(message, **kwargs)

    optimizer = GEPA(
        metric=_feedback_metric,
        instruction_proposer=lambda **_: {},
        max_metric_calls=1,
    )

    with pytest.raises(ValueError, match="unique Agent names"):
        optimizer.compile(
            DuplicateNamesProgram(),
            trainset=[Example(inputs="What is the capital of France?", labels="Paris")],
            valset=[Example(inputs="What is the capital of France?", labels="Paris")],
        )


# ---------------------------------------------------------------------------
# optimize_tools helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str, description: str):
    """Create a minimal callable with tool metadata for Agent tools param."""

    def tool_fn(**kwargs):
        return f"{name} called"

    tool_fn.__name__ = name
    tool_fn.__doc__ = description
    return tool_fn


def _make_agent_with_tools(
    *,
    name: str = "router_agent",
    response_text: str = "tool_a called",
    tool_names: list[str] | None = None,
    tool_descs: list[str] | None = None,
) -> Agent:
    """Create an Agent with mock model and real tools."""
    tool_names = tool_names or ["tool_a", "tool_b"]
    tool_descs = tool_descs or ["Does action A", "Does action B"]

    tools = [_make_tool(n, d) for n, d in zip(tool_names, tool_descs)]

    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    response = ModelResponse()
    response.data = response_text
    response.response_type = "text_generation"
    mock_model.return_value = response

    agent = Agent(
        name=name,
        model=mock_model,
        system_message="You are a helpful assistant.",
        instructions="Use the right tool for the task.",
        tools=tools,
    )
    agent.generator.forward = Mock(return_value=response)
    return agent


class TestCollectToolDescriptions:
    """Tests for _collect_tool_descriptions helper."""

    def test_collects_from_single_agent(self):
        agent = _make_agent_with_tools(
            tool_names=["search", "calculate"],
            tool_descs=["Search the web", "Do math"],
        )
        descs = _collect_tool_descriptions(agent)

        assert "router_agent::search" in descs
        assert "router_agent::calculate" in descs
        assert descs["router_agent::search"] == "Search the web"
        assert descs["router_agent::calculate"] == "Do math"

    def test_returns_empty_for_agent_without_tools(self):
        agent = _make_agent(name="no_tools_agent")
        descs = _collect_tool_descriptions(agent)
        assert descs == {}

    def test_collects_from_multi_agent_program(self):
        class MultiAgent(Module):
            def __init__(self):
                super().__init__()
                self.first = _make_agent_with_tools(
                    name="first", tool_names=["t1"], tool_descs=["First tool"]
                )
                self.second = _make_agent_with_tools(
                    name="second", tool_names=["t2"], tool_descs=["Second tool"]
                )

            def forward(self, message=None, **kwargs):
                return self.first(message, **kwargs)

        program = MultiAgent()
        descs = _collect_tool_descriptions(program)

        assert len(descs) == 2
        assert "first::t1" in descs
        assert "second::t2" in descs


class TestApplyToolDescriptions:
    """Tests for _apply_tool_descriptions helper."""

    def test_applies_new_descriptions(self):
        agent = _make_agent_with_tools(
            tool_names=["search", "calculate"],
            tool_descs=["old search desc", "old calc desc"],
        )

        _apply_tool_descriptions(agent, {
            "router_agent::search": "NEW search description",
            "router_agent::calculate": "NEW calc description",
        })

        descs = _collect_tool_descriptions(agent)
        assert descs["router_agent::search"] == "NEW search description"
        assert descs["router_agent::calculate"] == "NEW calc description"

    def test_ignores_unknown_keys(self):
        agent = _make_agent_with_tools(
            tool_names=["search"], tool_descs=["original"]
        )

        _apply_tool_descriptions(agent, {
            "router_agent::search": "updated",
            "router_agent::nonexistent": "should be ignored",
        })

        descs = _collect_tool_descriptions(agent)
        assert descs["router_agent::search"] == "updated"
        assert "router_agent::nonexistent" not in descs


class TestOptimizeToolsIntegration:
    """Integration test: GEPA compile with optimize_tools=True."""

    def test_compile_optimize_tools_flag(self):
        """Verify optimize_tools runs optimize_anything post-compile."""
        agent = _make_agent_with_tools(
            tool_names=["search", "calculate"],
            tool_descs=["Search the web", "Do math"],
        )
        trainset = [
            Example(inputs="Find info about Python", labels="search called"),
        ]

        # Track optimize_anything calls
        oa_calls = []

        def mock_optimize_anything(
            seed_candidate,
            evaluator,
            dataset,
            valset,
            objective,
            config,
        ):
            oa_calls.append({
                "seed": seed_candidate,
                "objective": objective,
            })

            # Return a mock result with improved descriptions
            result = Mock()
            result.best_candidate = {
                k: f"OPTIMIZED: {v}" for k, v in seed_candidate.items()
            }
            result.best_score = 1.0
            return result

        import msgflux.optim.gepa.gepa as gepa_module

        original_import = gepa_module._import_gepa

        optimizer = GEPA(
            metric=_feedback_metric,
            instruction_proposer=lambda **_: {},
            max_metric_calls=2,
            optimize_tools=True,
        )

        # Monkeypatch optimize_anything
        import unittest.mock

        with unittest.mock.patch(
            "msgflux.optim.gepa.gepa.optimize_anything",
            mock_optimize_anything,
            create=True,
        ), unittest.mock.patch.dict(
            "sys.modules",
            {
                "gepa.optimize_anything": Mock(
                    optimize_anything=mock_optimize_anything,
                    EngineConfig=Mock(),
                    GEPAConfig=Mock(return_value=Mock()),
                    ReflectionConfig=Mock(),
                    log=Mock(),
                ),
            },
        ):
            compiled = optimizer.compile(
                agent, trainset=trainset, valset=trainset,
            )

        assert compiled.compiled is True
        assert len(oa_calls) == 1
        assert "router_agent::search" in oa_calls[0]["seed"]
        assert "router_agent::calculate" in oa_calls[0]["seed"]
        assert "tool" in oa_calls[0]["objective"].lower()

        # Verify optimized descriptions were applied
        descs = _collect_tool_descriptions(compiled)
        assert descs["router_agent::search"].startswith("OPTIMIZED:")
        assert descs["router_agent::calculate"].startswith("OPTIMIZED:")
