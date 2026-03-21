"""Tests for the GEPA optimizer."""

from unittest.mock import Mock

import pytest

pytest.importorskip("gepa")

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, Module
from msgflux.optim import GEPA


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
