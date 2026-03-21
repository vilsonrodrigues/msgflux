"""Tests for BootstrapFewShot parity-sensitive behavior."""

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.optim.bootstrap import BootstrapFewShot


class StaticResponseModel:
    model_type = "chat_completion"

    def __init__(self, text: str):
        self.text = text
        self.sampling_run_params: dict[str, float] = {}

    def __call__(self, **kwargs):
        _ = kwargs
        response = ModelResponse()
        response.data = self.text
        response.response_type = "text_generation"
        return response


class TemperatureSensitiveModel:
    model_type = "chat_completion"

    def __init__(self):
        self.sampling_run_params: dict[str, float] = {}

    def __call__(self, **kwargs):
        _ = kwargs
        response = ModelResponse()
        temperature = self.sampling_run_params.get("temperature", 0.0)
        response.data = "Paris" if temperature == 1.0 else "Wrong"
        response.response_type = "text_generation"
        return response


def _make_agent(model, *, name: str = "qa") -> Agent:
    return Agent(
        name=name,
        model=model,
        system_message="You are a geography expert.",
        instructions="Answer with just the city name.",
    )


def test_bootstrap_uses_trace_aware_metric_and_residual_validation_backfill():
    student = _make_agent(StaticResponseModel("bootstrapped"))
    trainset = [
        Example(inputs="q1", labels="gold1"),
        Example(inputs="q2", labels="gold2"),
        Example(inputs="q3", labels="gold3"),
        Example(inputs="q4", labels="gold4"),
    ]
    seen_traces = []

    def metric(example, prediction, trace=None):
        seen_traces.append(trace)
        _ = prediction
        return example.inputs in {"q1", "q2"}

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=3,
        max_rounds=1,
    )

    compiled = optimizer.compile(student, trainset=trainset)

    assert compiled._compiled is True
    assert seen_traces
    assert all(trace is not None for trace in seen_traces[:2])
    assert len(compiled.optimized_examples) == 3
    assert [demo.labels for demo in compiled.optimized_examples[:2]] == [
        "bootstrapped",
        "bootstrapped",
    ]
    assert compiled.optimized_examples[2] in trainset[2:]


def test_bootstrap_uses_second_round_with_temperature_override():
    student = _make_agent(TemperatureSensitiveModel())
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]

    def metric(example, prediction):
        return float(str(prediction).strip().lower() == str(example.labels).lower())

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=1,
        max_labeled_demos=0,
        max_rounds=2,
        teacher_settings={"temperature": 0.0},
    )

    compiled = optimizer.compile(student, trainset=trainset)

    assert len(compiled.optimized_examples) == 1
    assert compiled.optimized_examples[0].labels == "Paris"
