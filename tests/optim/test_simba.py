"""Tests for SIMBA parity-sensitive behavior."""

import random

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.optim.simba.simba import SIMBA, _CompileState, _wrap_program


class StaticResponseModel:
    model_type = "chat_completion"

    def __init__(self, text: str):
        self.text = text
        self.sampling_run_params: dict[str, float] = {}

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        response = ModelResponse()
        response.data = self.text
        response.response_type = "text_generation"
        return response


class AdviceFallbackModel:
    model_type = "chat_completion"

    def __init__(self, seen_prompts: list[str] | None = None):
        self.seen_prompts = [] if seen_prompts is None else seen_prompts
        self.sampling_run_params: dict[str, float] = {}

    def __deepcopy__(self, memo):
        _ = memo
        clone = type(self)(self.seen_prompts)
        clone.sampling_run_params = dict(self.sampling_run_params)
        return clone

    def __call__(self, *args, **kwargs):
        _ = kwargs
        response = ModelResponse()
        if args:
            self.seen_prompts.append(str(args[0]))
            response.data = '{"qa": "Prefer the better city-level answer."}'
        else:
            response.data = "Paris"
        response.response_type = "text_generation"
        return response


class TemperatureModel:
    model_type = "chat_completion"

    def __init__(self):
        self.sampling_run_params: dict[str, float] = {}

    def __deepcopy__(self, memo):
        _ = memo
        clone = type(self)()
        clone.sampling_run_params = dict(self.sampling_run_params)
        return clone

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        response = ModelResponse()
        response.data = "Paris"
        response.response_type = "text_generation"
        return response


def _make_agent(model, *, name: str = "qa") -> Agent:
    return Agent(
        name=name,
        model=model,
        system_message="You are a geography expert.",
        instructions="Answer with just the city name.",
    )


def exact_match(example, prediction):
    return float(str(prediction).strip().lower() == str(example.labels).lower())


def test_simba_append_rule_uses_agent_model_when_prompt_model_missing():
    seen_prompts: list[str] = []
    program = _make_agent(AdviceFallbackModel(seen_prompts))
    example = Example(inputs="What is the capital of France?", labels="Paris")
    bucket = [
        {
            "score": 1.0,
            "trace": [(program, {"inputs": example.inputs, "assistant_output": "Paris"})],
            "prediction": "Paris",
            "example": example,
            "output_metadata": {"feedback": "correct"},
        },
        {
            "score": 0.0,
            "trace": [(program, {"inputs": example.inputs, "assistant_output": "Rome"})],
            "prediction": "Rome",
            "example": example,
            "output_metadata": {"feedback": "incorrect"},
        },
    ]

    optimizer = SIMBA(metric=exact_match, bsize=1, num_candidates=1, max_steps=1)
    applied = optimizer._append_a_rule(
        bucket=bucket,
        system=program,
        agent2name={id(program): "qa"},
        name2agent={"qa": program},
        batch_10p_score=0.1,
        batch_90p_score=0.9,
    )

    assert applied is True
    assert any("Worse reward metadata:" in prompt for prompt in seen_prompts)
    assert any("Better reward metadata:" in prompt for prompt in seen_prompts)
    assert "Prefer the better city-level answer." in program.optimized_system_prompt.data


def test_simba_prepare_models_for_resampling_uses_teacher_and_temperature():
    base_model = TemperatureModel()
    teacher_model = TemperatureModel()
    program = _make_agent(base_model)
    optimizer = SIMBA(
        metric=exact_match,
        bsize=1,
        num_candidates=3,
        max_steps=1,
        teacher_settings={"model": teacher_model, "temperature": 0.4},
    )

    models = optimizer._prepare_models_for_resampling(program, 3)

    assert len(models) == 3
    assert models[0].sampling_run_params["temperature"] == 0.4
    assert models[1].sampling_run_params["temperature"] == 1.0
    assert models[2].sampling_run_params["temperature"] == 1.0


def test_wrap_program_preserves_metric_metadata():
    program = _make_agent(StaticResponseModel("Paris"))
    example = Example(inputs="What is the capital of France?", labels="Paris")

    result = _wrap_program(
        program,
        example,
        lambda ex, pred: {"score": exact_match(ex, pred), "feedback": "ok"},
    )

    assert result["score"] == 1.0
    assert result["output_metadata"] == {"feedback": "ok"}


def test_simba_final_validation_attaches_candidate_programs_and_logs():
    optimizer = SIMBA(metric=exact_match, bsize=1, num_candidates=1, max_steps=1)
    program = _make_agent(StaticResponseModel("Paris"))
    state = _CompileState(random.Random(0))  # noqa: S311
    state.winning_programs = [program]
    state.total_calls = 5
    state.trial_logs = {0: {"candidate_scores": [1.0]}}
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]

    compiled = optimizer._final_validation(state, trainset)

    assert compiled._compiled is True
    assert len(compiled.candidate_programs) == 1
    assert compiled.trial_logs == {0: {"candidate_scores": [1.0]}}
    assert compiled.total_calls == 6
    assert compiled.get_compile_info()["total_calls"] == 6
