"""Tests for COPRO parity-sensitive behavior."""

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.optim.copro import COPRO


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


class FallbackPromptModel:
    model_type = "chat_completion"

    def __init__(
        self,
        seen_prompts: list[str] | None = None,
        seen_temperatures: list[float | None] | None = None,
    ):
        self.seen_prompts = [] if seen_prompts is None else seen_prompts
        self.seen_temperatures = (
            [] if seen_temperatures is None else seen_temperatures
        )
        self.sampling_run_params: dict[str, float] = {}

    def __deepcopy__(self, memo):
        _ = memo
        clone = type(self)(self.seen_prompts, self.seen_temperatures)
        clone.sampling_run_params = dict(self.sampling_run_params)
        return clone

    def __call__(self, *args, **kwargs):
        response = ModelResponse()
        if args:
            self.seen_prompts.append(str(args[0]))
            self.seen_temperatures.append(
                self.sampling_run_params.get("temperature")
            )
            response.data = "Improved instruction"
        else:
            _ = kwargs
            response.data = "Paris"
        response.response_type = "text_generation"
        return response


class TemperatureTrackingPromptModel:
    model_type = "chat_completion"

    def __init__(self, seen_temperatures: list[float | None] | None = None):
        self.seen_temperatures = (
            [] if seen_temperatures is None else seen_temperatures
        )
        self.sampling_run_params: dict[str, float] = {}

    def __deepcopy__(self, memo):
        _ = memo
        clone = type(self)(self.seen_temperatures)
        clone.sampling_run_params = dict(self.sampling_run_params)
        return clone

    def __call__(self, prompt):
        _ = prompt
        self.seen_temperatures.append(
            self.sampling_run_params.get("temperature")
        )
        response = ModelResponse()
        response.data = "Improved instruction"
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


def test_copro_falls_back_to_agent_model_for_prompt_generation():
    seen_prompts: list[str] = []
    seen_temperatures: list[float | None] = []
    model = FallbackPromptModel(seen_prompts, seen_temperatures)
    student = _make_agent(model)
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]

    optimizer = COPRO(
        metric=exact_match,
        prompt_model=None,
        breadth=2,
        depth=1,
        init_temperature=0.8,
    )
    compiled = optimizer.compile(student, trainset=trainset)

    assert any("Current system prompt fragments:" in prompt for prompt in seen_prompts)
    assert 0.8 in seen_temperatures
    assert compiled.optimized_system_prompt.data == "Improved instruction"
    assert compiled.get_compile_info()["prompt_model_calls"] == 1


def test_copro_uses_init_temperature_for_explicit_prompt_model():
    seen_temperatures: list[float | None] = []
    prompt_model = TemperatureTrackingPromptModel(seen_temperatures)
    student = _make_agent(StaticResponseModel("Paris"))
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]

    optimizer = COPRO(
        metric=exact_match,
        prompt_model=prompt_model,
        breadth=2,
        depth=1,
        init_temperature=0.7,
    )
    optimizer.compile(student, trainset=trainset)

    assert seen_temperatures == [0.7]


def test_copro_tracks_stats_and_candidate_programs():
    student = _make_agent(StaticResponseModel("Paris"))
    trainset = [Example(inputs="What is the capital of France?", labels="Paris")]

    optimizer = COPRO(
        metric=exact_match,
        prompt_model=lambda prompt: "Improved instruction",
        breadth=2,
        depth=1,
        track_stats=True,
    )
    compiled = optimizer.compile(student, trainset=trainset)

    assert compiled._compiled is True
    assert compiled.total_calls == 2
    assert compiled.prompt_model_total_calls == 1
    assert len(compiled.candidate_programs) >= 1
    assert hasattr(compiled, "results_best")
    assert hasattr(compiled, "results_latest")
    assert compiled.get_compile_info()["prompt_model_calls"] == 1
