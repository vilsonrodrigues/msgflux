"""Tests for MIPROv2 parity-sensitive behavior."""

import random
from types import SimpleNamespace

import pytest

from msgflux.examples import Example
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.optim.mipro import MIPROv2


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


class TemperatureTrackingModel:
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
        return "Improved instruction"


def _make_agent(model=None, *, name: str = "qa") -> Agent:
    return Agent(
        name=name,
        model=model or StaticResponseModel("Paris"),
        system_message="You are a geography expert.",
        instructions="Answer with just the city name.",
    )


def test_bootstrap_demo_sets_follow_dspy_recipe():
    optimizer = MIPROv2(
        metric=lambda example, prediction, trace=None: True,
        prompt_model=lambda prompt: prompt,
        auto=None,
        num_candidates=4,
        num_trials=1,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    )
    program = _make_agent()
    trainset = [
        Example(inputs="What is the capital of France?", labels="Paris"),
        Example(inputs="What is the capital of Italy?", labels="Rome"),
        Example(inputs="What is the capital of Spain?", labels="Madrid"),
        Example(inputs="What is the capital of Germany?", labels="Berlin"),
    ]

    candidates = optimizer._bootstrap_demo_sets(
        program=program,
        trainset=trainset,
        teacher=None,
        num_sets=4,
        seed=0,
        zeroshot_opt=False,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    )

    assert list(candidates) == ["qa"]
    assert len(candidates["qa"]) == 4
    assert candidates["qa"][0] == []
    assert all(isinstance(demo, Example) for demo in candidates["qa"][1])
    assert len(candidates["qa"][2]) > 0
    assert len(candidates["qa"][3]) > 0


def test_optuna_search_runs_periodic_full_eval(monkeypatch):
    pytest.importorskip("optuna")

    optimizer = MIPROv2(
        metric=lambda example, prediction: 1.0,
        prompt_model=lambda prompt: prompt,
        auto=None,
        num_candidates=2,
        num_trials=3,
        minibatch_size=10,
        minibatch_full_eval_steps=2,
    )
    program = _make_agent()
    valset = [
        Example(inputs=f"Question {idx}", labels=f"Answer {idx}")
        for idx in range(100)
    ]
    batch_sizes: list[int] = []

    def fake_eval_candidate_program(
        batch_size,
        trainset,
        candidate_program,
        evaluate,
        rng,
    ):
        _ = trainset, candidate_program, evaluate, rng
        batch_sizes.append(batch_size)
        return SimpleNamespace(score=float(batch_size))

    monkeypatch.setattr(
        "msgflux.optim.mipro.mipro.eval_candidate_program",
        fake_eval_candidate_program,
    )
    monkeypatch.setattr(
        optimizer,
        "_final_full_eval",
        lambda score_data, evaluate, valset, rng, best_score, best_program: (
            best_score,
            best_program,
        ),
    )

    best_program = optimizer._optuna_search(
        program=program,
        instruction_candidates={"qa": ["orig", "new"]},
        demo_candidates={"qa": [[], []]},
        evaluate=object(),
        valset=valset,
        num_trials=3,
        use_minibatch=True,
        seed=0,
        rng=random.Random(0),  # noqa: S311
        minibatch_size=10,
        minibatch_full_eval_steps=2,
    )

    assert batch_sizes == [100, 10, 100, 10, 100, 10, 100]
    assert best_program.get_compile_info()["total_calls"] == sum(batch_sizes)


def test_grounded_proposer_uses_dataset_program_and_demo_context():
    prompts: list[str] = []

    def prompt_model(prompt):
        prompts.append(prompt)
        return "Improved instruction"

    optimizer = MIPROv2(
        metric=lambda example, prediction: 1.0,
        prompt_model=prompt_model,
        auto=None,
        num_candidates=3,
        num_trials=1,
    )
    program = _make_agent()
    trainset = [
        Example(inputs="What is the capital of France?", labels="Paris"),
        Example(inputs="What is the capital of Italy?", labels="Rome"),
        Example(inputs="What is the capital of Spain?", labels="Madrid"),
    ]
    demo_candidates = {
        "qa": [
            [],
            [Example(inputs="What is the capital of Germany?", labels="Berlin")],
        ]
    }

    candidates = optimizer._propose_instructions(
        program=program,
        trainset=trainset,
        demo_candidates=demo_candidates,
        num_candidates=3,
        rng=random.Random(0),  # noqa: S311
        view_data_batch_size=10,
        program_aware=True,
        data_aware=True,
        tip_aware=True,
        fewshot_aware=True,
    )

    assert candidates["qa"][0] == program.get_system_prompt()
    assert candidates["qa"][1:] == [
        "Improved instruction",
        "Improved instruction",
    ]
    assert any("Representative training examples:" in prompt for prompt in prompts)
    assert any("Program outline:" in prompt for prompt in prompts)
    assert any("Component name:\nqa" in prompt for prompt in prompts)
    assert any("No task demos provided." in prompt for prompt in prompts)
    assert any("Input: What is the capital of Germany?" in prompt for prompt in prompts)


def test_grounded_proposer_generates_requested_candidates_without_demos():
    optimizer = MIPROv2(
        metric=lambda example, prediction: 1.0,
        prompt_model=lambda prompt: f"Improved: {prompt.splitlines()[0]}",
        auto=None,
        num_candidates=3,
        num_trials=1,
    )
    program = _make_agent()

    candidates = optimizer._propose_instructions(
        program=program,
        trainset=[],
        demo_candidates={},
        num_candidates=3,
        rng=random.Random(0),  # noqa: S311
        view_data_batch_size=10,
        program_aware=True,
        data_aware=True,
        tip_aware=True,
        fewshot_aware=True,
    )

    assert len(candidates["qa"]) == 3
    assert candidates["qa"][0] == program.get_system_prompt()


def test_grounded_proposer_uses_init_temperature_for_model_objects():
    seen_temperatures: list[float | None] = []
    prompt_model = TemperatureTrackingModel(seen_temperatures)
    optimizer = MIPROv2(
        metric=lambda example, prediction: 1.0,
        prompt_model=prompt_model,
        auto=None,
        num_candidates=2,
        num_trials=1,
        init_temperature=0.7,
    )
    program = _make_agent()

    optimizer._propose_instructions(
        program=program,
        trainset=[
            Example(inputs="What is the capital of France?", labels="Paris"),
        ],
        demo_candidates={"qa": [[]]},
        num_candidates=2,
        rng=random.Random(0),  # noqa: S311
        view_data_batch_size=10,
        program_aware=True,
        data_aware=True,
        tip_aware=True,
        fewshot_aware=True,
    )

    assert 0.2 in seen_temperatures
    assert 0.7 in seen_temperatures


def test_zero_shot_bootstrap_still_collects_context_for_proposer(monkeypatch):
    captured: dict[str, int] = {}

    def fake_create_n_fewshot_demo_sets(**kwargs):
        captured["max_bootstrapped_demos"] = kwargs["max_bootstrapped_demos"]
        captured["max_labeled_demos"] = kwargs["max_labeled_demos"]
        return {"qa": [[] for _ in range(kwargs["num_candidate_sets"])]}

    monkeypatch.setattr(
        "msgflux.optim.mipro.mipro.create_n_fewshot_demo_sets",
        fake_create_n_fewshot_demo_sets,
    )
    optimizer = MIPROv2(
        metric=lambda example, prediction: 1.0,
        prompt_model=lambda prompt: prompt,
        auto=None,
        num_candidates=2,
        num_trials=1,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    optimizer._bootstrap_demo_sets(
        program=_make_agent(),
        trainset=[
            Example(inputs="What is the capital of France?", labels="Paris"),
        ],
        teacher=None,
        num_sets=2,
        seed=0,
        zeroshot_opt=True,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
    )

    assert captured == {
        "max_bootstrapped_demos": 3,
        "max_labeled_demos": 0,
    }
