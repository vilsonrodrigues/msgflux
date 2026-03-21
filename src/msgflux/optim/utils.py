"""Shared utilities for prompt optimizers."""

import random
from typing import Any


def create_minibatch(
    trainset: list[Any],
    batch_size: int = 50,
    rng: random.Random | None = None,
) -> list[Any]:
    """Sample a minibatch from *trainset*."""
    batch_size = min(batch_size, len(trainset))
    rng = rng or random
    sampled_indices = rng.sample(range(len(trainset)), batch_size)
    return [trainset[i] for i in sampled_indices]


def eval_candidate_program(
    batch_size: int,
    trainset: list[Any],
    candidate_program: Any,
    evaluate: Any,
    rng: random.Random | None = None,
) -> Any:
    """Evaluate a candidate program, using a minibatch if needed."""
    if batch_size >= len(trainset):
        return evaluate(candidate_program, devset=trainset)
    return evaluate(
        candidate_program,
        devset=create_minibatch(trainset, batch_size, rng),
    )


def get_program_with_highest_avg_score(
    param_score_dict: dict[str, list[tuple[float, Any, Any]]],
    fully_evaled_param_combos: set[str],
) -> tuple[Any, float, str, Any]:
    """Return the not-yet-fully-evaluated parameter combo with best mean score."""
    results: list[tuple[str, float, Any, Any]] = []
    for key, values in param_score_dict.items():
        if not values:
            continue
        mean_score = sum(score for score, _program, _params in values) / len(values)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean_score, program, params))

    if not results:
        raise ValueError("No candidate programs available.")

    sorted_results = sorted(results, key=lambda row: row[1], reverse=True)
    for key, mean_score, program, params in sorted_results:
        if key not in fully_evaled_param_combos:
            return program, mean_score, key, params

    return (
        sorted_results[0][2],
        sorted_results[0][1],
        sorted_results[0][0],
        sorted_results[0][3],
    )


def create_n_fewshot_demo_sets(
    student: Any,
    num_candidate_sets: int,
    trainset: list[Any],
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    metric: Any,
    teacher_settings: dict[str, Any] | None,
    max_errors: int | None = None,
    max_rounds: int = 1,
    metric_threshold: float | None = None,
    teacher: Any = None,
    seed: int = 0,
    rng: random.Random | None = None,
) -> dict[str, list[list[Any]]]:
    """Create few-shot candidate demo sets following DSPy's MIPRO recipe."""
    from msgflux.optim.bootstrap import BootstrapFewShot  # noqa: PLC0415
    from msgflux.optim.labeled_fewshot import LabeledFewShot  # noqa: PLC0415

    if num_candidate_sets < 1:
        raise ValueError("num_candidate_sets must be >= 1.")

    demo_candidates = {
        _agent_key(name, agent): []
        for name, agent in student.named_agents()
    }
    rng = rng or random.Random(seed)  # noqa: S311

    # Match DSPy's recipe: zero-shot, labels-only, unshuffled bootstrap,
    # then shuffled variants.
    remaining_sets = max(num_candidate_sets - 3, 0)

    for candidate_seed in range(-3, remaining_sets):
        trainset_copy = list(trainset)

        if candidate_seed == -3:
            program2 = student.reset_copy()
        elif candidate_seed == -2 and max_labeled_demos > 0:
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = teleprompter.compile(
                student,
                trainset=trainset_copy,
                seed=seed,
            )
        elif candidate_seed == -1:
            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
            )
        else:
            rng.shuffle(trainset_copy)
            shuffled_size = rng.randint(1, max(max_bootstrapped_demos, 1))
            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=shuffled_size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy,
            )

        for name, agent in program2.named_agents():
            demo_candidates[_agent_key(name, agent)].append(
                list(agent.optimized_examples)
            )

    return demo_candidates


def _agent_key(name: str, agent: Any) -> str:
    if getattr(agent, "name", None):
        return str(agent.name)
    return str(name)
