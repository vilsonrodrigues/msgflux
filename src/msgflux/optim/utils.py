"""Shared utilities for prompt optimizers."""

import random
from typing import Any, Callable, List, Optional


def create_minibatch(
    trainset: List[Any],
    batch_size: int = 50,
    rng: Optional[random.Random] = None,
) -> List[Any]:
    """Sample a minibatch from *trainset*.

    Args:
        trainset: Full training set.
        batch_size: Desired batch size (clamped to ``len(trainset)``).
        rng: Random generator.  Falls back to ``random`` module if *None*.

    Returns:
        A list of sampled examples.
    """
    batch_size = min(batch_size, len(trainset))
    rng = rng or random
    sampled_indices = rng.sample(range(len(trainset)), batch_size)
    return [trainset[i] for i in sampled_indices]


def eval_candidate_program(
    batch_size: int,
    trainset: List[Any],
    candidate_program: Any,
    evaluate: Callable,
    rng: Optional[random.Random] = None,
) -> Any:
    """Evaluate a candidate program, using a minibatch if needed."""
    if batch_size >= len(trainset):
        return evaluate(candidate_program, devset=trainset)
    return evaluate(
        candidate_program,
        devset=create_minibatch(trainset, batch_size, rng),
    )
