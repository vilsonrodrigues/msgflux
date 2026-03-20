"""LabeledFewShot — select labeled examples as demonstrations.

The simplest optimizer: sample *k* examples from the training set and
assign them as ``demos`` on every Agent in the student program.
"""

import random
from typing import Any, List

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.teleprompter import Teleprompter


class LabeledFewShot(Teleprompter):
    """Select labeled examples from a trainset as demonstrations.

    Args:
        k: Number of examples to select per agent. Defaults to 16.

    Example::

        optimizer = LabeledFewShot(k=4)
        compiled = optimizer.compile(student, trainset=examples)
    """

    def __init__(self, k: int = 16):
        self.k = k

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        sample: bool = True,
        seed: int = 0,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Select *k* examples and assign them as demos on every Agent.

        Args:
            student: Module to optimize (deep-copied internally).
            trainset: Training examples to sample from.
            sample: If *True*, randomly sample. If *False*, take first *k*.
            seed: Random seed for reproducibility.

        Returns:
            Compiled copy of *student* with demos assigned.
        """
        student = student.reset_copy()
        rng = random.Random(seed)  # noqa: S311
        k = min(self.k, len(trainset))

        for _name, agent in student.named_agents():
            if sample:
                agent.optimized_examples = rng.sample(trainset, k)
            else:
                agent.optimized_examples = list(trainset[:k])

        student.compile_(optimizer="LabeledFewShot", score=None)
        return student
