"""Prompt optimization module for msgflux.

This module provides optimizers for improving prompt performance through
various strategies like few-shot selection and bootstrapping.

Example:
    >>> from msgflux.optim import LabeledFewShot, BootstrapFewShot
    >>>
    >>> # Simple few-shot selection
    >>> optimizer = LabeledFewShot(agent.parameters(), trainset=examples, k=16)
    >>> optimizer.step()
    >>>
    >>> # Bootstrap with metric
    >>> optimizer = BootstrapFewShot(
    ...     agent.parameters(),
    ...     metric=exact_match,
    ...     max_bootstrapped_demos=4,
    ... )
    >>> optimizer.step(trainset=examples, teacher=agent)
"""

from msgflux.optim.bootstrap import BootstrapFewShot, BootstrapResult, Trace
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.optimizer import Optimizer

__all__ = [
    # Base class
    "Optimizer",
    # Optimizers
    "LabeledFewShot",
    "BootstrapFewShot",
    # Data classes
    "BootstrapResult",
    "Trace",
]
