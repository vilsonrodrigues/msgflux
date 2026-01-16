"""Prompt optimization module for msgflux.

This module provides optimizers for improving prompt performance through
various strategies including few-shot selection, bootstrapping, instruction
optimization, and evolutionary algorithms.

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
    >>>
    >>> # COPRO instruction optimization
    >>> optimizer = COPRO(
    ...     agent.parameters(),
    ...     metric=exact_match,
    ...     prompt_model=generator_agent,
    ... )
    >>> optimizer.step(trainset, valset)
    >>>
    >>> # MIPROv2 multi-prompt optimization
    >>> optimizer = MIPROv2(
    ...     agent.parameters(),
    ...     metric=exact_match,
    ...     prompt_model=generator_agent,
    ...     num_trials=50,
    ... )
    >>> optimizer.step(trainset, valset)
    >>>
    >>> # GEPA genetic algorithm
    >>> optimizer = GEPA(
    ...     agent.parameters(),
    ...     metric=exact_match,
    ...     prompt_model=generator_agent,
    ...     num_generations=10,
    ... )
    >>> optimizer.step(trainset, valset)
"""

from msgflux.optim.bootstrap import BootstrapFewShot, BootstrapResult, Trace
from msgflux.optim.copro import COPRO, CoproCandidate, CoproResult
from msgflux.optim.gepa import GEPA, GEPAStats, Individual
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.mipro import MIPROv2, MiproTrial, PromptCandidate
from msgflux.optim.optimizer import Optimizer
from msgflux.optim.progress import Colors, OptimProgress, TrialInfo, StepInfo
from msgflux.optim.simba import SIMBA, SIMBACandidate, SIMBAResult, SIMBATrialLog

__all__ = [
    # Base class
    "Optimizer",
    # Demonstration-based optimizers
    "LabeledFewShot",
    "BootstrapFewShot",
    # Instruction optimizers
    "COPRO",
    "MIPROv2",
    # Evolutionary optimizers
    "GEPA",
    # Self-reflective optimizers
    "SIMBA",
    # Progress utilities
    "OptimProgress",
    "Colors",
    "TrialInfo",
    "StepInfo",
    # Data classes
    "BootstrapResult",
    "Trace",
    "CoproCandidate",
    "CoproResult",
    "PromptCandidate",
    "MiproTrial",
    "Individual",
    "GEPAStats",
    "SIMBACandidate",
    "SIMBAResult",
    "SIMBATrialLog",
]
