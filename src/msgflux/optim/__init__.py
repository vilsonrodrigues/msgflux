"""Prompt optimization module for msgflux.

Optimizers follow the DSPy ``compile()`` interface::

    optimizer = LabeledFewShot(k=4)
    compiled = optimizer.compile(student, trainset=examples)

Available optimizers:

- :class:`LabeledFewShot` — sample labeled examples as demos
- :class:`BootstrapFewShot` — bootstrap demos from a teacher
- :class:`COPRO` — coordinate prompt optimization via instruction search
- :class:`SIMBA` — stochastic mini-batch ascent with reflection
- :class:`MIPROv2` — multi-prompt instruction proposal with Optuna
- :class:`GEPA` — guided evolution for prompt adaptation
"""

from msgflux.optim.bootstrap import BootstrapFewShot
from msgflux.optim.copro import COPRO
from msgflux.optim.evaluate import EvalResult, Evaluate
from msgflux.optim.gepa import GEPA
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.mipro import MIPROv2
from msgflux.optim.optimizer import Optimizer
from msgflux.optim.simba import SIMBA
from msgflux.optim.teleprompter import Teleprompter
from msgflux.optim.utils import create_minibatch, eval_candidate_program

__all__ = [
    # Base class
    "Teleprompter",
    "Optimizer",
    # Optimizers
    "LabeledFewShot",
    "BootstrapFewShot",
    "COPRO",
    "SIMBA",
    "MIPROv2",
    "GEPA",
    # Evaluation
    "Evaluate",
    "EvalResult",
    # Utilities
    "create_minibatch",
    "eval_candidate_program",
]
