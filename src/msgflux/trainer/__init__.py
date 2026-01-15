"""Training module for msgflux.

This module provides the Trainer class for orchestrating the prompt
optimization process, including training loops, evaluation, and callbacks.

Example:
    >>> from msgflux.trainer import Trainer, TrainerConfig
    >>> from msgflux.optim import BootstrapFewShot
    >>> from msgflux.evaluate import Evaluator, exact_match
    >>>
    >>> trainer = Trainer(
    ...     module=agent,
    ...     optimizer=BootstrapFewShot(agent.parameters(), metric=exact_match),
    ...     evaluator=Evaluator(metric=exact_match),
    ...     config=TrainerConfig(max_epochs=5, early_stopping_patience=3),
    ... )
    >>>
    >>> trainer.fit(trainset=train_data, valset=val_data)
    >>> print(f"Best score: {trainer.state.best_score}%")
"""

from msgflux.trainer.trainer import (
    Callback,
    EarlyStopping,
    ProgressCallback,
    Trainer,
    TrainerConfig,
    TrainingState,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ProgressCallback",
]
