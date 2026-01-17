"""Callback system for optimizers.

This module provides a callback system for extending optimizer behavior
without modifying the optimizer code.

Example:
    >>> from msgflux.optim.callbacks import (
    ...     Callback,
    ...     CallbackList,
    ...     EarlyStoppingCallback,
    ...     CheckpointCallback,
    ...     ProgressCallback,
    ...     HistoryCallback,
    ... )
    >>>
    >>> callbacks = [
    ...     EarlyStoppingCallback(patience=5),
    ...     CheckpointCallback(directory="./checkpoints"),
    ...     ProgressCallback(log_every=1),
    ... ]
    >>>
    >>> optimizer = MIPROv2(
    ...     agent.parameters(),
    ...     metric=exact_match,
    ...     callbacks=callbacks,
    ... )
"""

from msgflux.optim.callbacks.base import Callback, CallbackList
from msgflux.optim.callbacks.builtin import (
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    ProgressCallback,
)

__all__ = [
    # Base classes
    "Callback",
    "CallbackList",
    # Built-in callbacks
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "ProgressCallback",
    "HistoryCallback",
]
