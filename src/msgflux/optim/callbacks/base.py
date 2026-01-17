"""Base classes for optimizer callbacks.

This module provides the base Callback class and CallbackList for managing
multiple callbacks during optimization.
"""

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from msgflux.optim.optimizer import Optimizer


class Callback(ABC):
    """Base class for optimizer callbacks.

    Callbacks are called at specific points during optimization:

    - on_optimization_begin: Called once at the start
    - on_optimization_end: Called once at the end
    - on_step_begin: Called before each optimization step
    - on_step_end: Called after each optimization step
    - on_trial_begin: Called before each trial (for trial-based optimizers)
    - on_trial_end: Called after each trial
    - on_batch_begin: Called before processing a batch
    - on_batch_end: Called after processing a batch

    All methods receive the optimizer instance and a logs dictionary
    that can be used to pass information between callbacks.

    Example:
        >>> class MyCallback(Callback):
        ...     def on_step_begin(self, step, logs=None):
        ...         print(f"Starting step {step}")
        ...
        ...     def on_step_end(self, step, logs=None):
        ...         print(f"Finished step {step} with score {logs.get('score')}")
        ...         return False  # Don't stop
    """

    optimizer: Optional["Optimizer"] = None

    def set_optimizer(self, optimizer: "Optimizer") -> None:
        """Set the optimizer reference.

        Args:
            optimizer: The optimizer being used.
        """
        self.optimizer = optimizer

    def on_optimization_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of optimization.

        Args:
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_optimization_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of optimization.

        Args:
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called before each optimization step.

        Args:
            step: Current step number.
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Called after each optimization step.

        Args:
            step: Current step number.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True to stop optimization early, False to continue.
        """
        return False

    def on_trial_begin(
        self, trial: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called before each trial (for trial-based optimizers).

        Args:
            trial: Current trial number.
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Called after each trial.

        Args:
            trial: Current trial number.
            score: Score achieved in this trial.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True to stop optimization early, False to continue.
        """
        return False

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called before processing a batch.

        Args:
            batch: Current batch number.
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called after processing a batch.

        Args:
            batch: Current batch number.
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_generation_begin(
        self, generation: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called before each generation (for evolutionary optimizers).

        Args:
            generation: Current generation number.
            logs: Dictionary for passing information between callbacks.
        """
        pass

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Called after each generation.

        Args:
            generation: Current generation number.
            best_score: Best score in this generation.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True to stop optimization early, False to continue.
        """
        return False


class CallbackList:
    """Container for managing multiple callbacks.

    Args:
        callbacks: List of Callback instances.

    Example:
        >>> callbacks = CallbackList([
        ...     EarlyStoppingCallback(patience=5),
        ...     CheckpointCallback(directory="./checkpoints"),
        ... ])
        >>> callbacks.set_optimizer(optimizer)
        >>> callbacks.on_optimization_begin()
        >>> for step in range(100):
        ...     callbacks.on_step_begin(step)
        ...     # ... optimization step ...
        ...     if callbacks.on_step_end(step, logs={'score': score}):
        ...         break
        >>> callbacks.on_optimization_end()
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self._logs: Dict[str, Any] = {}

    def append(self, callback: Callback) -> None:
        """Add a callback.

        Args:
            callback: The callback to add.
        """
        self.callbacks.append(callback)

    def extend(self, callbacks: List[Callback]) -> None:
        """Add multiple callbacks.

        Args:
            callbacks: List of callbacks to add.
        """
        self.callbacks.extend(callbacks)

    def set_optimizer(self, optimizer: "Optimizer") -> None:
        """Set optimizer reference on all callbacks.

        Args:
            optimizer: The optimizer being used.
        """
        for callback in self.callbacks:
            callback.set_optimizer(optimizer)

    def on_optimization_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_optimization_begin on all callbacks.

        Args:
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_optimization_begin(self._logs)

    def on_optimization_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_optimization_end on all callbacks.

        Args:
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_optimization_end(self._logs)

    def on_step_begin(self, step: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_step_begin on all callbacks.

        Args:
            step: Current step number.
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_step_begin(step, self._logs)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Call on_step_end on all callbacks.

        Args:
            step: Current step number.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True if any callback requests stopping.
        """
        logs = logs or {}
        self._logs.update(logs)
        stop = False
        for callback in self.callbacks:
            if callback.on_step_end(step, self._logs):
                stop = True
        return stop

    def on_trial_begin(
        self, trial: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_trial_begin on all callbacks.

        Args:
            trial: Current trial number.
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_trial_begin(trial, self._logs)

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Call on_trial_end on all callbacks.

        Args:
            trial: Current trial number.
            score: Score achieved in this trial.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True if any callback requests stopping.
        """
        logs = logs or {}
        logs["score"] = score
        self._logs.update(logs)
        stop = False
        for callback in self.callbacks:
            if callback.on_trial_end(trial, score, self._logs):
                stop = True
        return stop

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_batch_begin on all callbacks.

        Args:
            batch: Current batch number.
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_batch_begin(batch, self._logs)

    def on_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_batch_end on all callbacks.

        Args:
            batch: Current batch number.
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_batch_end(batch, self._logs)

    def on_generation_begin(
        self, generation: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Call on_generation_begin on all callbacks.

        Args:
            generation: Current generation number.
            logs: Dictionary for passing information between callbacks.
        """
        logs = logs or {}
        self._logs.update(logs)
        for callback in self.callbacks:
            callback.on_generation_begin(generation, self._logs)

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Call on_generation_end on all callbacks.

        Args:
            generation: Current generation number.
            best_score: Best score in this generation.
            logs: Dictionary for passing information between callbacks.

        Returns:
            True if any callback requests stopping.
        """
        logs = logs or {}
        logs["best_score"] = best_score
        self._logs.update(logs)
        stop = False
        for callback in self.callbacks:
            if callback.on_generation_end(generation, best_score, self._logs):
                stop = True
        return stop

    def __len__(self) -> int:
        return len(self.callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    @property
    def logs(self) -> Dict[str, Any]:
        """Get the accumulated logs."""
        return self._logs.copy()
