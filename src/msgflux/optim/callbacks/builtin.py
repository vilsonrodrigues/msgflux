"""Built-in callbacks for optimizers.

This module provides commonly used callbacks like early stopping,
checkpointing, progress logging, and history tracking.
"""

from typing import Any, Dict, List, Optional

from msgflux.logger import init_logger
from msgflux.optim.callbacks.base import Callback
from msgflux.optim.checkpointer import Checkpointer
from msgflux.optim.early_stopping import EarlyStopping

logger = init_logger(__name__)


class EarlyStoppingCallback(Callback):
    """Early stopping as a callback.

    Wraps the EarlyStopping class to be used as a callback.

    Args:
        patience: Number of steps to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' if higher is better, 'min' if lower is better.
        restore_best: Whether to restore the best state when stopping.
        verbose: Whether to log early stopping events.

    Example:
        >>> callback = EarlyStoppingCallback(patience=5)
        >>> optimizer = MIPROv2(..., callbacks=[callback])
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best: bool = True,
        verbose: bool = True,
    ):
        self._early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            restore_best=restore_best,
            verbose=verbose,
        )

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if early stopping should trigger after a trial."""
        if self.optimizer is None:
            return False
        return self._early_stopping(score, self.optimizer, trial)

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Check if early stopping should trigger after a step."""
        if logs and "score" in logs and self.optimizer is not None:
            return self._early_stopping(logs["score"], self.optimizer, step)
        return False

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if early stopping should trigger after a generation."""
        if self.optimizer is None:
            return False
        return self._early_stopping(best_score, self.optimizer, generation)

    @property
    def stopped(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._early_stopping.stopped

    @property
    def best_score(self) -> float:
        """Best score observed."""
        return self._early_stopping.best_score

    @property
    def best_step(self) -> int:
        """Step at which best score was observed."""
        return self._early_stopping.best_step

    def reset(self) -> None:
        """Reset early stopping state."""
        self._early_stopping.reset()


class CheckpointCallback(Callback):
    """Checkpointing as a callback.

    Wraps the Checkpointer class to be used as a callback.

    Args:
        directory: Directory to save checkpoints.
        save_every: Save checkpoint every N steps.
        keep_last: Number of recent checkpoints to keep.
        save_best: Whether to always keep the best checkpoint.
        mode: 'max' if higher is better, 'min' if lower is better.
        verbose: Whether to log checkpoint events.

    Example:
        >>> callback = CheckpointCallback("./checkpoints", save_every=10)
        >>> optimizer = MIPROv2(..., callbacks=[callback])
    """

    def __init__(
        self,
        directory: str,
        save_every: int = 10,
        keep_last: int = 3,
        save_best: bool = True,
        mode: str = "max",
        verbose: bool = True,
    ):
        self._checkpointer = Checkpointer(
            directory=directory,
            save_every=save_every,
            keep_last=keep_last,
            save_best=save_best,
            mode=mode,
            verbose=verbose,
        )

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Save checkpoint after a step if needed."""
        if self.optimizer is None:
            return False
        score = logs.get("score") if logs else None
        self._checkpointer.step(self.optimizer, score)
        return False

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save checkpoint after a trial if needed."""
        if self.optimizer is None:
            return False
        self._checkpointer.step(self.optimizer, score)
        return False

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save checkpoint after a generation if needed."""
        if self.optimizer is None:
            return False
        self._checkpointer.step(self.optimizer, best_score)
        return False

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        return self._checkpointer.load_latest()

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        return self._checkpointer.load_best()

    @property
    def checkpointer(self) -> Checkpointer:
        """Get the underlying checkpointer."""
        return self._checkpointer


class ProgressCallback(Callback):
    """Progress logging callback.

    Logs optimization progress at specified intervals.

    Args:
        log_every: Log progress every N steps.
        include_scores: Whether to include scores in log messages.

    Example:
        >>> callback = ProgressCallback(log_every=5)
        >>> optimizer = MIPROv2(..., callbacks=[callback])
    """

    def __init__(self, log_every: int = 1, include_scores: bool = True):
        self.log_every = log_every
        self.include_scores = include_scores

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Log progress after a step."""
        if step % self.log_every == 0:
            if self.include_scores and logs and "score" in logs:
                score = logs["score"]
                best = logs.get("best_score", score)
                logger.info(f"Step {step}: score={score:.4f}, best={best:.4f}")
            else:
                logger.info(f"Step {step}")
        return False

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log progress after a trial."""
        if trial % self.log_every == 0:
            best = logs.get("best_score", score) if logs else score
            logger.info(f"Trial {trial}: score={score:.4f}, best={best:.4f}")
        return False

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log progress after a generation."""
        if generation % self.log_every == 0:
            avg_score = logs.get("avg_score") if logs else None
            if avg_score is not None:
                logger.info(
                    f"Generation {generation}: best={best_score:.4f}, avg={avg_score:.4f}"
                )
            else:
                logger.info(f"Generation {generation}: best={best_score:.4f}")
        return False


class HistoryCallback(Callback):
    """Records optimization history.

    Stores all scores and metrics for later analysis.

    Example:
        >>> history = HistoryCallback()
        >>> optimizer = MIPROv2(..., callbacks=[history])
        >>> # ... run optimization ...
        >>> print(history.get_history())
    """

    def __init__(self):
        self.history: Dict[str, List[Any]] = {
            "step": [],
            "trial": [],
            "generation": [],
            "score": [],
            "best_score": [],
        }
        self._step_count = 0
        self._trial_count = 0
        self._generation_count = 0

    def on_step_end(self, step: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Record step information."""
        self._step_count += 1
        self.history["step"].append(step)
        if logs:
            self.history["score"].append(logs.get("score"))
            self.history["best_score"].append(logs.get("best_score"))
        return False

    def on_trial_end(
        self, trial: int, score: float, logs: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record trial information."""
        self._trial_count += 1
        self.history["trial"].append(trial)
        self.history["score"].append(score)
        if logs:
            self.history["best_score"].append(logs.get("best_score"))
        return False

    def on_generation_end(
        self,
        generation: int,
        best_score: float,
        logs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record generation information."""
        self._generation_count += 1
        self.history["generation"].append(generation)
        self.history["best_score"].append(best_score)
        if logs:
            self.history["score"].append(logs.get("avg_score"))
        return False

    def get_history(self) -> Dict[str, List[Any]]:
        """Get the recorded history.

        Returns:
            Dictionary with lists of recorded values.
        """
        return {k: v[:] for k, v in self.history.items() if v}

    def get_scores(self) -> List[float]:
        """Get all recorded scores.

        Returns:
            List of scores.
        """
        return [s for s in self.history["score"] if s is not None]

    def get_best_scores(self) -> List[float]:
        """Get all recorded best scores.

        Returns:
            List of best scores.
        """
        return [s for s in self.history["best_score"] if s is not None]

    @property
    def final_best_score(self) -> Optional[float]:
        """Get the final best score."""
        best_scores = self.get_best_scores()
        return max(best_scores) if best_scores else None

    def clear(self) -> None:
        """Clear the recorded history."""
        for key in self.history:
            self.history[key] = []
        self._step_count = 0
        self._trial_count = 0
        self._generation_count = 0
