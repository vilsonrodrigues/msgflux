"""Early stopping callback for optimizers.

This module provides the EarlyStopping class for stopping optimization
when the metric stops improving.
"""

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

from msgflux.logger import init_logger

if TYPE_CHECKING:
    from msgflux.optim.optimizer import Optimizer

logger = init_logger(__name__)


@dataclass
class EarlyStoppingState:
    """State for early stopping tracker."""

    best_score: float
    best_step: int
    best_state: Optional[Dict[str, Any]]
    steps_without_improvement: int
    stopped: bool
    stop_reason: Optional[str]


class EarlyStopping:
    """Early stopping callback for optimizers.

    Monitors a metric and stops training when it stops improving.

    Args:
        patience: Number of steps to wait for improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'max' if higher is better, 'min' if lower is better.
        restore_best: Whether to restore the best state when stopping.
        verbose: Whether to log early stopping events.

    Example:
        >>> early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        >>>
        >>> for step in range(100):
        ...     score = optimizer.step(trainset)
        ...
        ...     if early_stopping(score, optimizer):
        ...         print(f"Stopped early at step {step}")
        ...         break
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        restore_best: bool = True,
        verbose: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose

        # Initialize state
        self._state = EarlyStoppingState(
            best_score=float("-inf") if mode == "max" else float("inf"),
            best_step=0,
            best_state=None,
            steps_without_improvement=0,
            stopped=False,
            stop_reason=None,
        )

    def __call__(
        self,
        score: float,
        optimizer: "Optimizer",
        step: Optional[int] = None,
    ) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.
            optimizer: The optimizer being monitored.
            step: Current step number (optional).

        Returns:
            True if training should stop, False otherwise.
        """
        if self._state.stopped:
            return True

        current_step = step if step is not None else optimizer._step_count
        improved = self._is_improvement(score)

        if improved:
            self._state.best_score = score
            self._state.best_step = current_step
            self._state.steps_without_improvement = 0

            # Save best state
            if self.restore_best:
                self._state.best_state = copy.deepcopy(optimizer.state_dict())

            if self.verbose:
                logger.info(
                    f"EarlyStopping: New best score {score:.4f} at step {current_step}"
                )
        else:
            self._state.steps_without_improvement += 1

            if self.verbose:
                logger.info(
                    f"EarlyStopping: No improvement for "
                    f"{self._state.steps_without_improvement}/{self.patience} steps"
                )

        # Check if should stop
        if self._state.steps_without_improvement >= self.patience:
            self._state.stopped = True
            self._state.stop_reason = (
                f"No improvement for {self.patience} steps. "
                f"Best score: {self._state.best_score:.4f} at step {self._state.best_step}"
            )

            if self.verbose:
                logger.info(f"EarlyStopping: {self._state.stop_reason}")

            # Restore best state
            if self.restore_best and self._state.best_state is not None:
                optimizer.load_state_dict(self._state.best_state)
                if self.verbose:
                    logger.info(
                        f"EarlyStopping: Restored best state from step {self._state.best_step}"
                    )

            return True

        return False

    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement over best."""
        if self.mode == "max":
            return score > self._state.best_score + self.min_delta
        else:
            return score < self._state.best_score - self.min_delta

    def reset(self) -> None:
        """Reset early stopping state."""
        self._state = EarlyStoppingState(
            best_score=float("-inf") if self.mode == "max" else float("inf"),
            best_step=0,
            best_state=None,
            steps_without_improvement=0,
            stopped=False,
            stop_reason=None,
        )

    @property
    def stopped(self) -> bool:
        """Whether early stopping has been triggered."""
        return self._state.stopped

    @property
    def best_score(self) -> float:
        """Best score observed."""
        return self._state.best_score

    @property
    def best_step(self) -> int:
        """Step at which best score was observed."""
        return self._state.best_step

    @property
    def stop_reason(self) -> Optional[str]:
        """Reason for stopping, if stopped."""
        return self._state.stop_reason

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the early stopping as a dict."""
        return {
            "best_score": self._state.best_score,
            "best_step": self._state.best_step,
            "steps_without_improvement": self._state.steps_without_improvement,
            "stopped": self._state.stopped,
            "stop_reason": self._state.stop_reason,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load early stopping state from a dict."""
        self._state.best_score = state_dict.get("best_score", self._state.best_score)
        self._state.best_step = state_dict.get("best_step", self._state.best_step)
        self._state.steps_without_improvement = state_dict.get(
            "steps_without_improvement", self._state.steps_without_improvement
        )
        self._state.stopped = state_dict.get("stopped", self._state.stopped)
        self._state.stop_reason = state_dict.get("stop_reason", self._state.stop_reason)
