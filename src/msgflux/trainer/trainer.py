"""Trainer for orchestrating prompt optimization.

This module provides the Trainer class that manages the training loop,
evaluation, and callbacks for prompt optimization.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the Trainer.

    Attributes:
        max_epochs: Maximum number of training epochs. Defaults to 10.
        eval_every: Evaluate every N epochs. Defaults to 1.
        save_best: Whether to save the best model state. Defaults to True.
        early_stopping_patience: Stop if no improvement for N epochs.
            If None, no early stopping. Defaults to None.
        verbose: Whether to print progress. Defaults to True.
        log_level: Logging level. Defaults to "INFO".
    """

    max_epochs: int = 10
    eval_every: int = 1
    save_best: bool = True
    early_stopping_patience: Optional[int] = None
    verbose: bool = True
    log_level: str = "INFO"


@dataclass
class TrainingState:
    """Current state of training.

    Attributes:
        epoch: Current epoch number.
        step: Total optimization steps taken.
        best_score: Best validation score achieved.
        best_epoch: Epoch where best score was achieved.
        history: List of epoch results.
        is_best: Whether current epoch is the best.
    """

    epoch: int = 0
    step: int = 0
    best_score: float = 0.0
    best_epoch: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "epoch": self.epoch,
            "step": self.step,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }


class Callback:
    """Base class for trainer callbacks.

    Subclass this to create custom callbacks for training events.
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_evaluate(
        self, trainer: "Trainer", result: EvaluationResult
    ) -> None:
        """Called after evaluation."""
        pass


class EarlyStopping(Callback):
    """Callback for early stopping based on validation score."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]
    ) -> None:
        score = logs.get("val_score", 0.0)

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")


class ProgressCallback(Callback):
    """Callback for logging training progress."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        logger.info(f"Starting training for {trainer.config.max_epochs} epochs")

    def on_train_end(self, trainer: "Trainer") -> None:
        logger.info(
            f"Training complete. Best score: {trainer.state.best_score:.2f}% "
            f"(epoch {trainer.state.best_epoch + 1})"
        )

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Dict[str, Any]
    ) -> None:
        val_score = logs.get("val_score", "N/A")
        if isinstance(val_score, float):
            val_score = f"{val_score:.2f}%"
        logger.info(f"Epoch {epoch + 1}: val_score={val_score}")


class Trainer:
    """Trainer for orchestrating prompt optimization.

    The Trainer manages the training loop, coordinates the optimizer
    and evaluator, and handles callbacks.

    Args:
        module: The module to train.
        optimizer: The optimizer to use.
        evaluator: Optional evaluator for validation.
        config: Training configuration.
        callbacks: List of callback objects.

    Example:
        >>> agent = Agent(name="qa", model=model)
        >>> optimizer = BootstrapFewShot(
        ...     agent.parameters(),
        ...     metric=exact_match,
        ... )
        >>> evaluator = Evaluator(metric=exact_match)
        >>>
        >>> trainer = Trainer(
        ...     module=agent,
        ...     optimizer=optimizer,
        ...     evaluator=evaluator,
        ...     config=TrainerConfig(max_epochs=5),
        ... )
        >>>
        >>> trainer.fit(trainset=train_data, valset=val_data)
        >>> print(f"Best score: {trainer.state.best_score}%")
    """

    def __init__(
        self,
        module: Module,
        optimizer: Optimizer,
        evaluator: Optional[Evaluator] = None,
        config: Optional[TrainerConfig] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.module = module
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.config = config or TrainerConfig()
        self.callbacks = callbacks or []
        self.state = TrainingState()

        # Setup logging
        if self.config.verbose:
            logging.basicConfig(level=getattr(logging, self.config.log_level))

        # Add default progress callback if verbose
        if self.config.verbose and not any(
            isinstance(cb, ProgressCallback) for cb in self.callbacks
        ):
            self.callbacks.append(ProgressCallback())

        # Add early stopping callback if configured
        if self.config.early_stopping_patience:
            self.callbacks.append(
                EarlyStopping(patience=self.config.early_stopping_patience)
            )

        # Store best module state
        self._best_state: Optional[Dict[str, Any]] = None

    def fit(
        self,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        teacher: Optional[Module] = None,
    ) -> TrainingState:
        """Run the training loop.

        Args:
            trainset: Training examples.
            valset: Optional validation examples.
            teacher: Optional teacher module for bootstrap optimizers.

        Returns:
            Final training state.
        """
        self._on_train_begin()

        try:
            for epoch in range(self.config.max_epochs):
                self.state.epoch = epoch
                self._on_epoch_begin(epoch)

                # Training step
                self._train_epoch(trainset, teacher)

                # Evaluation
                logs = {"epoch": epoch}

                if (
                    valset
                    and self.evaluator
                    and (epoch + 1) % self.config.eval_every == 0
                ):
                    eval_result = self._evaluate(valset)
                    logs["val_score"] = eval_result.score

                    # Check if best
                    if eval_result.score > self.state.best_score:
                        self.state.best_score = eval_result.score
                        self.state.best_epoch = epoch
                        self.state.is_best = True

                        if self.config.save_best:
                            self._save_best_state()
                    else:
                        self.state.is_best = False

                # Record history
                self.state.history.append(logs)

                self._on_epoch_end(epoch, logs)

                # Check early stopping
                if self._should_stop():
                    break

        finally:
            self._on_train_end()

        return self.state

    def _train_epoch(
        self,
        trainset: List[Example],
        teacher: Optional[Module] = None,
    ) -> None:
        """Execute one training epoch."""
        # Set module to training mode
        self.module.train()

        # Zero gradients
        self.optimizer.zero_grad()

        # Execute optimizer step
        # Different optimizers may need different arguments
        if hasattr(self.optimizer, "_bootstrap"):
            # BootstrapFewShot needs trainset and teacher
            self.optimizer.step(trainset, teacher=teacher or self.module)
        elif hasattr(self.optimizer, "trainset"):
            # LabeledFewShot has trainset stored
            self.optimizer.step()
        else:
            # Generic optimizer
            self.optimizer.step()

        self.state.step += 1

    def _evaluate(self, valset: List[Example]) -> EvaluationResult:
        """Run evaluation on validation set."""
        # Set module to evaluation mode
        self.module.eval()

        result = self.evaluator(self.module, valset)

        self._on_evaluate(result)

        return result

    def _save_best_state(self) -> None:
        """Save current module state as best."""
        # Save parameter states
        self._best_state = {}
        for name, param in self.module.named_parameters():
            self._best_state[name] = param.clone()

    def restore_best(self) -> None:
        """Restore module to best saved state."""
        if self._best_state is None:
            logger.warning("No best state saved to restore")
            return

        for name, param in self.module.named_parameters():
            if name in self._best_state:
                param.copy_(self._best_state[name])

        logger.info(f"Restored best state from epoch {self.state.best_epoch + 1}")

    def _should_stop(self) -> bool:
        """Check if training should stop early."""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping) and callback.should_stop:
                return True
        return False

    def _on_train_begin(self) -> None:
        """Call train begin callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def _on_train_end(self) -> None:
        """Call train end callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(self)

    def _on_epoch_begin(self, epoch: int) -> None:
        """Call epoch begin callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(self, epoch)

    def _on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Call epoch end callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(self, epoch, logs)

    def _on_evaluate(self, result: EvaluationResult) -> None:
        """Call evaluate callbacks."""
        for callback in self.callbacks:
            callback.on_evaluate(self, result)

    def state_dict(self) -> Dict[str, Any]:
        """Return full trainer state as dict."""
        return {
            "training_state": self.state.to_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_state": self._best_state,
            "config": {
                "max_epochs": self.config.max_epochs,
                "eval_every": self.config.eval_every,
                "save_best": self.config.save_best,
                "early_stopping_patience": self.config.early_stopping_patience,
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load trainer state from dict."""
        # Restore training state
        ts = state_dict.get("training_state", {})
        self.state.epoch = ts.get("epoch", 0)
        self.state.step = ts.get("step", 0)
        self.state.best_score = ts.get("best_score", 0.0)
        self.state.best_epoch = ts.get("best_epoch", 0)
        self.state.history = ts.get("history", [])

        # Restore optimizer state
        if "optimizer_state" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state"])

        # Restore best state
        self._best_state = state_dict.get("best_state")
