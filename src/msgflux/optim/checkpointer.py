"""Checkpoint manager for optimizers.

This module provides the Checkpointer class for saving and loading
optimizer state during long-running optimizations.
"""

import json
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from msgflux.logger import init_logger

if TYPE_CHECKING:
    from msgflux.optim.optimizer import Optimizer

logger = init_logger(__name__)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    path: Path
    step: int
    score: Optional[float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Checkpointer:
    """Checkpoint manager for optimizers.

    Handles saving and loading optimizer state during long-running optimizations.

    Args:
        directory: Directory to save checkpoints.
        save_every: Save checkpoint every N steps. If None, only save on request.
        keep_last: Number of recent checkpoints to keep. Older ones are deleted.
        save_best: Whether to always keep the best checkpoint.
        metric_name: Name of the metric to track for "best" determination.
        mode: 'max' if higher is better, 'min' if lower is better.
        verbose: Whether to log checkpoint events.

    Example:
        >>> checkpointer = Checkpointer("./checkpoints", save_every=5, keep_last=3)
        >>>
        >>> for step in range(100):
        ...     score = optimizer.step(trainset)
        ...     checkpointer.step(optimizer, score)
        >>>
        >>> # Later, resume from checkpoint
        >>> optimizer.load_state_dict(checkpointer.load_latest()["optimizer_state"])
    """

    CHECKPOINT_PATTERN = "checkpoint_step_{step:06d}.pt"
    BEST_CHECKPOINT_NAME = "best.pt"
    INDEX_FILE = "checkpoints.json"

    def __init__(
        self,
        directory: Union[str, Path],
        save_every: Optional[int] = None,
        keep_last: int = 5,
        save_best: bool = True,
        metric_name: str = "score",
        mode: str = "max",
        verbose: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.directory = Path(directory)
        self.save_every = save_every
        self.keep_last = keep_last
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.verbose = verbose

        # Create directory
        self.directory.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self._checkpoints: List[CheckpointInfo] = []
        self._best_score: Optional[float] = None
        self._best_checkpoint: Optional[CheckpointInfo] = None
        self._current_step = 0

        # Load existing index if present
        self._load_index()

    def step(
        self,
        optimizer: "Optimizer",
        score: Optional[float] = None,
        force_save: bool = False,
    ) -> Optional[CheckpointInfo]:
        """Called after each optimizer step to potentially save checkpoint.

        Args:
            optimizer: The optimizer to checkpoint.
            score: Current metric value (optional).
            force_save: Force saving regardless of save_every.

        Returns:
            CheckpointInfo if a checkpoint was saved, None otherwise.
        """
        self._current_step += 1

        # Check if we should save
        should_save = force_save
        if self.save_every is not None:
            should_save = should_save or (self._current_step % self.save_every == 0)

        # Check if this is the best score
        is_best = False
        if score is not None and self.save_best:
            is_best = self._is_best(score)
            if is_best:
                self._best_score = score

        if should_save or is_best:
            return self.save(optimizer, score, is_best=is_best)

        return None

    def save(
        self,
        optimizer: "Optimizer",
        score: Optional[float] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CheckpointInfo:
        """Save a checkpoint.

        Args:
            optimizer: The optimizer to checkpoint.
            score: Current metric value (optional).
            is_best: Whether this is the best checkpoint so far.
            metadata: Additional metadata to save.

        Returns:
            CheckpointInfo for the saved checkpoint.
        """
        # Create checkpoint filename
        filename = self.CHECKPOINT_PATTERN.format(step=self._current_step)
        filepath = self.directory / filename

        # Build checkpoint data
        checkpoint_data = {
            "optimizer_state": optimizer.state_dict(),
            "step": self._current_step,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save checkpoint
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)

        if self.verbose:
            score_str = f" (score: {score:.4f})" if score is not None else ""
            logger.info(f"Saved checkpoint at step {self._current_step}{score_str}")

        # Create info
        info = CheckpointInfo(
            path=filepath,
            step=self._current_step,
            score=score,
            timestamp=checkpoint_data["timestamp"],
            metadata=checkpoint_data["metadata"],
        )

        self._checkpoints.append(info)

        # Save as best if applicable
        if is_best:
            best_path = self.directory / self.BEST_CHECKPOINT_NAME
            shutil.copy(filepath, best_path)
            self._best_checkpoint = CheckpointInfo(
                path=best_path,
                step=self._current_step,
                score=score,
                timestamp=checkpoint_data["timestamp"],
                metadata=checkpoint_data["metadata"],
            )
            if self.verbose:
                logger.info(f"Saved best checkpoint (score: {score:.4f})")

        # Cleanup old checkpoints
        self._cleanup()

        # Save index
        self._save_index()

        return info

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a checkpoint from a specific path.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Checkpoint data dictionary.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Returns:
            Checkpoint data dictionary, or None if no checkpoints exist.
        """
        if not self._checkpoints:
            return None

        latest = max(self._checkpoints, key=lambda c: c.step)
        if self.verbose:
            logger.info(f"Loading checkpoint from step {latest.step}")
        return self.load(latest.path)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint.

        Returns:
            Checkpoint data dictionary, or None if no best checkpoint exists.
        """
        best_path = self.directory / self.BEST_CHECKPOINT_NAME
        if best_path.exists():
            if self.verbose:
                logger.info(f"Loading best checkpoint (score: {self._best_score})")
            return self.load(best_path)
        return None

    def load_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Load a checkpoint from a specific step.

        Args:
            step: The step number to load.

        Returns:
            Checkpoint data dictionary, or None if not found.
        """
        for checkpoint in self._checkpoints:
            if checkpoint.step == step:
                if self.verbose:
                    logger.info(f"Loading checkpoint from step {step}")
                return self.load(checkpoint.path)
        return None

    def _is_best(self, score: float) -> bool:
        """Check if score is the best so far."""
        if self._best_score is None:
            return True

        if self.mode == "max":
            return score > self._best_score
        else:
            return score < self._best_score

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self._checkpoints) <= self.keep_last:
            return

        # Sort by step, oldest first
        sorted_checkpoints = sorted(self._checkpoints, key=lambda c: c.step)

        # Remove oldest, keeping keep_last
        to_remove = sorted_checkpoints[: -self.keep_last]

        for checkpoint in to_remove:
            # Don't remove best checkpoint
            if self._best_checkpoint and checkpoint.path == self._best_checkpoint.path:
                continue

            # Don't remove the best.pt file
            if checkpoint.path.name == self.BEST_CHECKPOINT_NAME:
                continue

            try:
                checkpoint.path.unlink()
                self._checkpoints.remove(checkpoint)
                if self.verbose:
                    logger.debug(f"Removed old checkpoint: {checkpoint.path.name}")
            except OSError:
                pass

    def _save_index(self) -> None:
        """Save checkpoint index to JSON file."""
        index_data = {
            "checkpoints": [
                {
                    "path": str(c.path),
                    "step": c.step,
                    "score": c.score,
                    "timestamp": c.timestamp,
                    "metadata": c.metadata,
                }
                for c in self._checkpoints
            ],
            "best_score": self._best_score,
            "current_step": self._current_step,
        }

        index_path = self.directory / self.INDEX_FILE
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

    def _load_index(self) -> None:
        """Load checkpoint index from JSON file."""
        index_path = self.directory / self.INDEX_FILE
        if not index_path.exists():
            return

        try:
            with open(index_path, "r") as f:
                index_data = json.load(f)

            self._checkpoints = [
                CheckpointInfo(
                    path=Path(c["path"]),
                    step=c["step"],
                    score=c["score"],
                    timestamp=c["timestamp"],
                    metadata=c["metadata"],
                )
                for c in index_data.get("checkpoints", [])
                if Path(c["path"]).exists()
            ]

            self._best_score = index_data.get("best_score")
            self._current_step = index_data.get("current_step", 0)

            # Check if best checkpoint exists
            best_path = self.directory / self.BEST_CHECKPOINT_NAME
            if best_path.exists():
                # Find the checkpoint info for the best
                for c in self._checkpoints:
                    if c.score == self._best_score:
                        self._best_checkpoint = CheckpointInfo(
                            path=best_path,
                            step=c.step,
                            score=c.score,
                            timestamp=c.timestamp,
                            metadata=c.metadata,
                        )
                        break

            if self.verbose and self._checkpoints:
                logger.info(
                    f"Loaded checkpoint index: {len(self._checkpoints)} checkpoints, "
                    f"current step: {self._current_step}"
                )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load checkpoint index: {e}")

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints."""
        return sorted(self._checkpoints, key=lambda c: c.step, reverse=True)

    @property
    def best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get info about the best checkpoint."""
        return self._best_checkpoint

    @property
    def latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get info about the latest checkpoint."""
        if not self._checkpoints:
            return None
        return max(self._checkpoints, key=lambda c: c.step)

    @property
    def current_step(self) -> int:
        """Get the current step count."""
        return self._current_step

    def clear(self) -> None:
        """Clear all checkpoints."""
        for checkpoint in self._checkpoints:
            try:
                checkpoint.path.unlink()
            except OSError:
                pass

        # Remove best checkpoint
        best_path = self.directory / self.BEST_CHECKPOINT_NAME
        if best_path.exists():
            best_path.unlink()

        # Remove index
        index_path = self.directory / self.INDEX_FILE
        if index_path.exists():
            index_path.unlink()

        self._checkpoints = []
        self._best_score = None
        self._best_checkpoint = None
        self._current_step = 0

        if self.verbose:
            logger.info("Cleared all checkpoints")
