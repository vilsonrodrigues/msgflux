"""Export utilities for optimization experiments.

This module provides utilities for exporting optimization experiment data
to various formats like JSON and CSV.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Record of a single optimization step."""

    step: int
    score: Optional[float] = None
    best_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    """Record of an entire optimization experiment."""

    name: str
    optimizer: str
    started_at: str
    finished_at: Optional[str] = None
    final_score: Optional[float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    steps: List[StepRecord] = field(default_factory=list)


class ExperimentExporter:
    """Export optimization experiment data to various formats.

    Tracks optimization progress and exports to JSON or CSV files.

    Args:
        name: Name of the experiment.
        optimizer: Name of the optimizer being used.
        config: Optional configuration dictionary.

    Example:
        >>> exporter = ExperimentExporter(name="my-experiment", optimizer="MIPROv2")
        >>>
        >>> for step in range(100):
        ...     score = optimizer.step(trainset)
        ...     exporter.log_step(step, score)
        >>>
        >>> exporter.finish(final_score=0.85)
        >>> exporter.save_json("results.json")
        >>> exporter.save_csv("results.csv")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        optimizer: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.record = ExperimentRecord(
            name=name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            optimizer=optimizer or "unknown",
            started_at=datetime.now().isoformat(),
            config=config or {},
        )
        self._best_score: Optional[float] = None
        self._mode = "max"  # Default: higher is better

    def set_mode(self, mode: str) -> None:
        """Set the optimization mode.

        Args:
            mode: 'max' if higher scores are better, 'min' if lower is better.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self._mode = mode

    def log_step(
        self,
        step: int,
        score: Optional[float] = None,
        best_score: Optional[float] = None,
        **metadata,
    ) -> None:
        """Log a single optimization step.

        Args:
            step: Step number.
            score: Score achieved at this step.
            best_score: Best score so far (auto-tracked if not provided).
            **metadata: Additional metadata to record.
        """
        # Auto-track best score
        if score is not None and best_score is None:
            if self._best_score is None:
                self._best_score = score
            elif self._mode == "max" and score > self._best_score:
                self._best_score = score
            elif self._mode == "min" and score < self._best_score:
                self._best_score = score
            best_score = self._best_score

        record = StepRecord(
            step=step,
            score=score,
            best_score=best_score,
            metadata=metadata,
        )
        self.record.steps.append(record)

    def log_trial(
        self,
        trial: int,
        score: float,
        best_score: Optional[float] = None,
        **metadata,
    ) -> None:
        """Log a trial (alias for log_step with trial terminology).

        Args:
            trial: Trial number.
            score: Score achieved in this trial.
            best_score: Best score so far.
            **metadata: Additional metadata to record.
        """
        metadata["trial"] = trial
        self.log_step(trial, score, best_score, **metadata)

    def log_generation(
        self,
        generation: int,
        best_score: float,
        avg_score: Optional[float] = None,
        population_size: Optional[int] = None,
        **metadata,
    ) -> None:
        """Log a generation (for evolutionary optimizers).

        Args:
            generation: Generation number.
            best_score: Best score in this generation.
            avg_score: Average score in this generation.
            population_size: Size of the population.
            **metadata: Additional metadata to record.
        """
        metadata["generation"] = generation
        if avg_score is not None:
            metadata["avg_score"] = avg_score
        if population_size is not None:
            metadata["population_size"] = population_size
        self.log_step(generation, best_score, best_score, **metadata)

    def finish(self, final_score: Optional[float] = None) -> None:
        """Mark the experiment as finished.

        Args:
            final_score: Final score achieved.
        """
        self.record.finished_at = datetime.now().isoformat()
        self.record.final_score = final_score or self._best_score

    def save_json(self, path: str) -> None:
        """Save experiment data to JSON file.

        Args:
            path: Path to save the JSON file.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self.record)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save_csv(self, path: str) -> None:
        """Save step data to CSV file.

        Args:
            path: Path to save the CSV file.
        """
        if not self.record.steps:
            return

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["step", "score", "best_score", "timestamp"]

        # Add any metadata keys
        metadata_keys = set()
        for step in self.record.steps:
            metadata_keys.update(step.metadata.keys())
        fieldnames.extend(sorted(metadata_keys))

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for step in self.record.steps:
                row = {
                    "step": step.step,
                    "score": step.score,
                    "best_score": step.best_score,
                    "timestamp": step.timestamp,
                    **step.metadata,
                }
                writer.writerow(row)

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary.

        Returns:
            Dictionary representation of the experiment.
        """
        return asdict(self.record)

    def get_scores(self) -> List[float]:
        """Get all recorded scores.

        Returns:
            List of scores (excluding None values).
        """
        return [s.score for s in self.record.steps if s.score is not None]

    def get_best_scores(self) -> List[float]:
        """Get all recorded best scores.

        Returns:
            List of best scores (excluding None values).
        """
        return [s.best_score for s in self.record.steps if s.best_score is not None]

    def get_steps(self) -> List[int]:
        """Get all step numbers.

        Returns:
            List of step numbers.
        """
        return [s.step for s in self.record.steps]

    @property
    def best_score(self) -> Optional[float]:
        """Get the best score observed."""
        return self._best_score

    @property
    def num_steps(self) -> int:
        """Get the number of steps recorded."""
        return len(self.record.steps)

    @property
    def duration(self) -> Optional[float]:
        """Get the duration in seconds (if finished).

        Returns:
            Duration in seconds, or None if not finished.
        """
        if self.record.finished_at is None:
            return None

        start = datetime.fromisoformat(self.record.started_at)
        end = datetime.fromisoformat(self.record.finished_at)
        return (end - start).total_seconds()

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment.

        Returns:
            Dictionary with summary statistics.
        """
        scores = self.get_scores()
        return {
            "name": self.record.name,
            "optimizer": self.record.optimizer,
            "num_steps": self.num_steps,
            "final_score": self.record.final_score,
            "best_score": self._best_score,
            "avg_score": sum(scores) / len(scores) if scores else None,
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "duration_seconds": self.duration,
            "started_at": self.record.started_at,
            "finished_at": self.record.finished_at,
        }
