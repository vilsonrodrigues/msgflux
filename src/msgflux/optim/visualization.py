"""Visualization utilities for optimization progress.

This module provides utilities for visualizing optimization progress
using matplotlib or plotly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from msgflux.logger import init_logger

logger = init_logger(__name__)

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    ticker = None

try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None


@dataclass
class PlotData:
    """Data point for plotting."""

    step: int
    score: Optional[float] = None
    best_score: Optional[float] = None
    trial: Optional[int] = None
    generation: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationPlotter:
    """Visualization utilities for optimization progress.

    Supports both matplotlib (static) and plotly (interactive) backends.

    Args:
        backend: 'matplotlib' or 'plotly'. If None, auto-detects.
        style: Plot style ('default', 'dark', 'minimal').
        figsize: Figure size as (width, height).

    Example:
        >>> plotter = OptimizationPlotter()
        >>>
        >>> for step in range(100):
        ...     score = optimizer.step(trainset)
        ...     plotter.log(step, score)
        >>>
        >>> plotter.plot_score_history()
        >>> plotter.save("optimization_progress.png")
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        style: str = "default",
        figsize: Tuple[int, int] = (10, 6),
    ):
        # Auto-detect backend
        if backend is None:
            if PLOTLY_AVAILABLE:
                backend = "plotly"
            elif MATPLOTLIB_AVAILABLE:
                backend = "matplotlib"
            else:
                raise ImportError(
                    "No plotting backend available. "
                    "Install matplotlib or plotly: pip install matplotlib plotly"
                )

        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is not installed. pip install matplotlib")
        if backend == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError("plotly is not installed. pip install plotly")

        self.backend = backend
        self.style = style
        self.figsize = figsize

        # Data storage
        self._data: List[PlotData] = []
        self._current_fig = None

        # Style configuration
        self._colors = {
            "score": "#2196F3",  # Blue
            "best_score": "#4CAF50",  # Green
            "trial": "#FF9800",  # Orange
            "error": "#F44336",  # Red
            "generation": "#9C27B0",  # Purple
        }

    def log(
        self,
        step: int,
        score: Optional[float] = None,
        best_score: Optional[float] = None,
        trial: Optional[int] = None,
        generation: Optional[int] = None,
        **metadata,
    ) -> None:
        """Log a data point.

        Args:
            step: Current step number.
            score: Score at this step.
            best_score: Best score so far.
            trial: Trial number (for trial-based optimizers).
            generation: Generation number (for evolutionary optimizers).
            **metadata: Additional data to log.
        """
        self._data.append(
            PlotData(
                step=step,
                score=score,
                best_score=best_score,
                trial=trial,
                generation=generation,
                metadata=metadata,
            )
        )

    def plot_score_history(
        self,
        title: str = "Optimization Progress",
        show_best: bool = True,
        show_trials: bool = False,
    ) -> Any:
        """Plot score history over time.

        Args:
            title: Plot title.
            show_best: Whether to show best score line.
            show_trials: Whether to mark trial boundaries.

        Returns:
            Figure object (matplotlib Figure or plotly Figure).
        """
        if not self._data:
            raise ValueError("No data to plot. Call log() first.")

        steps = [d.step for d in self._data]
        scores = [d.score for d in self._data if d.score is not None]
        best_scores = [d.best_score for d in self._data if d.best_score is not None]

        if self.backend == "matplotlib":
            return self._plot_matplotlib_score_history(
                steps, scores, best_scores, title, show_best
            )
        else:
            return self._plot_plotly_score_history(
                steps, scores, best_scores, title, show_best
            )

    def _plot_matplotlib_score_history(
        self,
        steps: List[int],
        scores: List[float],
        best_scores: List[float],
        title: str,
        show_best: bool,
    ) -> Any:
        """Create matplotlib score history plot."""
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot score
        if scores:
            ax.plot(
                steps[: len(scores)],
                scores,
                color=self._colors["score"],
                label="Score",
                alpha=0.7,
            )

        # Plot best score
        if show_best and best_scores:
            ax.plot(
                steps[: len(best_scores)],
                best_scores,
                color=self._colors["best_score"],
                label="Best Score",
                linewidth=2,
            )

        # Formatting
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis as percentage if scores are in [0, 1]
        if scores and max(scores) <= 1.0:
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

        self._current_fig = fig
        return fig

    def _plot_plotly_score_history(
        self,
        steps: List[int],
        scores: List[float],
        best_scores: List[float],
        title: str,
        show_best: bool,
    ) -> Any:
        """Create plotly score history plot."""
        fig = go.Figure()

        # Plot score
        if scores:
            fig.add_trace(
                go.Scatter(
                    x=steps[: len(scores)],
                    y=scores,
                    mode="lines",
                    name="Score",
                    line=dict(color=self._colors["score"]),
                    opacity=0.7,
                )
            )

        # Plot best score
        if show_best and best_scores:
            fig.add_trace(
                go.Scatter(
                    x=steps[: len(best_scores)],
                    y=best_scores,
                    mode="lines",
                    name="Best Score",
                    line=dict(color=self._colors["best_score"], width=2),
                )
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title="Score",
            template="plotly_white",
        )

        self._current_fig = fig
        return fig

    def plot_score_distribution(
        self,
        title: str = "Score Distribution",
        bins: int = 20,
    ) -> Any:
        """Plot histogram of scores.

        Args:
            title: Plot title.
            bins: Number of histogram bins.

        Returns:
            Figure object.
        """
        scores = [d.score for d in self._data if d.score is not None]

        if not scores:
            raise ValueError("No scores to plot.")

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.hist(
                scores, bins=bins, color=self._colors["score"], alpha=0.7, edgecolor="white"
            )
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            self._current_fig = fig
            return fig
        else:
            fig = px.histogram(
                x=scores,
                nbins=bins,
                title=title,
                labels={"x": "Score", "y": "Count"},
            )
            fig.update_traces(marker_color=self._colors["score"])
            self._current_fig = fig
            return fig

    def plot_improvement_rate(
        self,
        window: int = 10,
        title: str = "Improvement Rate",
    ) -> Any:
        """Plot rolling improvement rate.

        Shows how fast scores are improving over time.

        Args:
            window: Rolling window size.
            title: Plot title.

        Returns:
            Figure object.
        """
        scores = [d.score for d in self._data if d.score is not None]

        if len(scores) < window + 1:
            raise ValueError(f"Need at least {window + 1} data points.")

        # Calculate rolling improvement
        improvements = []
        for i in range(window, len(scores)):
            window_scores = scores[i - window : i + 1]
            improvement = (window_scores[-1] - window_scores[0]) / window
            improvements.append(improvement)

        steps = list(range(window, len(scores)))

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(steps, improvements, color=self._colors["trial"])
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.fill_between(
                steps,
                improvements,
                0,
                where=[i > 0 for i in improvements],
                color=self._colors["best_score"],
                alpha=0.3,
                label="Improving",
            )
            ax.fill_between(
                steps,
                improvements,
                0,
                where=[i < 0 for i in improvements],
                color=self._colors["error"],
                alpha=0.3,
                label="Declining",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("Improvement Rate")
            ax.set_title(f"{title} (window={window})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._current_fig = fig
            return fig
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=improvements,
                    mode="lines",
                    name="Improvement Rate",
                    line=dict(color=self._colors["trial"]),
                )
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"{title} (window={window})",
                xaxis_title="Step",
                yaxis_title="Improvement Rate",
            )
            self._current_fig = fig
            return fig

    def plot_generation_summary(
        self,
        title: str = "Generation Progress",
    ) -> Any:
        """Plot generation-by-generation progress for evolutionary optimizers.

        Args:
            title: Plot title.

        Returns:
            Figure object.
        """
        generations = [d for d in self._data if d.generation is not None]

        if not generations:
            raise ValueError("No generation data to plot.")

        gen_nums = [d.generation for d in generations]
        scores = [d.score for d in generations if d.score is not None]
        best_scores = [d.best_score for d in generations if d.best_score is not None]

        if self.backend == "matplotlib":
            fig, ax = plt.subplots(figsize=self.figsize)

            if scores:
                ax.plot(
                    gen_nums[: len(scores)],
                    scores,
                    color=self._colors["generation"],
                    label="Avg Score",
                    alpha=0.7,
                    marker="o",
                    markersize=4,
                )

            if best_scores:
                ax.plot(
                    gen_nums[: len(best_scores)],
                    best_scores,
                    color=self._colors["best_score"],
                    label="Best Score",
                    linewidth=2,
                    marker="s",
                    markersize=4,
                )

            ax.set_xlabel("Generation")
            ax.set_ylabel("Score")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._current_fig = fig
            return fig
        else:
            fig = go.Figure()

            if scores:
                fig.add_trace(
                    go.Scatter(
                        x=gen_nums[: len(scores)],
                        y=scores,
                        mode="lines+markers",
                        name="Avg Score",
                        line=dict(color=self._colors["generation"]),
                        opacity=0.7,
                    )
                )

            if best_scores:
                fig.add_trace(
                    go.Scatter(
                        x=gen_nums[: len(best_scores)],
                        y=best_scores,
                        mode="lines+markers",
                        name="Best Score",
                        line=dict(color=self._colors["best_score"], width=2),
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title="Generation",
                yaxis_title="Score",
                template="plotly_white",
            )
            self._current_fig = fig
            return fig

    def save(
        self,
        path: Union[str, Path],
        dpi: int = 150,
    ) -> None:
        """Save current figure to file.

        Args:
            path: Output file path.
            dpi: Resolution for raster formats (matplotlib only).
        """
        if self._current_fig is None:
            raise ValueError("No figure to save. Create a plot first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "matplotlib":
            self._current_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            if path.suffix == ".html":
                self._current_fig.write_html(str(path))
            else:
                self._current_fig.write_image(str(path))

    def save_all(
        self,
        output_dir: Union[str, Path],
        prefix: str = "optimization",
        format: str = "png",
    ) -> List[str]:
        """Save all available plots.

        Args:
            output_dir: Directory to save plots.
            prefix: Filename prefix.
            format: Output format ('png', 'svg', 'html' for plotly).

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []

        # Score history
        try:
            self.plot_score_history()
            path = output_dir / f"{prefix}_score_history.{format}"
            self.save(str(path))
            saved.append(str(path))
        except Exception as e:
            logger.debug(f"Could not save score history: {e}")

        # Distribution
        try:
            self.plot_score_distribution()
            path = output_dir / f"{prefix}_score_distribution.{format}"
            self.save(str(path))
            saved.append(str(path))
        except Exception as e:
            logger.debug(f"Could not save score distribution: {e}")

        # Improvement rate
        try:
            self.plot_improvement_rate()
            path = output_dir / f"{prefix}_improvement_rate.{format}"
            self.save(str(path))
            saved.append(str(path))
        except Exception as e:
            logger.debug(f"Could not save improvement rate: {e}")

        # Generation summary
        try:
            self.plot_generation_summary()
            path = output_dir / f"{prefix}_generation_summary.{format}"
            self.save(str(path))
            saved.append(str(path))
        except Exception as e:
            logger.debug(f"Could not save generation summary: {e}")

        return saved

    def show(self) -> None:
        """Display current figure (interactive environments)."""
        if self._current_fig is None:
            raise ValueError("No figure to show. Create a plot first.")

        if self.backend == "matplotlib":
            plt.show()
        else:
            self._current_fig.show()

    def clear(self) -> None:
        """Clear all logged data."""
        self._data = []
        self._current_fig = None

    @property
    def num_points(self) -> int:
        """Get the number of logged data points."""
        return len(self._data)

    def get_data(self) -> List[PlotData]:
        """Get all logged data points.

        Returns:
            List of PlotData objects.
        """
        return self._data.copy()

    def from_exporter(self, exporter: "ExperimentExporter") -> None:
        """Load data from an ExperimentExporter.

        Args:
            exporter: ExperimentExporter with recorded data.
        """
        from msgflux.optim.export import ExperimentExporter

        for step_record in exporter.record.steps:
            self.log(
                step=step_record.step,
                score=step_record.score,
                best_score=step_record.best_score,
                **step_record.metadata,
            )
