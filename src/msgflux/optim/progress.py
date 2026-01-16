"""Progress tracking and logging utilities for optimizers.

This module provides utilities for displaying progress during optimization,
including progress bars, colored output, and structured logging.

Usage:
    from msgflux.optim.progress import OptimProgress, Colors

    progress = OptimProgress(verbose=True)
    progress.step("BOOTSTRAP EXAMPLES", 1, 3)

    with progress.iterate(trainset, desc="Processing") as pbar:
        for example in pbar:
            # process example
            pbar.set_postfix(score=0.85)
"""

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, TypeVar

from msgflux.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

# Check if tqdm is available
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None


class Colors:
    """ANSI color codes for terminal output."""

    # Basic colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Reset
    RESET = "\033[0m"
    ENDC = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY output)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")

    @classmethod
    def is_tty(cls) -> bool:
        """Check if stdout is a TTY."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Auto-disable colors if not TTY
if not Colors.is_tty():
    Colors.disable()


@dataclass
class TrialInfo:
    """Information about an optimization trial."""

    trial_num: int
    total_trials: int
    score: Optional[float] = None
    best_score: Optional[float] = None
    params: Optional[Dict[str, Any]] = None
    is_best: bool = False


@dataclass
class StepInfo:
    """Information about an optimization step."""

    step_num: int
    total_steps: int
    name: str
    description: Optional[str] = None


class SimpleProgress:
    """Simple progress indicator without tqdm dependency."""

    def __init__(
        self,
        iterable: Iterable[T],
        total: Optional[int] = None,
        desc: str = "",
        disable: bool = False,
    ):
        self.iterable = iterable
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None)
        self.desc = desc
        self.disable = disable
        self.n = 0
        self._postfix: Dict[str, Any] = {}

    def __iter__(self) -> Iterator[T]:
        if not self.disable and self.total:
            print(f"{self.desc}: 0/{self.total}", end="", flush=True)

        for item in self.iterable:
            yield item
            self.n += 1
            if not self.disable and self.total:
                postfix_str = ""
                if self._postfix:
                    postfix_str = " | " + ", ".join(
                        f"{k}={v}" for k, v in self._postfix.items()
                    )
                print(
                    f"\r{self.desc}: {self.n}/{self.total}{postfix_str}",
                    end="",
                    flush=True,
                )

        if not self.disable:
            print()  # New line at end

    def set_postfix(self, **kwargs) -> None:
        """Set postfix values to display."""
        self._postfix.update(kwargs)

    def set_description(self, desc: str) -> None:
        """Set description."""
        self.desc = desc

    def update(self, n: int = 1) -> None:
        """Update progress by n."""
        self.n += n


def create_progress_bar(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
    disable: bool = False,
    **kwargs,
) -> Iterator[T]:
    """Create a progress bar iterator.

    Uses tqdm if available, otherwise falls back to SimpleProgress.

    Args:
        iterable: The iterable to wrap.
        total: Total number of items (optional).
        desc: Description to show.
        disable: If True, disable the progress bar.
        **kwargs: Additional arguments passed to tqdm.

    Returns:
        An iterator that displays progress.
    """
    if TQDM_AVAILABLE and not disable:
        return tqdm(
            iterable,
            total=total,
            desc=desc,
            disable=disable,
            dynamic_ncols=True,
            file=sys.stdout,
            **kwargs,
        )
    else:
        return SimpleProgress(iterable, total=total, desc=desc, disable=disable)


class OptimProgress:
    """Progress tracker for optimization runs.

    Provides structured logging and progress tracking for optimizers.

    Args:
        verbose: If True, show detailed progress information.
        show_progress_bar: If True, show progress bars during iteration.
        log_level: Logging level when verbose is True.

    Example:
        >>> progress = OptimProgress(verbose=True)
        >>> progress.start("MIPROv2 Optimization")
        >>> progress.step("BOOTSTRAP EXAMPLES", 1, 3)
        >>> for item in progress.iterate(trainset, desc="Bootstrapping"):
        ...     # process item
        >>> progress.trial(TrialInfo(1, 50, score=0.85))
        >>> progress.finish(best_score=0.92)
    """

    def __init__(
        self,
        verbose: bool = False,
        show_progress_bar: bool = True,
        log_level: int = logging.INFO,
    ):
        self.verbose = verbose
        self.show_progress_bar = show_progress_bar and TQDM_AVAILABLE
        self.log_level = log_level

        self._current_step: Optional[StepInfo] = None
        self._trials: List[TrialInfo] = []
        self._best_score: Optional[float] = None

    def start(self, optimizer_name: str, **config) -> None:
        """Log the start of optimization.

        Args:
            optimizer_name: Name of the optimizer.
            **config: Configuration parameters to display.
        """
        if not self.verbose:
            return

        header = f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}"
        title = f"{Colors.BOLD}{optimizer_name} Optimization{Colors.RESET}"

        logger.log(self.log_level, header)
        logger.log(self.log_level, title)

        if config:
            logger.log(self.log_level, f"{Colors.DIM}Configuration:{Colors.RESET}")
            for key, value in config.items():
                logger.log(self.log_level, f"  {key}: {value}")

        logger.log(self.log_level, header)

    def step(
        self,
        name: str,
        step_num: int,
        total_steps: int,
        description: Optional[str] = None,
    ) -> None:
        """Log the start of an optimization step.

        Args:
            name: Name of the step.
            step_num: Current step number (1-indexed).
            total_steps: Total number of steps.
            description: Optional description of the step.
        """
        self._current_step = StepInfo(
            step_num=step_num,
            total_steps=total_steps,
            name=name,
            description=description,
        )

        if not self.verbose:
            return

        step_header = (
            f"\n{Colors.BOLD}{Colors.YELLOW}"
            f"==> STEP {step_num}/{total_steps}: {name} <=={Colors.RESET}"
        )
        logger.log(self.log_level, step_header)

        if description:
            logger.log(self.log_level, f"{Colors.DIM}{description}{Colors.RESET}")

    def substep(self, message: str) -> None:
        """Log a substep message.

        Args:
            message: The substep message.
        """
        if not self.verbose:
            return

        logger.log(self.log_level, f"  {Colors.CYAN}→{Colors.RESET} {message}")

    def trial(self, info: TrialInfo) -> None:
        """Log trial information.

        Args:
            info: Trial information.
        """
        self._trials.append(info)

        if info.is_best or (info.best_score and info.score == info.best_score):
            self._best_score = info.score

        if not self.verbose:
            return

        # Format trial header
        trial_header = (
            f"{Colors.BOLD}== Trial {info.trial_num}/{info.total_trials} =={Colors.RESET}"
        )
        logger.log(self.log_level, trial_header)

        # Format score
        if info.score is not None:
            score_color = Colors.GREEN if info.is_best else Colors.RESET
            score_msg = f"Score: {score_color}{info.score:.4f}{Colors.RESET}"

            if info.best_score is not None:
                score_msg += f" (Best: {info.best_score:.4f})"

            logger.log(self.log_level, score_msg)

        # Format parameters
        if info.params:
            params_str = ", ".join(f"{k}={v}" for k, v in info.params.items())
            logger.log(
                self.log_level, f"{Colors.DIM}Parameters: {params_str}{Colors.RESET}"
            )

        # Highlight new best
        if info.is_best:
            logger.log(
                self.log_level,
                f"{Colors.BOLD}{Colors.GREEN}★ New best score!{Colors.RESET}",
            )

    def metric(
        self,
        name: str,
        value: float,
        total: Optional[float] = None,
        is_percentage: bool = True,
    ) -> None:
        """Log a metric value.

        Args:
            name: Name of the metric.
            value: Metric value.
            total: Total value for percentage calculation.
            is_percentage: If True, display as percentage.
        """
        if not self.verbose:
            return

        if is_percentage and total:
            pct = 100 * value / total if total else 0
            msg = f"{name}: {value:.2f} / {total:.0f} ({pct:.1f}%)"
        elif is_percentage:
            msg = f"{name}: {value:.2%}"
        else:
            msg = f"{name}: {value:.4f}"

        logger.log(self.log_level, msg)

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: The message to log.
        """
        if not self.verbose:
            return

        logger.log(self.log_level, message)

    def success(self, message: str) -> None:
        """Log a success message.

        Args:
            message: The success message.
        """
        if not self.verbose:
            return

        logger.log(
            self.log_level, f"{Colors.GREEN}{Colors.BOLD}✓{Colors.RESET} {message}"
        )

    def warning(self, message: str) -> None:
        """Log a warning message (always shown).

        Args:
            message: The warning message.
        """
        logger.warning(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")

    def error(self, message: str) -> None:
        """Log an error message (always shown).

        Args:
            message: The error message.
        """
        logger.error(f"{Colors.RED}✗{Colors.RESET} {message}")

    def iterate(
        self,
        iterable: Iterable[T],
        desc: str = "",
        total: Optional[int] = None,
        **kwargs,
    ) -> Iterator[T]:
        """Create a progress bar for iteration.

        Args:
            iterable: The iterable to wrap.
            desc: Description to show.
            total: Total number of items.
            **kwargs: Additional arguments for progress bar.

        Returns:
            An iterator with progress tracking.
        """
        disable = not (self.verbose and self.show_progress_bar)
        return create_progress_bar(
            iterable, total=total, desc=desc, disable=disable, **kwargs
        )

    def finish(
        self,
        best_score: Optional[float] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the end of optimization.

        Args:
            best_score: The best score achieved.
            summary: Optional summary statistics.
        """
        if not self.verbose:
            return

        footer = f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}"
        logger.log(self.log_level, footer)

        if best_score is not None:
            logger.log(
                self.log_level,
                f"{Colors.BOLD}Final Score: {Colors.GREEN}{best_score:.4f}{Colors.RESET}",
            )
        elif self._best_score is not None:
            logger.log(
                self.log_level,
                f"{Colors.BOLD}Final Score: {Colors.GREEN}{self._best_score:.4f}{Colors.RESET}",
            )

        if summary:
            logger.log(self.log_level, f"{Colors.DIM}Summary:{Colors.RESET}")
            for key, value in summary.items():
                logger.log(self.log_level, f"  {key}: {value}")

        logger.log(self.log_level, footer)

    @contextmanager
    def section(self, name: str):
        """Context manager for a named section.

        Args:
            name: Name of the section.

        Yields:
            None
        """
        if self.verbose:
            logger.log(self.log_level, f"\n{Colors.BOLD}[{name}]{Colors.RESET}")
        try:
            yield
        finally:
            pass


def format_score(score: float, best: Optional[float] = None, width: int = 6) -> str:
    """Format a score value with optional best comparison.

    Args:
        score: The score to format.
        best: Optional best score for comparison.
        width: Minimum width for formatting.

    Returns:
        Formatted score string with colors.
    """
    score_str = f"{score:.4f}"

    if best is not None and score >= best:
        return f"{Colors.GREEN}{Colors.BOLD}{score_str}{Colors.RESET}"
    return score_str


def format_progress(current: int, total: int, width: int = 0) -> str:
    """Format progress as 'current/total'.

    Args:
        current: Current position.
        total: Total count.
        width: Minimum width for numbers.

    Returns:
        Formatted progress string.
    """
    if width:
        return f"{current:{width}d}/{total:{width}d}"
    return f"{current}/{total}"


def format_percentage(value: float, total: float) -> str:
    """Format a value as percentage.

    Args:
        value: The value.
        total: The total.

    Returns:
        Formatted percentage string.
    """
    if total <= 0:
        return "0.0%"
    pct = 100 * value / total
    return f"{pct:.1f}%"
