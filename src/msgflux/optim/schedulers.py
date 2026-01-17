"""Temperature schedulers for optimizers.

This module provides schedulers for adapting parameters like temperature
during optimization.
"""

import math
from abc import ABC, abstractmethod
from typing import Optional


class TemperatureScheduler(ABC):
    """Base class for temperature schedulers.

    Schedulers provide a way to adapt parameters like temperature
    during optimization, enabling strategies like annealing.

    Example:
        >>> scheduler = LinearScheduler(start_value=1.0, end_value=0.1, num_steps=100)
        >>> for step in range(100):
        ...     temperature = scheduler.get_value(step)
        ...     # Use temperature in optimization
    """

    @abstractmethod
    def get_value(self, step: int) -> float:
        """Get the parameter value for the given step.

        Args:
            step: Current step number.

        Returns:
            The parameter value for this step.
        """
        pass

    def __call__(self, step: int) -> float:
        """Alias for get_value.

        Args:
            step: Current step number.

        Returns:
            The parameter value for this step.
        """
        return self.get_value(step)

    def state_dict(self) -> dict:
        """Return the state of the scheduler as a dict."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state from a dict."""
        pass


class ConstantScheduler(TemperatureScheduler):
    """Constant value scheduler (no scheduling).

    Always returns the same value regardless of step.

    Args:
        value: The constant value to return.

    Example:
        >>> scheduler = ConstantScheduler(1.0)
        >>> scheduler(0)  # Returns 1.0
        >>> scheduler(100)  # Returns 1.0
    """

    def __init__(self, value: float = 1.0):
        self.value = value

    def get_value(self, step: int) -> float:
        """Get the constant value."""
        return self.value

    def state_dict(self) -> dict:
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict) -> None:
        self.value = state_dict.get("value", self.value)


class LinearScheduler(TemperatureScheduler):
    """Linear interpolation between start and end values.

    Linearly interpolates from start_value to end_value over num_steps.
    After num_steps, returns end_value.

    Args:
        start_value: Initial value.
        end_value: Final value.
        num_steps: Number of steps for interpolation.

    Example:
        >>> scheduler = LinearScheduler(start_value=1.0, end_value=0.1, num_steps=100)
        >>> scheduler(0)  # Returns 1.0
        >>> scheduler(50)  # Returns 0.55
        >>> scheduler(100)  # Returns 0.1
    """

    def __init__(
        self,
        start_value: float = 1.0,
        end_value: float = 0.1,
        num_steps: int = 100,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

    def get_value(self, step: int) -> float:
        """Get the linearly interpolated value."""
        if step >= self.num_steps:
            return self.end_value

        if step <= 0:
            return self.start_value

        progress = step / self.num_steps
        return self.start_value + (self.end_value - self.start_value) * progress

    def state_dict(self) -> dict:
        return {
            "start_value": self.start_value,
            "end_value": self.end_value,
            "num_steps": self.num_steps,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.start_value = state_dict.get("start_value", self.start_value)
        self.end_value = state_dict.get("end_value", self.end_value)
        self.num_steps = state_dict.get("num_steps", self.num_steps)


class CosineScheduler(TemperatureScheduler):
    """Cosine annealing schedule.

    Uses cosine function for smooth annealing from max_value to min_value.
    Supports multiple cycles for warm restarts.

    Args:
        max_value: Maximum value (at start of each cycle).
        min_value: Minimum value (at end of each cycle).
        num_steps: Number of steps for one cycle.
        num_cycles: Number of cycles (default 1 for single annealing).

    Example:
        >>> scheduler = CosineScheduler(max_value=1.0, min_value=0.0, num_steps=100)
        >>> scheduler(0)  # Returns 1.0
        >>> scheduler(50)  # Returns 0.5
        >>> scheduler(100)  # Returns 0.0
    """

    def __init__(
        self,
        max_value: float = 1.0,
        min_value: float = 0.0,
        num_steps: int = 100,
        num_cycles: int = 1,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive")

        self.max_value = max_value
        self.min_value = min_value
        self.num_steps = num_steps
        self.num_cycles = num_cycles

    def get_value(self, step: int) -> float:
        """Get the cosine annealed value."""
        # Handle cycling
        if self.num_cycles > 1:
            step = step % self.num_steps

        if step >= self.num_steps:
            return self.min_value

        if step <= 0:
            return self.max_value

        progress = step / self.num_steps
        cosine_value = (1 + math.cos(math.pi * progress)) / 2
        return self.min_value + (self.max_value - self.min_value) * cosine_value

    def state_dict(self) -> dict:
        return {
            "max_value": self.max_value,
            "min_value": self.min_value,
            "num_steps": self.num_steps,
            "num_cycles": self.num_cycles,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.max_value = state_dict.get("max_value", self.max_value)
        self.min_value = state_dict.get("min_value", self.min_value)
        self.num_steps = state_dict.get("num_steps", self.num_steps)
        self.num_cycles = state_dict.get("num_cycles", self.num_cycles)


class ExponentialScheduler(TemperatureScheduler):
    """Exponential decay schedule.

    Decays the value exponentially by multiplying by decay_rate each step.

    Args:
        start_value: Initial value.
        decay_rate: Multiplicative decay factor per step (e.g., 0.99).
        min_value: Minimum value (floor).

    Example:
        >>> scheduler = ExponentialScheduler(start_value=1.0, decay_rate=0.99, min_value=0.01)
        >>> scheduler(0)  # Returns 1.0
        >>> scheduler(100)  # Returns ~0.366
        >>> scheduler(500)  # Returns min_value (0.01)
    """

    def __init__(
        self,
        start_value: float = 1.0,
        decay_rate: float = 0.99,
        min_value: float = 0.01,
    ):
        if decay_rate <= 0 or decay_rate > 1:
            raise ValueError("decay_rate must be in (0, 1]")

        self.start_value = start_value
        self.decay_rate = decay_rate
        self.min_value = min_value

    def get_value(self, step: int) -> float:
        """Get the exponentially decayed value."""
        if step <= 0:
            return self.start_value

        value = self.start_value * (self.decay_rate**step)
        return max(value, self.min_value)

    def state_dict(self) -> dict:
        return {
            "start_value": self.start_value,
            "decay_rate": self.decay_rate,
            "min_value": self.min_value,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.start_value = state_dict.get("start_value", self.start_value)
        self.decay_rate = state_dict.get("decay_rate", self.decay_rate)
        self.min_value = state_dict.get("min_value", self.min_value)


class StepScheduler(TemperatureScheduler):
    """Step-based decay schedule.

    Decays the value by a factor at specific step intervals.

    Args:
        start_value: Initial value.
        step_size: Number of steps between decays.
        gamma: Multiplicative factor for decay (e.g., 0.5 halves the value).
        min_value: Minimum value (floor).

    Example:
        >>> scheduler = StepScheduler(start_value=1.0, step_size=10, gamma=0.5)
        >>> scheduler(0)  # Returns 1.0
        >>> scheduler(9)  # Returns 1.0
        >>> scheduler(10)  # Returns 0.5
        >>> scheduler(20)  # Returns 0.25
    """

    def __init__(
        self,
        start_value: float = 1.0,
        step_size: int = 10,
        gamma: float = 0.5,
        min_value: float = 0.01,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be in (0, 1]")

        self.start_value = start_value
        self.step_size = step_size
        self.gamma = gamma
        self.min_value = min_value

    def get_value(self, step: int) -> float:
        """Get the step-decayed value."""
        if step <= 0:
            return self.start_value

        num_decays = step // self.step_size
        value = self.start_value * (self.gamma**num_decays)
        return max(value, self.min_value)

    def state_dict(self) -> dict:
        return {
            "start_value": self.start_value,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "min_value": self.min_value,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.start_value = state_dict.get("start_value", self.start_value)
        self.step_size = state_dict.get("step_size", self.step_size)
        self.gamma = state_dict.get("gamma", self.gamma)
        self.min_value = state_dict.get("min_value", self.min_value)


class WarmupScheduler(TemperatureScheduler):
    """Warmup followed by another scheduler.

    Linearly increases from warmup_start to the scheduler's initial value
    during the warmup phase, then follows the inner scheduler.

    Args:
        warmup_steps: Number of warmup steps.
        warmup_start: Starting value during warmup.
        scheduler: Scheduler to use after warmup. If None, uses constant value.

    Example:
        >>> inner = CosineScheduler(max_value=1.0, min_value=0.1, num_steps=100)
        >>> scheduler = WarmupScheduler(warmup_steps=10, warmup_start=0.0, scheduler=inner)
        >>> scheduler(0)  # Returns 0.0 (warmup start)
        >>> scheduler(5)  # Returns 0.5 (warmup)
        >>> scheduler(10)  # Returns 1.0 (end of warmup, start of cosine)
        >>> scheduler(60)  # Returns ~0.5 (cosine annealing)
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_start: float = 0.0,
        scheduler: Optional[TemperatureScheduler] = None,
    ):
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")

        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self.scheduler = scheduler or ConstantScheduler(1.0)
        self._warmup_end = self.scheduler.get_value(0)

    def get_value(self, step: int) -> float:
        """Get the value with warmup."""
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self._warmup_end
            # Linear warmup
            progress = step / self.warmup_steps
            return self.warmup_start + (self._warmup_end - self.warmup_start) * progress
        else:
            return self.scheduler.get_value(step - self.warmup_steps)

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "warmup_start": self.warmup_start,
            "scheduler_state": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.warmup_start = state_dict.get("warmup_start", self.warmup_start)
        if "scheduler_state" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_state"])
        self._warmup_end = self.scheduler.get_value(0)


class CyclicScheduler(TemperatureScheduler):
    """Cyclic scheduler with configurable cycle shape.

    Cycles between min_value and max_value using triangular or other patterns.

    Args:
        min_value: Minimum value in the cycle.
        max_value: Maximum value in the cycle.
        step_size: Number of steps to go from min to max (half cycle).
        mode: Cycle mode - 'triangular' (linear), 'triangular2' (halving amplitude),
            or 'exp_range' (exponential scaling).
        gamma: Scaling factor for 'triangular2' and 'exp_range' modes.

    Example:
        >>> scheduler = CyclicScheduler(min_value=0.1, max_value=1.0, step_size=10)
        >>> scheduler(0)  # Returns 0.1
        >>> scheduler(10)  # Returns 1.0
        >>> scheduler(20)  # Returns 0.1
    """

    def __init__(
        self,
        min_value: float = 0.1,
        max_value: float = 1.0,
        step_size: int = 10,
        mode: str = "triangular",
        gamma: float = 1.0,
    ):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if mode not in ("triangular", "triangular2", "exp_range"):
            raise ValueError(
                f"mode must be 'triangular', 'triangular2', or 'exp_range', got {mode}"
            )

        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def get_value(self, step: int) -> float:
        """Get the cyclic value."""
        cycle = math.floor(1 + step / (2 * self.step_size))
        x = abs(step / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1))
        else:  # exp_range
            scale = self.gamma ** (step)

        base = self.min_value
        amplitude = (self.max_value - self.min_value) * scale
        return base + amplitude * max(0, (1 - x))

    def state_dict(self) -> dict:
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step_size": self.step_size,
            "mode": self.mode,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.min_value = state_dict.get("min_value", self.min_value)
        self.max_value = state_dict.get("max_value", self.max_value)
        self.step_size = state_dict.get("step_size", self.step_size)
        self.mode = state_dict.get("mode", self.mode)
        self.gamma = state_dict.get("gamma", self.gamma)
