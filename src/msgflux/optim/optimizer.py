"""Minimal optimizer base used by trainer utilities.

This is intentionally lightweight and torch-free. It provides the small
surface area used by ``msgflux.trainer.Trainer`` and local tests:
``step()``, ``zero_grad()``, ``state_dict()``, and ``load_state_dict()``.
"""

from typing import Any, Iterable


class Optimizer:
    """Small optimizer base compatible with msgflux trainer utilities."""

    def __init__(
        self,
        params: Iterable[Any],
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.params = list(params)
        self.defaults = defaults or {}
        self.state: dict[str, Any] = {}
        self._step_count = 0

    def step(self, closure: Any = None) -> Any:  # pragma: no cover - abstract hook
        """Perform one optimization step."""
        raise NotImplementedError

    def zero_grad(self) -> None:
        """No-op default to mirror torch-like optimizers."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable optimizer state."""
        return {
            "defaults": self.defaults,
            "state": self.state,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state from a dictionary."""
        self.defaults = state_dict.get("defaults", {})
        self.state = state_dict.get("state", {})
        self._step_count = state_dict.get("_step_count", 0)
