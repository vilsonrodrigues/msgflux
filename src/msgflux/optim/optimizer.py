"""Base class for prompt optimizers.

This module provides the base `Optimizer` class that all prompt optimizers
inherit from. The interface is similar to PyTorch's `torch.optim.Optimizer`.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from msgflux.nn.parameter import Parameter


class Optimizer(ABC):
    """Base class for all prompt optimizers.

    Optimizers modify prompt components (Parameters) to improve model performance.
    This class provides a PyTorch-like interface for prompt optimization.

    Args:
        params: An iterable of Parameters to optimize.
        defaults: Default values for optimization options.

    Example:
        >>> agent = Agent(name="qa", model=model)
        >>> optimizer = LabeledFewShot(agent.parameters(), trainset=examples, k=16)
        >>> optimizer.step()  # Select demos and update parameters
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        defaults: Optional[Dict[str, Any]] = None,
    ):
        self.defaults = defaults or {}
        self.param_groups: List[Dict[str, Any]] = []
        self.state: Dict[Parameter, Dict[str, Any]] = defaultdict(dict)
        self._step_count: int = 0

        # Handle different input types
        if isinstance(params, Parameter):
            raise TypeError(
                "params argument should be an iterable of Parameters or dicts, "
                f"but got {type(params).__name__}"
            )

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # Check if params is a list of dicts (param groups) or Parameters
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
            "_step_count": self._step_count,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step.

        This method should be overridden by all subclasses.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional for most optimizers.

        Returns:
            Optional loss value if closure was provided.
        """
        raise NotImplementedError

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        """Reset gradients of all optimized Parameters.

        Args:
            set_to_none: If True, set gradients to None instead of empty string.
                This can reduce memory usage.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad = ""

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict.

        Returns:
            A dict containing:
                - state: Current state for each parameter
                - param_groups: Parameter groups with their options
                - step_count: Number of optimization steps taken
        """
        # Pack the state
        packed_state = {}
        for idx, (param, param_state) in enumerate(self.state.items()):
            packed_state[idx] = param_state

        # Pack param groups (without actual params, just indices)
        packed_groups = []
        param_to_idx = {id(p): i for group in self.param_groups for i, p in enumerate(group["params"])}

        for group in self.param_groups:
            packed_group = {k: v for k, v in group.items() if k != "params"}
            packed_group["params"] = [param_to_idx[id(p)] for p in group["params"]]
            packed_groups.append(packed_group)

        return {
            "state": packed_state,
            "param_groups": packed_groups,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state.

        Args:
            state_dict: Optimizer state dict as returned by state_dict().
        """
        # Reconstruct param mapping
        all_params = []
        for group in self.param_groups:
            all_params.extend(group["params"])

        # Load state
        self.state = defaultdict(dict)
        for idx, param_state in state_dict["state"].items():
            if int(idx) < len(all_params):
                self.state[all_params[int(idx)]] = param_state

        # Load param groups (keeping current params)
        for group, saved_group in zip(self.param_groups, state_dict["param_groups"]):
            for key, value in saved_group.items():
                if key != "params":
                    group[key] = value

        self._step_count = state_dict.get("step_count", 0)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer's param_groups.

        This can be useful for fine-grained control of optimization,
        such as different learning rates for different parameters.

        Args:
            param_group: Dict containing 'params' key with Parameters
                and optional optimization options.

        Raises:
            ValueError: If 'params' key is missing or a parameter
                appears in multiple groups.
        """
        if "params" not in param_group:
            raise ValueError("param group must contain 'params' key")

        params = param_group["params"]
        if isinstance(params, Parameter):
            param_group["params"] = [params]
        elif isinstance(params, (list, tuple)):
            param_group["params"] = list(params)
        else:
            param_group["params"] = list(params)

        # Check for duplicate parameters
        param_set = set()
        for group in self.param_groups:
            for p in group["params"]:
                param_set.add(id(p))

        for p in param_group["params"]:
            if id(p) in param_set:
                raise ValueError("some parameters appear in more than one parameter group")
            param_set.add(id(p))

        # Apply defaults
        for key, default_value in self.defaults.items():
            param_group.setdefault(key, default_value)

        self.param_groups.append(param_group)

    @property
    def step_count(self) -> int:
        """Return the number of optimization steps taken."""
        return self._step_count


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self) -> str:
        return "<required parameter>"


required = _RequiredParameter()
