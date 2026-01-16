"""LabeledFewShot optimizer for selecting few-shot examples.

This module provides the simplest prompt optimizer - it samples
examples from a training set and uses them as demonstrations.
"""

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from msgflux.examples import Example, ExampleCollection
from msgflux.generation.templates import PromptSpec
from msgflux.logger import init_logger
from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer
from msgflux.optim.progress import OptimProgress

logger = init_logger(__name__)


class LabeledFewShot(Optimizer):
    """Optimizer that selects labeled examples from a trainset as demonstrations.

    This is the simplest optimizer - it randomly samples k examples from the
    training set and assigns them as demonstrations to the module's example
    parameters.

    Args:
        params: An iterable of Parameters to optimize.
        trainset: List of Example objects to sample from.
        k: Number of examples to select. Defaults to 16.
        sample: If True, randomly sample examples. If False, take first k.
            Defaults to True.
        seed: Random seed for reproducibility. Defaults to 0.

    Example:
        >>> agent = Agent(name="qa", model=model)
        >>> trainset = [
        ...     Example(inputs="What is 2+2?", labels="4"),
        ...     Example(inputs="What is 3+3?", labels="6"),
        ...     # ... more examples
        ... ]
        >>> optimizer = LabeledFewShot(
        ...     agent.parameters(),
        ...     trainset=trainset,
        ...     k=4,
        ... )
        >>> optimizer.step()  # Select 4 examples as demos
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        trainset: List[Example],
        k: int = 16,
        sample: bool = True,
        verbose: bool = False,
        seed: int = 0,
    ):
        defaults = dict(k=k, sample=sample, seed=seed)
        super().__init__(params, defaults)

        if not trainset:
            raise ValueError("trainset cannot be empty")

        self.trainset = trainset
        self.k = k
        self.sample = sample
        self.verbose = verbose
        self.seed = seed
        self.rng = random.Random(seed)

        # Progress tracking
        self._progress = OptimProgress(verbose=verbose)

        # State for tracking selected examples
        self._selected_examples: List[Example] = []

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Select k examples and assign them to example parameters.

        Args:
            closure: Optional closure for computing loss (not typically used).

        Returns:
            Optional loss value if closure was provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Start progress tracking
        self._progress.start(
            "LabeledFewShot",
            trainset_size=len(self.trainset),
            k=self.k,
            sample=self.sample,
        )

        # Determine number of examples to select
        k = min(self.k, len(self.trainset))
        self._progress.step("SELECT EXAMPLES", 1, 2)

        # Select examples
        if self.sample:
            self._selected_examples = self.rng.sample(self.trainset, k)
            self._progress.substep(f"Randomly sampled {k} examples")
        else:
            self._selected_examples = self.trainset[:k]
            self._progress.substep(f"Selected first {k} examples")

        # Format demos
        formatted_demos = self._format_demos(self._selected_examples)

        # Update all example parameters
        self._progress.step("UPDATE PARAMETERS", 2, 2)
        params_updated = 0
        for group in self.param_groups:
            for param in group["params"]:
                if self._is_example_param(param) and param.requires_grad:
                    param.data = formatted_demos
                    params_updated += 1

                    # Store state for this parameter
                    self.state[param]["selected_indices"] = [
                        self.trainset.index(ex) for ex in self._selected_examples
                        if ex in self.trainset
                    ]

        self._progress.success(f"Updated {params_updated} parameter(s) with {k} demos")

        # Finish with summary
        self._progress.finish(
            summary={
                "selected_examples": k,
                "params_updated": params_updated,
            },
        )

        self._step_count += 1
        return loss

    def _is_example_param(self, param: Parameter) -> bool:
        """Check if a parameter is an examples parameter."""
        return param.spec == PromptSpec.EXAMPLES

    def _format_demos(self, examples: List[Example]) -> str:
        """Format examples into the demos string format.

        Args:
            examples: List of Example objects.

        Returns:
            Formatted string representation of examples.
        """
        collection = ExampleCollection(examples)
        return collection.get_formatted()

    def get_selected_examples(self) -> List[Example]:
        """Return the currently selected examples.

        Returns:
            List of Example objects that were selected in the last step.
        """
        return self._selected_examples.copy()

    def reseed(self, seed: int) -> None:
        """Reset the random number generator with a new seed.

        Args:
            seed: New random seed.
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state = super().state_dict()
        state["selected_examples"] = [
            {"inputs": ex.inputs, "labels": ex.labels, "reasoning": ex.reasoning}
            for ex in self._selected_examples
        ]
        state["seed"] = self.seed
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state."""
        super().load_state_dict(state_dict)

        # Restore selected examples
        if "selected_examples" in state_dict:
            self._selected_examples = [
                Example(**ex_dict) for ex_dict in state_dict["selected_examples"]
            ]

        if "seed" in state_dict:
            self.reseed(state_dict["seed"])
