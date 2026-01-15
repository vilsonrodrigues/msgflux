"""BootstrapFewShot optimizer for generating high-quality demonstrations.

This module provides an optimizer that uses bootstrapping to generate
demonstrations from successful executions, similar to DSPy's BootstrapFewShot.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from msgflux.examples import Example, ExampleCollection
from msgflux.generation.templates import PromptSpec
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class Trace:
    """Represents an execution trace from a module."""

    module_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    augmented: bool = True


@dataclass
class BootstrapResult:
    """Result of bootstrapping a single example."""

    example: Example
    prediction: Any
    traces: List[Trace] = field(default_factory=list)
    score: float = 0.0
    success: bool = False
    error: Optional[str] = None


class BootstrapFewShot(Optimizer):
    """Optimizer that uses bootstrapping to generate high-quality demonstrations.

    This optimizer executes a teacher module on the training set, collects
    traces from successful executions (where metric > threshold), and uses
    those traces as demonstrations for the student module.

    The key idea is to find examples where the model performs well, capture
    the intermediate reasoning/outputs, and use those as few-shot examples.

    Args:
        params: An iterable of Parameters to optimize.
        metric: Function that takes (example, prediction) and returns a score.
        metric_threshold: Minimum score to consider an execution successful.
            If None, any non-zero score is considered successful.
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations.
        max_labeled_demos: Maximum number of labeled (non-bootstrapped) demos.
        max_rounds: Number of bootstrap rounds to run.
        max_errors: Maximum errors before stopping.
        teacher_settings: Optional settings to apply to the teacher model.
        seed: Random seed for reproducibility.

    Example:
        >>> agent = Agent(name="qa", model=model)
        >>> optimizer = BootstrapFewShot(
        ...     agent.parameters(),
        ...     metric=lambda ex, pred: float(ex.labels.lower() in pred.lower()),
        ...     max_bootstrapped_demos=4,
        ...     max_labeled_demos=16,
        ... )
        >>> optimizer.step(trainset=train_examples, teacher=agent)
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        metric: Callable[[Example, Any], float],
        metric_threshold: Optional[float] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: int = 10,
        teacher_settings: Optional[Dict[str, Any]] = None,
        seed: int = 0,
    ):
        defaults = dict(
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=max_rounds,
        )
        super().__init__(params, defaults)

        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.teacher_settings = teacher_settings or {}
        self.seed = seed
        self.rng = random.Random(seed)

        # Internal state
        self._traces: Dict[str, List[Trace]] = {}
        self._bootstrapped_indices: set = set()
        self._error_count: int = 0
        self._bootstrap_results: List[BootstrapResult] = []

    def step(
        self,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Execute bootstrap and update demonstration parameters.

        Args:
            trainset: Training examples to bootstrap from.
            teacher: Teacher module to use for bootstrapping. If None,
                bootstrapping is skipped and only labeled demos are used.
            closure: Optional closure for computing loss.

        Returns:
            Optional loss value if closure was provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Reset state for new step
        self._traces.clear()
        self._bootstrapped_indices.clear()
        self._bootstrap_results.clear()
        self._error_count = 0

        # Bootstrap if teacher is provided
        if teacher is not None:
            self._bootstrap(trainset, teacher)

        # Update parameters with demos
        self._update_demos(trainset)

        self._step_count += 1
        return loss

    def _bootstrap(self, trainset: List[Example], teacher: Module) -> None:
        """Execute bootstrapping to collect traces from successful executions."""
        logger.info(f"Starting bootstrap with {len(trainset)} examples")

        for round_idx in range(self.max_rounds):
            logger.debug(f"Bootstrap round {round_idx + 1}/{self.max_rounds}")

            for idx, example in enumerate(trainset):
                # Stop if we have enough bootstrapped examples
                if len(self._bootstrapped_indices) >= self.max_bootstrapped_demos:
                    logger.info(
                        f"Reached max bootstrapped demos ({self.max_bootstrapped_demos})"
                    )
                    return

                # Skip already bootstrapped examples
                if idx in self._bootstrapped_indices:
                    continue

                # Bootstrap this example
                result = self._bootstrap_one(example, teacher, round_idx)
                self._bootstrap_results.append(result)

                if result.success:
                    self._bootstrapped_indices.add(idx)
                    self._collect_traces(result)
                    logger.debug(f"Successfully bootstrapped example {idx}")

        logger.info(
            f"Bootstrap complete: {len(self._bootstrapped_indices)} successful, "
            f"{self._error_count} errors"
        )

    def _bootstrap_one(
        self,
        example: Example,
        teacher: Module,
        round_idx: int,
    ) -> BootstrapResult:
        """Bootstrap a single example by executing the teacher."""
        traces: List[Trace] = []

        try:
            # Create a hook to collect traces
            def trace_hook(module, args, kwargs, output):
                # Extract module name
                module_name = getattr(module, "_name", None) or module.__class__.__name__

                # Extract inputs from args/kwargs
                inputs = {}
                if args:
                    inputs["input"] = args[0] if len(args) == 1 else args
                inputs.update(kwargs)

                # Extract outputs
                if isinstance(output, dict):
                    outputs = output
                elif isinstance(output, str):
                    outputs = {"output": output}
                else:
                    outputs = {"output": str(output)}

                traces.append(
                    Trace(
                        module_name=module_name,
                        inputs=inputs,
                        outputs=outputs,
                        augmented=True,
                    )
                )

            # Register hooks on all submodules
            handles = []
            for name, module in teacher.named_modules():
                if hasattr(module, "register_forward_hook"):
                    try:
                        h = module.register_forward_hook(trace_hook)
                        handles.append(h)
                    except Exception:
                        pass  # Some modules may not support hooks

            # Execute the teacher
            prediction = teacher(example.inputs)

            # Remove hooks
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

            # Evaluate with metric
            score = self.metric(example, prediction)
            success = self._is_success(score)

            return BootstrapResult(
                example=example,
                prediction=prediction,
                traces=traces,
                score=score,
                success=success,
            )

        except Exception as e:
            self._error_count += 1
            logger.warning(f"Bootstrap error: {e}")

            if self._error_count >= self.max_errors:
                raise RuntimeError(
                    f"Maximum errors ({self.max_errors}) reached during bootstrap"
                ) from e

            return BootstrapResult(
                example=example,
                prediction=None,
                traces=[],
                score=0.0,
                success=False,
                error=str(e),
            )

    def _is_success(self, score: float) -> bool:
        """Check if a score indicates success."""
        if self.metric_threshold is not None:
            return score >= self.metric_threshold
        return bool(score)

    def _collect_traces(self, result: BootstrapResult) -> None:
        """Collect traces from a successful bootstrap result."""
        for trace in result.traces:
            if trace.module_name not in self._traces:
                self._traces[trace.module_name] = []
            self._traces[trace.module_name].append(trace)

    def _update_demos(self, trainset: List[Example]) -> None:
        """Update demonstration parameters with bootstrapped and labeled demos."""
        # Convert bootstrapped traces to examples
        bootstrapped_demos = self._traces_to_examples()

        # Get non-bootstrapped examples for labeled demos
        non_bootstrapped = [
            trainset[i]
            for i in range(len(trainset))
            if i not in self._bootstrapped_indices
        ]
        self.rng.shuffle(non_bootstrapped)

        # Calculate how many of each type to use
        num_bootstrapped = min(len(bootstrapped_demos), self.max_bootstrapped_demos)
        num_labeled = min(
            len(non_bootstrapped), self.max_labeled_demos - num_bootstrapped
        )

        # Combine demos
        all_demos = bootstrapped_demos[:num_bootstrapped]
        all_demos.extend(non_bootstrapped[:num_labeled])

        # Format demos
        formatted_demos = self._format_demos(all_demos)

        # Update all example parameters
        for group in self.param_groups:
            for param in group["params"]:
                if self._is_example_param(param) and param.requires_grad:
                    param.data = formatted_demos

                    # Store state
                    self.state[param] = {
                        "num_bootstrapped": num_bootstrapped,
                        "num_labeled": num_labeled,
                        "total_demos": len(all_demos),
                    }

        logger.info(
            f"Updated demos: {num_bootstrapped} bootstrapped, {num_labeled} labeled"
        )

    def _traces_to_examples(self) -> List[Example]:
        """Convert collected traces to Example objects."""
        examples = []

        for module_name, traces in self._traces.items():
            for trace in traces[: self.max_bootstrapped_demos]:
                # Create example from trace
                example = Example(
                    inputs=trace.inputs.get("input", trace.inputs),
                    labels=trace.outputs.get("output", trace.outputs),
                )
                examples.append(example)

        return examples

    def _is_example_param(self, param: Parameter) -> bool:
        """Check if a parameter is an examples parameter."""
        return param.spec == PromptSpec.EXAMPLES

    def _format_demos(self, examples: List[Example]) -> str:
        """Format examples into the demos string format."""
        if not examples:
            return ""
        collection = ExampleCollection(examples)
        return collection.get_formatted()

    def get_bootstrap_results(self) -> List[BootstrapResult]:
        """Return the bootstrap results from the last step."""
        return self._bootstrap_results.copy()

    def get_success_rate(self) -> float:
        """Calculate the success rate of bootstrapping."""
        if not self._bootstrap_results:
            return 0.0
        successes = sum(1 for r in self._bootstrap_results if r.success)
        return successes / len(self._bootstrap_results)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state = super().state_dict()
        state["bootstrapped_indices"] = list(self._bootstrapped_indices)
        state["error_count"] = self._error_count
        state["seed"] = self.seed
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state."""
        super().load_state_dict(state_dict)

        if "bootstrapped_indices" in state_dict:
            self._bootstrapped_indices = set(state_dict["bootstrapped_indices"])
        if "error_count" in state_dict:
            self._error_count = state_dict["error_count"]
        if "seed" in state_dict:
            self.seed = state_dict["seed"]
            self.rng = random.Random(self.seed)
