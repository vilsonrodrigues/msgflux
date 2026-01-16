"""BootstrapFewShot optimizer for generating high-quality demonstrations.

This module provides an optimizer that uses bootstrapping to generate
demonstrations from successful executions, similar to DSPy's BootstrapFewShot.
Supports both synchronous and asynchronous execution.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from msgflux.examples import Example, ExampleCollection
from msgflux.generation.templates import PromptSpec
from msgflux.logger import init_logger
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer
from msgflux.optim.progress import OptimProgress, TrialInfo

logger = init_logger(__name__)


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
        verbose: bool = False,
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
        self.verbose = verbose
        self.seed = seed
        self.rng = random.Random(seed)

        # Progress tracking
        self._progress = OptimProgress(verbose=verbose)

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

        # Start progress tracking
        self._progress.start(
            "BootstrapFewShot",
            trainset_size=len(trainset),
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
        )

        # Reset state for new step
        self._traces.clear()
        self._bootstrapped_indices.clear()
        self._bootstrap_results.clear()
        self._error_count = 0

        # Bootstrap if teacher is provided
        if teacher is not None:
            self._progress.step("BOOTSTRAP EXAMPLES", 1, 2)
            self._bootstrap(trainset, teacher)
        else:
            self._progress.step("SKIP BOOTSTRAP (no teacher)", 1, 2)

        # Update parameters with demos
        self._progress.step("UPDATE DEMONSTRATIONS", 2, 2)
        self._update_demos(trainset)

        # Finish with summary
        success_rate = self.get_success_rate()
        self._progress.finish(
            best_score=success_rate,
            summary={
                "bootstrapped": len(self._bootstrapped_indices),
                "errors": self._error_count,
                "success_rate": f"{success_rate:.1%}",
            },
        )

        self._step_count += 1
        return loss

    def _bootstrap(self, trainset: List[Example], teacher: Module) -> None:
        """Execute bootstrapping to collect traces from successful executions."""
        self._progress.substep(f"Processing {len(trainset)} examples")

        for round_idx in range(self.max_rounds):
            self._progress.substep(f"Round {round_idx + 1}/{self.max_rounds}")

            for idx, example in enumerate(trainset):
                # Stop if we have enough bootstrapped examples
                if len(self._bootstrapped_indices) >= self.max_bootstrapped_demos:
                    self._progress.success(
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

        self._progress.success(
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
            self._progress.warning(f"Bootstrap error: {e}")

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

        self._progress.success(
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

    # Async methods

    async def astep(
        self,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> Optional[float]:
        """Execute bootstrap asynchronously and update demonstration parameters.

        This method evaluates examples concurrently using asyncio, which is
        more efficient for I/O-bound operations like API calls.

        Args:
            trainset: Training examples to bootstrap from.
            teacher: Teacher module to use for bootstrapping.
            closure: Optional closure for computing loss.
            max_concurrency: Maximum number of concurrent evaluations.
                If None, evaluates all examples concurrently.

        Returns:
            Optional loss value if closure was provided.

        Example:
            >>> result = await optimizer.astep(trainset, teacher=agent)
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Start progress tracking
        self._progress.start(
            "BootstrapFewShot (async)",
            trainset_size=len(trainset),
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            max_rounds=self.max_rounds,
            max_concurrency=max_concurrency or "unlimited",
        )

        # Reset state for new step
        self._traces.clear()
        self._bootstrapped_indices.clear()
        self._bootstrap_results.clear()
        self._error_count = 0

        # Bootstrap if teacher is provided
        if teacher is not None:
            self._progress.step("BOOTSTRAP EXAMPLES (async)", 1, 2)
            await self._abootstrap(trainset, teacher, max_concurrency)
        else:
            self._progress.step("SKIP BOOTSTRAP (no teacher)", 1, 2)

        # Update parameters with demos
        self._progress.step("UPDATE DEMONSTRATIONS", 2, 2)
        self._update_demos(trainset)

        # Finish with summary
        success_rate = self.get_success_rate()
        self._progress.finish(
            best_score=success_rate,
            summary={
                "bootstrapped": len(self._bootstrapped_indices),
                "errors": self._error_count,
                "success_rate": f"{success_rate:.1%}",
            },
        )

        self._step_count += 1
        return loss

    async def _abootstrap(
        self,
        trainset: List[Example],
        teacher: Module,
        max_concurrency: Optional[int] = None,
    ) -> None:
        """Execute bootstrapping asynchronously to collect traces."""
        self._progress.substep(f"Processing {len(trainset)} examples")

        for round_idx in range(self.max_rounds):
            self._progress.substep(f"Round {round_idx + 1}/{self.max_rounds}")

            # Get examples to process this round
            examples_to_process = [
                (idx, example)
                for idx, example in enumerate(trainset)
                if idx not in self._bootstrapped_indices
            ]

            # Limit to remaining needed
            remaining_needed = self.max_bootstrapped_demos - len(self._bootstrapped_indices)
            if remaining_needed <= 0:
                break

            examples_to_process = examples_to_process[:remaining_needed * 2]

            if max_concurrency:
                results = await self._abootstrap_with_semaphore(
                    examples_to_process, teacher, round_idx, max_concurrency
                )
            else:
                results = await self._abootstrap_concurrent(
                    examples_to_process, teacher, round_idx
                )

            # Process results
            for idx, result in results:
                self._bootstrap_results.append(result)
                if result.success:
                    self._bootstrapped_indices.add(idx)
                    self._collect_traces(result)

                    if len(self._bootstrapped_indices) >= self.max_bootstrapped_demos:
                        self._progress.success(
                            f"Reached max bootstrapped demos ({self.max_bootstrapped_demos})"
                        )
                        return

        self._progress.success(
            f"Async bootstrap complete: {len(self._bootstrapped_indices)} successful, "
            f"{self._error_count} errors"
        )

    async def _abootstrap_concurrent(
        self,
        examples_with_idx: List[Tuple[int, Example]],
        teacher: Module,
        round_idx: int,
    ) -> List[Tuple[int, BootstrapResult]]:
        """Bootstrap examples concurrently."""
        tasks = [
            self._abootstrap_one(idx, example, teacher, round_idx)
            for idx, example in examples_with_idx
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            idx = examples_with_idx[i][0]
            if isinstance(result, Exception):
                self._progress.error(f"Async bootstrap error: {result}")
                self._error_count += 1
                final_results.append((idx, BootstrapResult(
                    example=examples_with_idx[i][1],
                    prediction=None,
                    traces=[],
                    score=0.0,
                    success=False,
                    error=str(result),
                )))
            else:
                final_results.append(result)

        return final_results

    async def _abootstrap_with_semaphore(
        self,
        examples_with_idx: List[Tuple[int, Example]],
        teacher: Module,
        round_idx: int,
        max_concurrency: int,
    ) -> List[Tuple[int, BootstrapResult]]:
        """Bootstrap examples with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_bootstrap(idx: int, example: Example):
            async with semaphore:
                return await self._abootstrap_one(idx, example, teacher, round_idx)

        tasks = [bounded_bootstrap(idx, ex) for idx, ex in examples_with_idx]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            idx = examples_with_idx[i][0]
            if isinstance(result, Exception):
                self._progress.error(f"Async bootstrap error: {result}")
                self._error_count += 1
                final_results.append((idx, BootstrapResult(
                    example=examples_with_idx[i][1],
                    prediction=None,
                    traces=[],
                    score=0.0,
                    success=False,
                    error=str(result),
                )))
            else:
                final_results.append(result)

        return final_results

    async def _abootstrap_one(
        self,
        idx: int,
        example: Example,
        teacher: Module,
        round_idx: int,
    ) -> Tuple[int, BootstrapResult]:
        """Bootstrap a single example asynchronously."""
        traces: List[Trace] = []

        try:
            # Create a hook to collect traces
            def trace_hook(module, args, kwargs, output):
                module_name = getattr(module, "_name", None) or module.__class__.__name__
                inputs = {}
                if args:
                    inputs["input"] = args[0] if len(args) == 1 else args
                inputs.update(kwargs)

                if isinstance(output, dict):
                    outputs = output
                elif isinstance(output, str):
                    outputs = {"output": output}
                else:
                    outputs = {"output": str(output)}

                traces.append(Trace(
                    module_name=module_name,
                    inputs=inputs,
                    outputs=outputs,
                    augmented=True,
                ))

            # Register hooks
            handles = []
            for name, module in teacher.named_modules():
                if hasattr(module, "register_forward_hook"):
                    try:
                        h = module.register_forward_hook(trace_hook)
                        handles.append(h)
                    except Exception:
                        pass

            # Execute the teacher asynchronously
            if hasattr(teacher, "acall"):
                prediction = await teacher.acall(example.inputs)
            elif hasattr(teacher, "aforward"):
                prediction = await teacher.aforward(example.inputs)
            else:
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None, teacher, example.inputs
                )

            # Remove hooks
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

            # Evaluate with metric
            score = self.metric(example, prediction)
            success = self._is_success(score)

            return (idx, BootstrapResult(
                example=example,
                prediction=prediction,
                traces=traces,
                score=score,
                success=success,
            ))

        except Exception as e:
            self._error_count += 1
            self._progress.warning(f"Async bootstrap error: {e}")

            if self._error_count >= self.max_errors:
                raise RuntimeError(
                    f"Maximum errors ({self.max_errors}) reached during bootstrap"
                ) from e

            return (idx, BootstrapResult(
                example=example,
                prediction=None,
                traces=[],
                score=0.0,
                success=False,
                error=str(e),
            ))
