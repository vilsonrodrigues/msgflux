"""Evaluate — run a program against a dataset and aggregate scores."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""

    score: float
    results: List[Tuple[Any, Any, float]] = field(default_factory=list)


class Evaluate:
    """Evaluate a program on a dataset using a metric function.

    Example::

        evaluator = Evaluate(devset=examples, metric=exact_match)
        result = evaluator(program)
        print(result.score)

    Args:
        devset: Default evaluation dataset.
        metric: ``metric(example, prediction) -> float | bool``
        num_threads: Number of threads for parallel evaluation.
        max_errors: Maximum errors before aborting.
        display_progress: Show a tqdm progress bar.
        failure_score: Score assigned when a program raises an exception.
        return_all_scores: If *True*, populate ``result.results``.
    """

    def __init__(
        self,
        *,
        devset: List[Any],
        metric: Callable,
        num_threads: Optional[int] = None,
        max_errors: Optional[int] = None,
        display_progress: bool = True,
        failure_score: float = 0.0,
        return_all_scores: bool = False,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads or 1
        self.max_errors = max_errors
        self.display_progress = display_progress
        self.failure_score = failure_score
        self.return_all_scores = return_all_scores

    def __call__(
        self,
        program: Any,
        devset: Optional[List[Any]] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> EvalResult:
        """Run *program* on every example in *devset* and return the result."""
        devset = devset or self.devset
        if not devset:
            return EvalResult(score=0.0)

        if self.num_threads <= 1:
            results = self._run_sequential(program, devset)
        else:
            results = self._run_parallel(program, devset)

        total_score = sum(r[2] for r in results)
        avg_score = total_score / len(results) if results else 0.0

        return EvalResult(
            score=avg_score,
            results=results if self.return_all_scores else [],
        )

    def _evaluate_one(
        self, program: Any, example: Any
    ) -> Tuple[Any, Any, float]:
        try:
            inputs = example.inputs if hasattr(example, "inputs") else example
            prediction = program(inputs)
            score = self.metric(example, prediction)
            if not isinstance(score, (int, float)):
                score = float(score)
            return (example, prediction, score)
        except Exception:
            logger.error("Error evaluating example", exc_info=True)
            return (example, None, self.failure_score)

    def _run_sequential(
        self, program: Any, devset: List[Any]
    ) -> List[Tuple[Any, Any, float]]:
        results: List[Tuple[Any, Any, float]] = []
        error_count = 0
        for example in devset:
            result = self._evaluate_one(program, example)
            results.append(result)
            if result[1] is None:
                error_count += 1
            if self.max_errors and error_count >= self.max_errors:
                break
        return results

    def _run_parallel(
        self, program: Any, devset: List[Any]
    ) -> List[Tuple[Any, Any, float]]:
        results: List[Tuple[Any, Any, float]] = []
        error_count = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(self._evaluate_one, program, ex): ex
                for ex in devset
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result[1] is None:
                    error_count += 1
                if self.max_errors and error_count >= self.max_errors:
                    break
        return results
