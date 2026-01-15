"""Evaluator for measuring module performance.

This module provides the Evaluator class for evaluating msgflux modules
on development/test sets using custom metrics. Supports both synchronous
and asynchronous evaluation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from msgflux.examples import Example
from msgflux.nn.modules.module import Module

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a module on a dataset.

    Attributes:
        score: Overall score as a percentage (0-100).
        results: List of (example, prediction, score) tuples.
        num_errors: Number of examples that failed during evaluation.
        metadata: Optional metadata about the evaluation.
    """

    score: float
    results: List[Tuple[Example, Any, float]] = field(default_factory=list)
    num_errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(score={self.score:.2f}%, "
            f"n={len(self.results)}, errors={self.num_errors})"
        )

    def get_successes(self) -> List[Tuple[Example, Any, float]]:
        """Return examples where the score was positive."""
        return [(ex, pred, score) for ex, pred, score in self.results if score > 0]

    def get_failures(self) -> List[Tuple[Example, Any, float]]:
        """Return examples where the score was zero or negative."""
        return [(ex, pred, score) for ex, pred, score in self.results if score <= 0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "score": self.score,
            "num_examples": len(self.results),
            "num_errors": self.num_errors,
            "num_successes": len(self.get_successes()),
            "num_failures": len(self.get_failures()),
            "metadata": self.metadata,
        }


class Evaluator:
    """Evaluator for measuring module performance on datasets.

    The Evaluator runs a module on each example in a dataset, applies
    a metric function to score the predictions, and aggregates results.

    Args:
        metric: Function that takes (example, prediction) and returns a score.
            The score should typically be in [0, 1] range.
        num_workers: Number of parallel workers. If None, uses sequential
            evaluation. Defaults to None.
        max_errors: Maximum number of errors before stopping. If None,
            uses global settings. Defaults to 10.
        failure_score: Score to assign when evaluation fails. Defaults to 0.0.
        verbose: Whether to log progress. Defaults to False.

    Example:
        >>> def exact_match(example, prediction):
        ...     return float(example.labels.lower() == prediction.lower())
        >>>
        >>> evaluator = Evaluator(metric=exact_match, verbose=True)
        >>> result = evaluator(agent, test_examples)
        >>> print(f"Score: {result.score:.2f}%")
    """

    def __init__(
        self,
        metric: Callable[[Example, Any], float],
        *,
        num_workers: Optional[int] = None,
        max_errors: int = 10,
        failure_score: float = 0.0,
        verbose: bool = False,
    ):
        self.metric = metric
        self.num_workers = num_workers
        self.max_errors = max_errors
        self.failure_score = failure_score
        self.verbose = verbose
        self._error_count = 0

    def __call__(
        self,
        module: Module,
        devset: List[Example],
        **kwargs,
    ) -> EvaluationResult:
        """Evaluate the module on the devset.

        Args:
            module: The module to evaluate.
            devset: List of Example objects to evaluate on.
            **kwargs: Additional arguments passed to evaluation.

        Returns:
            EvaluationResult with score and detailed results.
        """
        return self.evaluate(module, devset, **kwargs)

    def evaluate(
        self,
        module: Module,
        devset: List[Example],
        *,
        return_predictions: bool = True,
    ) -> EvaluationResult:
        """Run full evaluation on a dataset.

        Args:
            module: The module to evaluate.
            devset: List of Example objects to evaluate on.
            return_predictions: Whether to include predictions in results.

        Returns:
            EvaluationResult with score and detailed results.
        """
        if not devset:
            return EvaluationResult(score=0.0, results=[], num_errors=0)

        # Reset error count
        self._error_count = 0

        # Store original training mode
        was_training = module.training

        # Set to evaluation mode
        module.eval()

        try:
            if self.num_workers and self.num_workers > 1:
                results = self._evaluate_parallel(module, devset)
            else:
                results = self._evaluate_sequential(module, devset)
        finally:
            # Restore training mode
            if was_training:
                module.train()

        # Calculate overall score
        total_score = sum(score for _, _, score in results)
        score = (total_score / len(results)) * 100 if results else 0.0

        if self.verbose:
            logger.info(
                f"Evaluation complete: {score:.2f}% "
                f"({int(total_score)}/{len(results)} correct)"
            )

        return EvaluationResult(
            score=round(score, 2),
            results=results if return_predictions else [],
            num_errors=self._error_count,
            metadata={
                "num_examples": len(devset),
                "num_evaluated": len(results),
                "metric_name": getattr(self.metric, "__name__", "custom"),
            },
        )

    def _evaluate_sequential(
        self,
        module: Module,
        devset: List[Example],
    ) -> List[Tuple[Example, Any, float]]:
        """Evaluate examples sequentially."""
        results = []

        for idx, example in enumerate(devset):
            if self.verbose and idx % 10 == 0:
                logger.info(f"Evaluating example {idx + 1}/{len(devset)}")

            result = self._evaluate_one(module, example)
            results.append(result)

        return results

    def _evaluate_parallel(
        self,
        module: Module,
        devset: List[Example],
    ) -> List[Tuple[Example, Any, float]]:
        """Evaluate examples in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(devset)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._evaluate_one, module, example): idx
                for idx, example in enumerate(devset)
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Parallel evaluation error: {e}")
                    results[idx] = (devset[idx], None, self.failure_score)

        return [r for r in results if r is not None]

    def _evaluate_one(
        self,
        module: Module,
        example: Example,
    ) -> Tuple[Example, Any, float]:
        """Evaluate a single example."""
        try:
            # Run the module
            prediction = module(example.inputs)

            # Score with metric
            score = self.metric(example, prediction)

            return (example, prediction, score)

        except Exception as e:
            self._error_count += 1

            if self.verbose:
                logger.warning(f"Evaluation error: {e}")

            if self._error_count >= self.max_errors:
                raise RuntimeError(
                    f"Maximum errors ({self.max_errors}) reached during evaluation"
                ) from e

            return (example, None, self.failure_score)

    def evaluate_single(
        self,
        module: Module,
        example: Example,
    ) -> Tuple[Any, float]:
        """Evaluate a single example and return prediction and score.

        Args:
            module: The module to evaluate.
            example: The example to evaluate.

        Returns:
            Tuple of (prediction, score).
        """
        _, prediction, score = self._evaluate_one(module, example)
        return prediction, score

    # Async methods

    async def aevaluate(
        self,
        module: Module,
        devset: List[Example],
        *,
        return_predictions: bool = True,
        max_concurrency: Optional[int] = None,
    ) -> EvaluationResult:
        """Run evaluation asynchronously on a dataset.

        This method evaluates examples concurrently using asyncio, which is
        more efficient for I/O-bound operations like API calls.

        Args:
            module: The module to evaluate (must support aforward).
            devset: List of Example objects to evaluate on.
            return_predictions: Whether to include predictions in results.
            max_concurrency: Maximum number of concurrent evaluations.
                If None, evaluates all examples concurrently.

        Returns:
            EvaluationResult with score and detailed results.

        Example:
            >>> result = await evaluator.aevaluate(agent, test_examples)
            >>> print(f"Score: {result.score:.2f}%")
        """
        if not devset:
            return EvaluationResult(score=0.0, results=[], num_errors=0)

        # Reset error count
        self._error_count = 0

        # Store original training mode
        was_training = module.training

        # Set to evaluation mode
        module.eval()

        try:
            if max_concurrency:
                results = await self._aevaluate_with_semaphore(
                    module, devset, max_concurrency
                )
            else:
                results = await self._aevaluate_concurrent(module, devset)
        finally:
            # Restore training mode
            if was_training:
                module.train()

        # Calculate overall score
        total_score = sum(score for _, _, score in results)
        score = (total_score / len(results)) * 100 if results else 0.0

        if self.verbose:
            logger.info(
                f"Async evaluation complete: {score:.2f}% "
                f"({int(total_score)}/{len(results)} correct)"
            )

        return EvaluationResult(
            score=round(score, 2),
            results=results if return_predictions else [],
            num_errors=self._error_count,
            metadata={
                "num_examples": len(devset),
                "num_evaluated": len(results),
                "metric_name": getattr(self.metric, "__name__", "custom"),
                "async": True,
            },
        )

    async def _aevaluate_concurrent(
        self,
        module: Module,
        devset: List[Example],
    ) -> List[Tuple[Example, Any, float]]:
        """Evaluate all examples concurrently."""
        tasks = [
            self._aevaluate_one(module, example)
            for example in devset
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async evaluation error: {result}")
                final_results.append((devset[idx], None, self.failure_score))
                self._error_count += 1
            else:
                final_results.append(result)

        return final_results

    async def _aevaluate_with_semaphore(
        self,
        module: Module,
        devset: List[Example],
        max_concurrency: int,
    ) -> List[Tuple[Example, Any, float]]:
        """Evaluate examples with limited concurrency using semaphore."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_evaluate(example: Example):
            async with semaphore:
                return await self._aevaluate_one(module, example)

        tasks = [bounded_evaluate(example) for example in devset]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async evaluation error: {result}")
                final_results.append((devset[idx], None, self.failure_score))
                self._error_count += 1
            else:
                final_results.append(result)

        return final_results

    async def _aevaluate_one(
        self,
        module: Module,
        example: Example,
    ) -> Tuple[Example, Any, float]:
        """Evaluate a single example asynchronously."""
        try:
            # Run the module asynchronously
            if hasattr(module, "acall"):
                prediction = await module.acall(example.inputs)
            elif hasattr(module, "aforward"):
                prediction = await module.aforward(example.inputs)
            else:
                # Fallback to sync in executor
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None, module, example.inputs
                )

            # Score with metric (sync operation)
            score = self.metric(example, prediction)

            return (example, prediction, score)

        except Exception as e:
            self._error_count += 1

            if self.verbose:
                logger.warning(f"Async evaluation error: {e}")

            if self._error_count >= self.max_errors:
                raise RuntimeError(
                    f"Maximum errors ({self.max_errors}) reached during evaluation"
                ) from e

            return (example, None, self.failure_score)

    async def aevaluate_single(
        self,
        module: Module,
        example: Example,
    ) -> Tuple[Any, float]:
        """Evaluate a single example asynchronously.

        Args:
            module: The module to evaluate.
            example: The example to evaluate.

        Returns:
            Tuple of (prediction, score).
        """
        _, prediction, score = await self._aevaluate_one(module, example)
        return prediction, score
