"""Evaluation module for msgflux.

This module provides tools for evaluating model performance on datasets
using various metrics.

Example:
    >>> from msgflux.evaluate import Evaluator, exact_match, f1_score
    >>>
    >>> evaluator = Evaluator(metric=exact_match, verbose=True)
    >>> result = evaluator(agent, test_examples)
    >>> print(f"Score: {result.score:.2f}%")
"""

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.evaluate.metrics import (
    answer_correctness,
    contains_match,
    create_metric,
    exact_match,
    f1_score,
    regex_match,
    semantic_similarity,
)

__all__ = [
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    # Metrics
    "exact_match",
    "contains_match",
    "f1_score",
    "semantic_similarity",
    "regex_match",
    "answer_correctness",
    "create_metric",
]
