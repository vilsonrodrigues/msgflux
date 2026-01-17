"""Evaluation module for msgflux.

This module provides tools for evaluating model performance on datasets
using various metrics.

Example:
    >>> from msgflux.evaluate import Evaluator, exact_match, f1_score
    >>>
    >>> evaluator = Evaluator(metric=exact_match, verbose=True)
    >>> result = evaluator(agent, test_examples)
    >>> print(f"Score: {result.score:.2f}%")
    >>>
    >>> # Use BLEU/ROUGE for text generation
    >>> from msgflux.evaluate import bleu_score, rouge_l
    >>> evaluator = Evaluator(metric=bleu_score)
    >>>
    >>> # Use LLM as judge for complex evaluations
    >>> from msgflux.evaluate import llm_as_judge
    >>> metric = lambda ex, pred: llm_as_judge(ex, pred, judge=my_llm)
    >>>
    >>> # Async metrics for parallel evaluation
    >>> from msgflux.evaluate import AsyncMetric, allm_as_judge
    >>> async_exact = AsyncMetric(exact_match)
    >>> score = await async_exact(example, prediction)
"""

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.evaluate.metrics import (
    AsyncMetric,
    allm_as_judge,
    answer_correctness,
    asemantic_similarity,
    bleu_score,
    contains_match,
    create_metric,
    exact_match,
    f1_score,
    jaccard_similarity,
    levenshtein_similarity,
    llm_as_judge,
    regex_match,
    rouge_1,
    rouge_2,
    rouge_l,
    semantic_similarity,
)

__all__ = [
    # Evaluator
    "Evaluator",
    "EvaluationResult",
    # Basic metrics
    "exact_match",
    "contains_match",
    "f1_score",
    "semantic_similarity",
    "regex_match",
    "answer_correctness",
    # Text generation metrics
    "bleu_score",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    # Similarity metrics
    "levenshtein_similarity",
    "jaccard_similarity",
    # Advanced metrics
    "llm_as_judge",
    # Async metrics
    "AsyncMetric",
    "allm_as_judge",
    "asemantic_similarity",
    # Utilities
    "create_metric",
]
