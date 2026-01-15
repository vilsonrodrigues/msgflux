"""Common metrics for evaluation.

This module provides commonly used metrics for evaluating model predictions
against ground truth labels.
"""

import re
from typing import Any, Callable, List, Optional, Set, Union

from msgflux.examples import Example


def exact_match(example: Example, prediction: Any) -> float:
    """Check if prediction exactly matches the label.

    Performs case-insensitive comparison after stripping whitespace.

    Args:
        example: Example with expected labels.
        prediction: Model prediction (will be converted to string).

    Returns:
        1.0 if exact match, 0.0 otherwise.

    Example:
        >>> ex = Example(inputs="What is 2+2?", labels="4")
        >>> exact_match(ex, "4")
        1.0
        >>> exact_match(ex, "four")
        0.0
    """
    label = _extract_label(example.labels)
    pred = str(prediction).strip().lower()
    return float(label.strip().lower() == pred)


def contains_match(example: Example, prediction: Any) -> float:
    """Check if the label is contained in the prediction.

    Useful for open-ended generation where the answer should
    appear somewhere in the response.

    Args:
        example: Example with expected labels.
        prediction: Model prediction (will be converted to string).

    Returns:
        1.0 if label is in prediction, 0.0 otherwise.

    Example:
        >>> ex = Example(inputs="Capital of France?", labels="Paris")
        >>> contains_match(ex, "The capital of France is Paris.")
        1.0
    """
    label = _extract_label(example.labels)
    pred = str(prediction).strip().lower()
    return float(label.strip().lower() in pred)


def f1_score(example: Example, prediction: Any) -> float:
    """Calculate token-level F1 score.

    Computes F1 score based on token overlap between
    prediction and label.

    Args:
        example: Example with expected labels.
        prediction: Model prediction (will be converted to string).

    Returns:
        F1 score between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the quick brown fox")
        >>> f1_score(ex, "the quick fox")
        0.857...
    """
    label = _extract_label(example.labels)
    pred = str(prediction)

    label_tokens = _tokenize(label)
    pred_tokens = _tokenize(pred)

    if not label_tokens or not pred_tokens:
        return float(label_tokens == pred_tokens)

    common = label_tokens & pred_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(label_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def semantic_similarity(
    example: Example,
    prediction: Any,
    *,
    embedder: Optional[Callable[[str], List[float]]] = None,
    threshold: float = 0.8,
) -> float:
    """Check semantic similarity using embeddings.

    Computes cosine similarity between embeddings of the
    prediction and label.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.
        embedder: Function to compute embeddings. If None, falls back
            to exact_match.
        threshold: Similarity threshold for success. Defaults to 0.8.

    Returns:
        Cosine similarity score if embedder provided, else exact_match.
    """
    if embedder is None:
        return exact_match(example, prediction)

    label = _extract_label(example.labels)
    pred = str(prediction)

    try:
        label_emb = embedder(label)
        pred_emb = embedder(pred)
        similarity = _cosine_similarity(label_emb, pred_emb)
        return similarity
    except Exception:
        return exact_match(example, prediction)


def regex_match(
    example: Example,
    prediction: Any,
    *,
    pattern: Optional[str] = None,
) -> float:
    """Check if prediction matches a regex pattern.

    If no pattern is provided, uses the label as the pattern.

    Args:
        example: Example with expected labels (used as pattern if not provided).
        prediction: Model prediction.
        pattern: Optional regex pattern to match.

    Returns:
        1.0 if pattern matches, 0.0 otherwise.
    """
    if pattern is None:
        pattern = _extract_label(example.labels)

    pred = str(prediction)

    try:
        if re.search(pattern, pred, re.IGNORECASE):
            return 1.0
        return 0.0
    except re.error:
        # Invalid regex, fall back to contains
        return float(pattern.lower() in pred.lower())


def answer_correctness(
    example: Example,
    prediction: Any,
    *,
    normalize: bool = True,
) -> float:
    """Flexible answer correctness metric.

    Handles various answer formats:
    - Exact match
    - Numeric comparison
    - Boolean comparison
    - List comparison

    Args:
        example: Example with expected labels.
        prediction: Model prediction.
        normalize: Whether to normalize strings. Defaults to True.

    Returns:
        Score between 0.0 and 1.0.
    """
    label = _extract_label(example.labels)
    pred = str(prediction)

    if normalize:
        label = _normalize_text(label)
        pred = _normalize_text(pred)

    # Try exact match first
    if label == pred:
        return 1.0

    # Try numeric comparison
    try:
        label_num = float(label)
        pred_num = float(pred)
        return float(abs(label_num - pred_num) < 1e-6)
    except ValueError:
        pass

    # Try boolean comparison
    label_bool = _to_bool(label)
    pred_bool = _to_bool(pred)
    if label_bool is not None and pred_bool is not None:
        return float(label_bool == pred_bool)

    # Check containment
    if label in pred or pred in label:
        return 0.5

    return 0.0


def create_metric(
    metric_fn: Callable[[str, str], float],
    *,
    name: Optional[str] = None,
) -> Callable[[Example, Any], float]:
    """Create a metric function from a simple string comparison function.

    This is useful for wrapping external metric functions.

    Args:
        metric_fn: Function that takes (label, prediction) strings.
        name: Optional name for the metric.

    Returns:
        Metric function compatible with Evaluator.

    Example:
        >>> def my_metric(label, pred):
        ...     return float(len(pred) > len(label))
        >>> metric = create_metric(my_metric, name="longer_than_label")
    """

    def wrapper(example: Example, prediction: Any) -> float:
        label = _extract_label(example.labels)
        pred = str(prediction)
        return metric_fn(label, pred)

    if name:
        wrapper.__name__ = name
    else:
        wrapper.__name__ = getattr(metric_fn, "__name__", "custom_metric")

    return wrapper


# Helper functions


def _extract_label(labels: Any) -> str:
    """Extract label string from various formats."""
    if isinstance(labels, str):
        return labels
    elif isinstance(labels, dict):
        # Try common keys
        for key in ["answer", "output", "label", "response"]:
            if key in labels:
                return str(labels[key])
        # Return first value
        return str(next(iter(labels.values())))
    else:
        return str(labels)


def _tokenize(text: str) -> Set[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    # Convert to lowercase and split
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove punctuation at edges
    text = text.strip(".,!?;:'\"")
    return text


def _to_bool(text: str) -> Optional[bool]:
    """Convert text to boolean if possible."""
    text = text.lower().strip()
    if text in ("true", "yes", "1", "correct", "right"):
        return True
    elif text in ("false", "no", "0", "incorrect", "wrong"):
        return False
    return None


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
