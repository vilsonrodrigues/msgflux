"""Common metrics for evaluation.

This module provides commonly used metrics for evaluating model predictions
against ground truth labels.
"""

import asyncio
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

from msgflux.examples import Example

# Type alias for async metrics
AsyncMetricFn = Callable[[Example, Any], Coroutine[Any, Any, float]]


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


def bleu_score(
    example: Example,
    prediction: Any,
    *,
    max_n: int = 4,
    weights: Optional[List[float]] = None,
) -> float:
    """Calculate BLEU score between prediction and label.

    BLEU (Bilingual Evaluation Understudy) measures n-gram precision
    with a brevity penalty. Commonly used for translation evaluation.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.
        max_n: Maximum n-gram size (1-4). Defaults to 4.
        weights: Weights for each n-gram (must sum to 1.0).
            Defaults to uniform weights.

    Returns:
        BLEU score between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the cat sat on the mat")
        >>> bleu_score(ex, "the cat is on the mat")
        0.668...
    """
    import math

    label = _extract_label(example.labels)
    pred = str(prediction)

    ref_tokens = label.lower().split()
    hyp_tokens = pred.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    if weights is None:
        weights = [1.0 / max_n] * max_n

    # Calculate n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(ref_tokens, n)
        hyp_ngrams = _get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        # Count matches with clipping
        matches = 0
        for ngram in hyp_ngrams:
            if ngram in ref_ngrams:
                matches += min(hyp_ngrams[ngram], ref_ngrams[ngram])

        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)

    # Brevity penalty
    if len(hyp_tokens) >= len(ref_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

    return bp * math.exp(log_precision)


def rouge_1(example: Example, prediction: Any) -> float:
    """Calculate ROUGE-1 (unigram) F1 score.

    ROUGE-1 measures unigram overlap between prediction and reference.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.

    Returns:
        ROUGE-1 F1 score between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the quick brown fox")
        >>> rouge_1(ex, "the quick fox jumps")
        0.75
    """
    return _rouge_n(example, prediction, n=1)


def rouge_2(example: Example, prediction: Any) -> float:
    """Calculate ROUGE-2 (bigram) F1 score.

    ROUGE-2 measures bigram overlap between prediction and reference.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.

    Returns:
        ROUGE-2 F1 score between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the quick brown fox")
        >>> rouge_2(ex, "the quick fox")
        0.4
    """
    return _rouge_n(example, prediction, n=2)


def rouge_l(example: Example, prediction: Any) -> float:
    """Calculate ROUGE-L (longest common subsequence) F1 score.

    ROUGE-L uses longest common subsequence for more flexible matching.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the quick brown fox")
        >>> rouge_l(ex, "the brown quick fox")
        0.8
    """
    label = _extract_label(example.labels)
    pred = str(prediction)

    ref_tokens = label.lower().split()
    hyp_tokens = pred.lower().split()

    if not ref_tokens or not hyp_tokens:
        return float(ref_tokens == hyp_tokens)

    lcs_length = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_length / len(hyp_tokens)
    recall = lcs_length / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def levenshtein_similarity(example: Example, prediction: Any) -> float:
    """Calculate normalized Levenshtein similarity.

    Measures how similar two strings are based on minimum edit operations
    (insertions, deletions, substitutions) needed to transform one into another.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.

    Returns:
        Similarity score between 0.0 and 1.0 (1.0 = identical).

    Example:
        >>> ex = Example(inputs="...", labels="kitten")
        >>> levenshtein_similarity(ex, "sitting")
        0.571...
    """
    label = _extract_label(example.labels).lower()
    pred = str(prediction).lower()

    if not label and not pred:
        return 1.0
    if not label or not pred:
        return 0.0

    distance = _levenshtein_distance(label, pred)
    max_len = max(len(label), len(pred))

    return 1.0 - (distance / max_len)


def jaccard_similarity(example: Example, prediction: Any) -> float:
    """Calculate Jaccard similarity between token sets.

    Measures overlap between token sets: |intersection| / |union|.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.

    Returns:
        Jaccard similarity between 0.0 and 1.0.

    Example:
        >>> ex = Example(inputs="...", labels="the quick brown fox")
        >>> jaccard_similarity(ex, "the quick fox jumps")
        0.6
    """
    label = _extract_label(example.labels)
    pred = str(prediction)

    label_tokens = _tokenize(label)
    pred_tokens = _tokenize(pred)

    if not label_tokens and not pred_tokens:
        return 1.0
    if not label_tokens or not pred_tokens:
        return 0.0

    intersection = label_tokens & pred_tokens
    union = label_tokens | pred_tokens

    return len(intersection) / len(union)


def llm_as_judge(
    example: Example,
    prediction: Any,
    *,
    judge: Optional[Callable[[str], str]] = None,
    criteria: Optional[str] = None,
) -> float:
    """Use an LLM to evaluate prediction quality.

    The LLM judge assesses whether the prediction correctly answers
    the question based on the expected label.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.
        judge: Callable that takes a prompt and returns LLM response.
            If None, falls back to exact_match.
        criteria: Optional evaluation criteria to include in the prompt.

    Returns:
        Score between 0.0 and 1.0 based on LLM judgment.

    Example:
        >>> def my_judge(prompt):
        ...     return call_llm(prompt)  # Returns "correct" or "incorrect"
        >>> ex = Example(inputs="What is 2+2?", labels="4")
        >>> llm_as_judge(ex, "The answer is 4", judge=my_judge)
        1.0
    """
    if judge is None:
        return exact_match(example, prediction)

    label = _extract_label(example.labels)
    pred = str(prediction)
    question = str(example.inputs) if hasattr(example, "inputs") else ""

    # Build evaluation prompt
    criteria_text = f"\nEvaluation criteria: {criteria}" if criteria else ""

    prompt = f"""Evaluate if the prediction correctly answers the question.

Question: {question}
Expected Answer: {label}
Prediction: {pred}
{criteria_text}
Is the prediction correct? Answer only with a number from 0 to 10, where:
- 0 = completely wrong
- 5 = partially correct
- 10 = completely correct

Score:"""

    try:
        response = judge(prompt)
        # Parse score from response
        score = _parse_llm_score(response)
        return score / 10.0
    except Exception:
        return exact_match(example, prediction)


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


def _get_ngrams(tokens: List[str], n: int) -> dict:
    """Get n-gram counts from token list."""
    from collections import Counter

    if len(tokens) < n:
        return {}

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def _rouge_n(example: Example, prediction: Any, n: int) -> float:
    """Calculate ROUGE-N F1 score."""
    label = _extract_label(example.labels)
    pred = str(prediction)

    ref_tokens = label.lower().split()
    hyp_tokens = pred.lower().split()

    if not ref_tokens or not hyp_tokens:
        return float(ref_tokens == hyp_tokens)

    ref_ngrams = _get_ngrams(ref_tokens, n)
    hyp_ngrams = _get_ngrams(hyp_tokens, n)

    if not ref_ngrams or not hyp_ngrams:
        return 0.0

    # Count overlapping n-grams
    overlap = 0
    for ngram in hyp_ngrams:
        if ngram in ref_ngrams:
            overlap += min(hyp_ngrams[ngram], ref_ngrams[ngram])

    precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0.0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate longest common subsequence length using dynamic programming."""
    m, n = len(seq1), len(seq2)

    # Use space-optimized DP (only need current and previous row)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev

    return prev[n]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Use space-optimized DP
    prev_row = list(range(len(s2) + 1))
    curr_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row[0] = i + 1
        for j, c2 in enumerate(s2):
            # Cost is 0 if chars match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            curr_row[j + 1] = min(
                prev_row[j + 1] + 1,  # Deletion
                curr_row[j] + 1,  # Insertion
                prev_row[j] + cost,  # Substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len(s2)]


def _parse_llm_score(response: str) -> float:
    """Parse numeric score from LLM response."""
    import re

    # Try to find a number 0-10 in the response
    numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)

    for num_str in numbers:
        num = float(num_str)
        if 0 <= num <= 10:
            return num

    # Fallback: check for keywords
    response_lower = response.lower()
    if any(word in response_lower for word in ["correct", "yes", "right", "perfect"]):
        return 10.0
    if any(word in response_lower for word in ["wrong", "incorrect", "no", "false"]):
        return 0.0
    if any(word in response_lower for word in ["partial", "somewhat", "close"]):
        return 5.0

    return 5.0  # Default to middle score if unparseable


# Async metrics


class AsyncMetric:
    """Wrapper to make sync metrics usable in async contexts.

    Runs the sync metric in an executor to avoid blocking.

    Args:
        metric: Synchronous metric function.

    Example:
        >>> async_exact = AsyncMetric(exact_match)
        >>> score = await async_exact(example, prediction)
    """

    def __init__(self, metric: Callable[[Example, Any], float]):
        self.metric = metric
        self.__name__ = getattr(metric, "__name__", "async_metric")

    async def __call__(self, example: Example, prediction: Any) -> float:
        """Evaluate the metric asynchronously.

        Args:
            example: Example with expected labels.
            prediction: Model prediction.

        Returns:
            Score from the wrapped metric.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.metric, example, prediction)


async def allm_as_judge(
    example: Example,
    prediction: Any,
    *,
    judge: Optional[Any] = None,
    criteria: Optional[str] = None,
    max_retries: int = 3,
) -> float:
    """Async version of llm_as_judge metric.

    Uses an async-capable LLM to evaluate prediction quality.

    Args:
        example: The example being evaluated.
        prediction: The model's prediction.
        judge: Async-capable model for evaluation. Should have acall or aforward method.
        criteria: Custom evaluation criteria.
        max_retries: Number of retries on failure.

    Returns:
        Score from 0.0 to 1.0.

    Example:
        >>> score = await allm_as_judge(example, prediction, judge=async_model)
    """
    if judge is None:
        # Fallback to sync exact_match in executor
        return await AsyncMetric(exact_match)(example, prediction)

    label = _extract_label(example.labels)
    pred = str(prediction)
    question = str(example.inputs) if hasattr(example, "inputs") else ""

    # Build evaluation prompt
    criteria_text = f"\nEvaluation criteria: {criteria}" if criteria else ""

    prompt = f"""Evaluate if the prediction correctly answers the question.

Question: {question}
Expected Answer: {label}
Prediction: {pred}
{criteria_text}
Is the prediction correct? Answer only with a number from 0 to 10, where:
- 0 = completely wrong
- 5 = partially correct
- 10 = completely correct

Score:"""

    for attempt in range(max_retries):
        try:
            # Call judge asynchronously
            if hasattr(judge, "acall"):
                response = await judge.acall(prompt)
            elif hasattr(judge, "aforward"):
                response = await judge.aforward(prompt)
            elif callable(judge):
                # Try calling directly (might be async)
                result = judge(prompt)
                if asyncio.iscoroutine(result):
                    response = await result
                else:
                    # Sync callable, run in executor
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(None, judge, prompt)
            else:
                raise ValueError("judge must be callable or have acall/aforward method")

            # Parse score from response
            score = _parse_llm_score(str(response))
            return score / 10.0

        except Exception as e:
            if attempt == max_retries - 1:
                # Final fallback to exact_match
                return await AsyncMetric(exact_match)(example, prediction)

    return 0.0


async def asemantic_similarity(
    example: Example,
    prediction: Any,
    *,
    embedder: Optional[Any] = None,
    threshold: float = 0.8,
) -> float:
    """Async version of semantic_similarity metric.

    Computes cosine similarity between embeddings asynchronously.

    Args:
        example: Example with expected labels.
        prediction: Model prediction.
        embedder: Async-capable embedder. Should have acall or aforward method.
        threshold: Similarity threshold for success.

    Returns:
        Cosine similarity score if embedder provided, else exact_match.
    """
    if embedder is None:
        return await AsyncMetric(exact_match)(example, prediction)

    label = _extract_label(example.labels)
    pred = str(prediction)

    try:
        # Get embeddings asynchronously
        if hasattr(embedder, "acall"):
            label_emb, pred_emb = await asyncio.gather(
                embedder.acall(label), embedder.acall(pred)
            )
        elif hasattr(embedder, "aforward"):
            label_emb, pred_emb = await asyncio.gather(
                embedder.aforward(label), embedder.aforward(pred)
            )
        else:
            # Fallback to sync in executor
            loop = asyncio.get_running_loop()
            label_emb = await loop.run_in_executor(None, embedder, label)
            pred_emb = await loop.run_in_executor(None, embedder, pred)

        similarity = _cosine_similarity(label_emb, pred_emb)
        return similarity
    except Exception:
        return await AsyncMetric(exact_match)(example, prediction)
