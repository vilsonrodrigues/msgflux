"""Tests for msgflux.evaluate.metrics module."""

import pytest

from msgflux.evaluate.metrics import (
    answer_correctness,
    contains_match,
    create_metric,
    exact_match,
    f1_score,
    regex_match,
)
from msgflux.examples import Example


class TestExactMatch:
    """Test suite for exact_match metric."""

    def test_exact_match_true(self):
        """Test exact match with matching strings."""
        example = Example(inputs="question", labels="answer")

        score = exact_match(example, "answer")

        assert score == 1.0

    def test_exact_match_false(self):
        """Test exact match with non-matching strings."""
        example = Example(inputs="question", labels="answer")

        score = exact_match(example, "wrong")

        assert score == 0.0

    def test_exact_match_case_insensitive(self):
        """Test that exact match is case insensitive."""
        example = Example(inputs="question", labels="Answer")

        score = exact_match(example, "ANSWER")

        assert score == 1.0

    def test_exact_match_whitespace(self):
        """Test exact match with whitespace."""
        example = Example(inputs="question", labels="  answer  ")

        score = exact_match(example, "answer")

        assert score == 1.0

    def test_exact_match_dict_labels(self):
        """Test exact match with dict labels."""
        example = Example(inputs="question", labels={"answer": "42"})

        score = exact_match(example, "42")

        assert score == 1.0

    def test_exact_match_numeric(self):
        """Test exact match with numeric strings."""
        example = Example(inputs="question", labels="42")

        score = exact_match(example, 42)

        assert score == 1.0


class TestContainsMatch:
    """Test suite for contains_match metric."""

    def test_contains_match_true(self):
        """Test when label is contained in prediction."""
        example = Example(inputs="question", labels="Paris")

        score = contains_match(example, "The capital of France is Paris.")

        assert score == 1.0

    def test_contains_match_false(self):
        """Test when label is not in prediction."""
        example = Example(inputs="question", labels="Paris")

        score = contains_match(example, "The capital of France is London.")

        assert score == 0.0

    def test_contains_match_case_insensitive(self):
        """Test case insensitive matching."""
        example = Example(inputs="question", labels="PARIS")

        score = contains_match(example, "paris is beautiful")

        assert score == 1.0


class TestF1Score:
    """Test suite for f1_score metric."""

    def test_f1_perfect_match(self):
        """Test F1 with perfect match."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = f1_score(example, "the quick brown fox")

        assert score == 1.0

    def test_f1_partial_match(self):
        """Test F1 with partial overlap."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = f1_score(example, "the quick fox")

        # 3 common tokens, 4 label tokens, 3 pred tokens
        # precision = 3/3 = 1.0, recall = 3/4 = 0.75
        # f1 = 2 * 1.0 * 0.75 / (1.0 + 0.75) = 0.857...
        assert 0.85 < score < 0.87

    def test_f1_no_match(self):
        """Test F1 with no overlap."""
        example = Example(inputs="q", labels="hello world")

        score = f1_score(example, "goodbye universe")

        assert score == 0.0

    def test_f1_empty_strings(self):
        """Test F1 with empty strings."""
        example = Example(inputs="q", labels="")

        score = f1_score(example, "")

        assert score == 1.0  # Both empty

    def test_f1_one_empty(self):
        """Test F1 with one empty string."""
        example = Example(inputs="q", labels="hello")

        score = f1_score(example, "")

        assert score == 0.0


class TestRegexMatch:
    """Test suite for regex_match metric."""

    def test_regex_match_true(self):
        """Test regex matching."""
        example = Example(inputs="q", labels=r"\d+")

        score = regex_match(example, "The answer is 42")

        assert score == 1.0

    def test_regex_match_false(self):
        """Test regex not matching."""
        example = Example(inputs="q", labels=r"^\d+$")

        score = regex_match(example, "no numbers here")

        assert score == 0.0

    def test_regex_match_custom_pattern(self):
        """Test regex with custom pattern."""
        example = Example(inputs="q", labels="ignored")

        score = regex_match(example, "hello world", pattern=r"world")

        assert score == 1.0

    def test_regex_match_invalid_pattern(self):
        """Test fallback for invalid regex."""
        example = Example(inputs="q", labels="[invalid")

        # Should fall back to contains match
        score = regex_match(example, "[invalid pattern")

        assert score == 1.0


class TestAnswerCorrectness:
    """Test suite for answer_correctness metric."""

    def test_exact_match(self):
        """Test exact match."""
        example = Example(inputs="q", labels="answer")

        score = answer_correctness(example, "answer")

        assert score == 1.0

    def test_numeric_match(self):
        """Test numeric comparison."""
        example = Example(inputs="q", labels="42")

        score = answer_correctness(example, "42.0")

        assert score == 1.0

    def test_boolean_match(self):
        """Test boolean comparison."""
        example = Example(inputs="q", labels="true")

        score = answer_correctness(example, "yes")

        assert score == 1.0

    def test_boolean_mismatch(self):
        """Test boolean mismatch."""
        example = Example(inputs="q", labels="true")

        score = answer_correctness(example, "false")

        assert score == 0.0

    def test_containment_partial(self):
        """Test partial score for containment."""
        example = Example(inputs="q", labels="Paris")

        score = answer_correctness(example, "Paris, France")

        assert score == 0.5

    def test_no_match(self):
        """Test no match."""
        example = Example(inputs="q", labels="Paris")

        score = answer_correctness(example, "London")

        assert score == 0.0

    def test_normalize_option(self):
        """Test normalization option."""
        example = Example(inputs="q", labels="  ANSWER.  ")

        score = answer_correctness(example, "answer", normalize=True)

        assert score == 1.0


class TestCreateMetric:
    """Test suite for create_metric factory."""

    def test_create_basic_metric(self):
        """Test creating a basic metric."""

        def simple_compare(label, pred):
            return float(label == pred)

        metric = create_metric(simple_compare)
        example = Example(inputs="q", labels="answer")

        score = metric(example, "answer")

        assert score == 1.0

    def test_create_metric_with_name(self):
        """Test creating metric with custom name."""

        def my_func(label, pred):
            return 0.5

        metric = create_metric(my_func, name="custom_metric")

        assert metric.__name__ == "custom_metric"

    def test_create_metric_extracts_label(self):
        """Test that metric extracts label from dict."""

        def compare(label, pred):
            return float(label == pred)

        metric = create_metric(compare)
        example = Example(inputs="q", labels={"answer": "42"})

        score = metric(example, "42")

        assert score == 1.0
