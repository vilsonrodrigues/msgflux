"""Tests for msgflux.evaluate.metrics module."""

import pytest

from msgflux.evaluate.metrics import (
    answer_correctness,
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


class TestBleuScore:
    """Test suite for BLEU score metric."""

    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match."""
        example = Example(inputs="q", labels="the cat sat on the mat")

        score = bleu_score(example, "the cat sat on the mat")

        assert score == 1.0

    def test_bleu_partial_match(self):
        """Test BLEU with partial overlap using BLEU-2."""
        example = Example(inputs="q", labels="the cat sat on the mat")

        # Use max_n=2 for shorter texts (BLEU-2)
        score = bleu_score(example, "the cat is on the mat", max_n=2)

        # Should have some score but not perfect
        assert 0.4 < score < 0.9

    def test_bleu_no_match(self):
        """Test BLEU with no overlap."""
        example = Example(inputs="q", labels="hello world")

        score = bleu_score(example, "goodbye universe")

        assert score == 0.0

    def test_bleu_empty_prediction(self):
        """Test BLEU with empty prediction."""
        example = Example(inputs="q", labels="hello world")

        score = bleu_score(example, "")

        assert score == 0.0

    def test_bleu_shorter_prediction(self):
        """Test BLEU brevity penalty for short predictions."""
        example = Example(inputs="q", labels="the quick brown fox jumps over the lazy dog")

        # Much shorter prediction should have brevity penalty
        score = bleu_score(example, "the fox jumps")

        assert score < 0.5


class TestRouge:
    """Test suite for ROUGE metrics."""

    def test_rouge_1_perfect_match(self):
        """Test ROUGE-1 with perfect match."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = rouge_1(example, "the quick brown fox")

        assert score == 1.0

    def test_rouge_1_partial_match(self):
        """Test ROUGE-1 with partial overlap."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = rouge_1(example, "the quick fox jumps")

        # 3 common words out of 4 reference and 4 prediction
        assert 0.7 < score < 0.8

    def test_rouge_2_perfect_match(self):
        """Test ROUGE-2 with perfect match."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = rouge_2(example, "the quick brown fox")

        assert score == 1.0

    def test_rouge_2_partial_match(self):
        """Test ROUGE-2 with partial bigram overlap."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = rouge_2(example, "the quick fox")

        # Only "the quick" bigram matches
        assert 0.3 < score < 0.5

    def test_rouge_l_perfect_match(self):
        """Test ROUGE-L with perfect match."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = rouge_l(example, "the quick brown fox")

        assert score == 1.0

    def test_rouge_l_reordered(self):
        """Test ROUGE-L with reordered words."""
        example = Example(inputs="q", labels="the quick brown fox")

        # LCS is "the brown fox" (length 3)
        score = rouge_l(example, "the brown quick fox")

        assert 0.7 < score < 0.9

    def test_rouge_no_match(self):
        """Test ROUGE with no overlap."""
        example = Example(inputs="q", labels="hello world")

        assert rouge_1(example, "goodbye universe") == 0.0
        assert rouge_2(example, "goodbye universe") == 0.0
        assert rouge_l(example, "goodbye universe") == 0.0


class TestLevenshteinSimilarity:
    """Test suite for Levenshtein similarity metric."""

    def test_levenshtein_perfect_match(self):
        """Test Levenshtein with identical strings."""
        example = Example(inputs="q", labels="kitten")

        score = levenshtein_similarity(example, "kitten")

        assert score == 1.0

    def test_levenshtein_one_edit(self):
        """Test Levenshtein with one edit distance."""
        example = Example(inputs="q", labels="kitten")

        # "sitten" is 1 edit away from "kitten"
        score = levenshtein_similarity(example, "sitten")

        # 1 - (1/6) = 0.833...
        assert 0.8 < score < 0.85

    def test_levenshtein_multiple_edits(self):
        """Test Levenshtein with multiple edits."""
        example = Example(inputs="q", labels="kitten")

        # "sitting" is 3 edits away from "kitten"
        score = levenshtein_similarity(example, "sitting")

        # 1 - (3/7) = 0.571...
        assert 0.55 < score < 0.6

    def test_levenshtein_empty_strings(self):
        """Test Levenshtein with empty strings."""
        example = Example(inputs="q", labels="")

        score = levenshtein_similarity(example, "")

        assert score == 1.0

    def test_levenshtein_one_empty(self):
        """Test Levenshtein with one empty string."""
        example = Example(inputs="q", labels="hello")

        score = levenshtein_similarity(example, "")

        assert score == 0.0


class TestJaccardSimilarity:
    """Test suite for Jaccard similarity metric."""

    def test_jaccard_perfect_match(self):
        """Test Jaccard with identical token sets."""
        example = Example(inputs="q", labels="the quick brown fox")

        score = jaccard_similarity(example, "the quick brown fox")

        assert score == 1.0

    def test_jaccard_partial_overlap(self):
        """Test Jaccard with partial overlap."""
        example = Example(inputs="q", labels="the quick brown fox")

        # Union: {the, quick, brown, fox, jumps} = 5
        # Intersection: {the, quick, fox} = 3
        score = jaccard_similarity(example, "the quick fox jumps")

        assert score == 0.6

    def test_jaccard_no_overlap(self):
        """Test Jaccard with no overlap."""
        example = Example(inputs="q", labels="hello world")

        score = jaccard_similarity(example, "goodbye universe")

        assert score == 0.0

    def test_jaccard_empty_strings(self):
        """Test Jaccard with empty strings."""
        example = Example(inputs="q", labels="")

        score = jaccard_similarity(example, "")

        assert score == 1.0

    def test_jaccard_order_independent(self):
        """Test that Jaccard is order independent."""
        example = Example(inputs="q", labels="a b c")

        score1 = jaccard_similarity(example, "c b a")
        score2 = jaccard_similarity(example, "a b c")

        assert score1 == score2 == 1.0


class TestLLMAsJudge:
    """Test suite for LLM as judge metric."""

    def test_llm_as_judge_without_judge(self):
        """Test fallback to exact match when no judge provided."""
        example = Example(inputs="What is 2+2?", labels="4")

        score = llm_as_judge(example, "4")

        assert score == 1.0

    def test_llm_as_judge_with_judge(self):
        """Test with mock judge that returns correct."""

        def mock_judge(prompt):
            return "10"

        example = Example(inputs="What is 2+2?", labels="4")

        score = llm_as_judge(example, "The answer is 4", judge=mock_judge)

        assert score == 1.0

    def test_llm_as_judge_partial_score(self):
        """Test with mock judge that returns partial score."""

        def mock_judge(prompt):
            return "5"

        example = Example(inputs="q", labels="Paris")

        score = llm_as_judge(example, "Paris, the capital", judge=mock_judge)

        assert score == 0.5

    def test_llm_as_judge_wrong_answer(self):
        """Test with mock judge that returns wrong."""

        def mock_judge(prompt):
            return "0"

        example = Example(inputs="q", labels="Paris")

        score = llm_as_judge(example, "London", judge=mock_judge)

        assert score == 0.0

    def test_llm_as_judge_keyword_response(self):
        """Test parsing of keyword responses."""

        def mock_judge_correct(prompt):
            return "This is correct!"

        def mock_judge_wrong(prompt):
            return "This is wrong."

        example = Example(inputs="q", labels="test")

        score_correct = llm_as_judge(example, "test", judge=mock_judge_correct)
        score_wrong = llm_as_judge(example, "wrong", judge=mock_judge_wrong)

        assert score_correct == 1.0
        assert score_wrong == 0.0

    def test_llm_as_judge_with_criteria(self):
        """Test that criteria is included in prompt."""
        received_prompts = []

        def mock_judge(prompt):
            received_prompts.append(prompt)
            return "10"

        example = Example(inputs="q", labels="test")
        llm_as_judge(example, "test", judge=mock_judge, criteria="Be strict")

        assert "Be strict" in received_prompts[0]

    def test_llm_as_judge_error_handling(self):
        """Test fallback on judge error."""

        def failing_judge(prompt):
            raise ValueError("API error")

        example = Example(inputs="q", labels="test")

        # Should fall back to exact_match
        score = llm_as_judge(example, "test", judge=failing_judge)

        assert score == 1.0
