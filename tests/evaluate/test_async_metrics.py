"""Tests for Async Metrics."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from msgflux.evaluate.metrics import (
    AsyncMetric,
    allm_as_judge,
    asemantic_similarity,
    exact_match,
    f1_score,
)
from msgflux.examples import Example


class TestAsyncMetric:
    @pytest.mark.asyncio
    async def test_wraps_sync_metric(self):
        async_exact = AsyncMetric(exact_match)

        example = Example(inputs="What is 2+2?", labels="4")
        result = await async_exact(example, "4")

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_preserves_metric_name(self):
        async_exact = AsyncMetric(exact_match)
        assert async_exact.__name__ == "exact_match"

    @pytest.mark.asyncio
    async def test_with_f1_score(self):
        async_f1 = AsyncMetric(f1_score)

        example = Example(inputs="...", labels="the quick brown fox")
        result = await async_f1(example, "the quick fox")

        # F1 should be between 0 and 1
        assert 0.0 <= result <= 1.0
        assert result > 0.5  # Should have decent overlap

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        async_exact = AsyncMetric(exact_match)

        examples = [
            Example(inputs="Q1", labels="a"),
            Example(inputs="Q2", labels="b"),
            Example(inputs="Q3", labels="c"),
        ]
        predictions = ["a", "b", "x"]

        results = await asyncio.gather(
            *[async_exact(ex, pred) for ex, pred in zip(examples, predictions)]
        )

        assert results == [1.0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_with_custom_metric(self):
        def custom_metric(example, prediction):
            return len(str(prediction)) / 10.0

        async_custom = AsyncMetric(custom_metric)

        example = Example(inputs="...", labels="expected")
        result = await async_custom(example, "hello")  # len=5

        assert result == 0.5

    @pytest.mark.asyncio
    async def test_default_name_for_lambda(self):
        async_lambda = AsyncMetric(lambda ex, pred: 1.0)
        # Lambda doesn't have __name__, should use default
        assert async_lambda.__name__ == "async_metric" or "<lambda>" in async_lambda.__name__


class TestAllmAsJudge:
    @pytest.mark.asyncio
    async def test_fallback_to_exact_match_when_no_judge(self):
        example = Example(inputs="What is 2+2?", labels="4")

        result = await allm_as_judge(example, "4", judge=None)

        # Should fall back to exact_match
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_fallback_to_exact_match_when_no_judge_mismatch(self):
        example = Example(inputs="What is 2+2?", labels="4")

        result = await allm_as_judge(example, "five", judge=None)

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_with_async_judge_acall(self):
        mock_judge = MagicMock()
        mock_judge.acall = AsyncMock(return_value="8")

        example = Example(inputs="What is 2+2?", labels="4")
        result = await allm_as_judge(example, "4", judge=mock_judge)

        assert result == 0.8  # 8/10
        mock_judge.acall.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_async_judge_aforward(self):
        mock_judge = MagicMock()
        mock_judge.aforward = AsyncMock(return_value="10")
        # Remove acall so aforward is used
        del mock_judge.acall

        example = Example(inputs="What is 2+2?", labels="4")
        result = await allm_as_judge(example, "4", judge=mock_judge)

        assert result == 1.0  # 10/10
        mock_judge.aforward.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_sync_callable_judge(self):
        def sync_judge(prompt):
            return "7"

        example = Example(inputs="What is 2+2?", labels="4")
        result = await allm_as_judge(example, "4", judge=sync_judge)

        assert result == 0.7  # 7/10

    @pytest.mark.asyncio
    async def test_with_async_callable_judge(self):
        async def async_judge(prompt):
            return "9"

        example = Example(inputs="What is 2+2?", labels="4")
        result = await allm_as_judge(example, "4", judge=async_judge)

        assert result == 0.9  # 9/10

    @pytest.mark.asyncio
    async def test_with_criteria(self):
        mock_judge = MagicMock()
        mock_judge.acall = AsyncMock(return_value="8")

        example = Example(inputs="Explain gravity", labels="Force of attraction")
        result = await allm_as_judge(
            example,
            "Gravity is the force that pulls objects together",
            judge=mock_judge,
            criteria="Must mention force and attraction",
        )

        # Verify criteria was included in prompt
        call_args = mock_judge.acall.call_args[0][0]
        assert "Must mention force and attraction" in call_args

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        call_count = 0

        async def failing_judge(prompt):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("API Error")
            return "10"

        example = Example(inputs="Q", labels="A")
        result = await allm_as_judge(example, "A", judge=failing_judge, max_retries=3)

        assert result == 1.0
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fallback_after_max_retries(self):
        async def always_failing_judge(prompt):
            raise ValueError("Always fails")

        example = Example(inputs="What is 2+2?", labels="4")
        result = await allm_as_judge(
            example, "4", judge=always_failing_judge, max_retries=2
        )

        # Should fall back to exact_match
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_parse_score_from_response(self):
        mock_judge = MagicMock()

        # Test various response formats
        # Note: "partially correct" contains "correct" which is matched first
        test_cases = [
            ("10", 1.0),
            ("Score: 8", 0.8),
            ("I give this a 5 out of 10", 0.5),
            ("completely correct", 1.0),  # Keyword fallback
            ("wrong answer", 0.0),  # Keyword fallback
            ("partial match", 0.5),  # Keyword fallback (partial without correct)
            ("somewhat close", 0.5),  # Keyword fallback
        ]

        for response, expected_score in test_cases:
            mock_judge.acall = AsyncMock(return_value=response)
            example = Example(inputs="Q", labels="A")
            result = await allm_as_judge(example, "A", judge=mock_judge)
            assert result == expected_score, f"Failed for response: {response}"

    @pytest.mark.asyncio
    async def test_invalid_judge_raises(self):
        example = Example(inputs="Q", labels="A")

        # Object without callable interface should eventually fall back
        result = await allm_as_judge(example, "A", judge="not_callable", max_retries=1)

        # Should fall back to exact_match
        assert result == 1.0


class TestAsemanicSimilarity:
    @pytest.mark.asyncio
    async def test_fallback_to_exact_match_when_no_embedder(self):
        example = Example(inputs="...", labels="hello")

        result = await asemantic_similarity(example, "hello", embedder=None)

        assert result == 1.0

    @pytest.mark.asyncio
    async def test_fallback_to_exact_match_when_no_embedder_mismatch(self):
        example = Example(inputs="...", labels="hello")

        result = await asemantic_similarity(example, "world", embedder=None)

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_with_async_embedder_acall(self):
        mock_embedder = MagicMock()
        # Return identical embeddings for perfect similarity
        mock_embedder.acall = AsyncMock(return_value=[1.0, 0.0, 0.0])

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(example, "hello", embedder=mock_embedder)

        assert result == 1.0  # Identical embeddings = perfect similarity
        assert mock_embedder.acall.call_count == 2

    @pytest.mark.asyncio
    async def test_with_async_embedder_aforward(self):
        mock_embedder = MagicMock()
        mock_embedder.aforward = AsyncMock(return_value=[1.0, 0.0, 0.0])
        # Remove acall so aforward is used
        del mock_embedder.acall

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(example, "hello", embedder=mock_embedder)

        assert result == 1.0
        assert mock_embedder.aforward.call_count == 2

    @pytest.mark.asyncio
    async def test_with_sync_embedder(self):
        def sync_embedder(text):
            # Simple embedding: return different vectors for different texts
            if "hello" in text.lower():
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(example, "hello world", embedder=sync_embedder)

        # hello and "hello world" should be same embedding
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_orthogonal_embeddings(self):
        call_count = 0

        def orthogonal_embedder(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [1.0, 0.0, 0.0]  # First call (label)
            return [0.0, 1.0, 0.0]  # Second call (prediction)

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(
            example, "completely different", embedder=orthogonal_embedder
        )

        # Orthogonal vectors have 0 similarity
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_fallback_on_embedder_error(self):
        def failing_embedder(text):
            raise ValueError("Embedding failed")

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(example, "hello", embedder=failing_embedder)

        # Should fall back to exact_match
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_concurrent_embedding_calls(self):
        mock_embedder = MagicMock()
        mock_embedder.acall = AsyncMock(return_value=[1.0, 0.0, 0.0])

        example = Example(inputs="...", labels="hello")
        result = await asemantic_similarity(example, "hello", embedder=mock_embedder)

        # Both embeddings should be fetched (possibly concurrently)
        assert mock_embedder.acall.call_count == 2


class TestAsyncMetricIntegration:
    @pytest.mark.asyncio
    async def test_mixed_async_metrics(self):
        """Test using multiple async metrics together."""
        async_exact = AsyncMetric(exact_match)
        async_f1 = AsyncMetric(f1_score)

        example = Example(inputs="...", labels="the quick brown fox")

        # Run multiple metrics concurrently
        exact_result, f1_result = await asyncio.gather(
            async_exact(example, "the quick brown fox"),
            async_f1(example, "the quick brown fox"),
        )

        assert exact_result == 1.0
        assert f1_result == 1.0

    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        """Test evaluating a batch of examples concurrently."""
        async_exact = AsyncMetric(exact_match)

        examples = [
            (Example(inputs="Q1", labels="answer1"), "answer1"),
            (Example(inputs="Q2", labels="answer2"), "answer2"),
            (Example(inputs="Q3", labels="answer3"), "wrong"),
            (Example(inputs="Q4", labels="answer4"), "answer4"),
        ]

        results = await asyncio.gather(
            *[async_exact(ex, pred) for ex, pred in examples]
        )

        assert results == [1.0, 1.0, 0.0, 1.0]
        assert sum(results) / len(results) == 0.75  # 75% accuracy

