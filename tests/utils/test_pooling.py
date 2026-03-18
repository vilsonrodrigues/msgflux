"""Tests for msgflux.utils.pooling module."""

import pytest

np = pytest.importorskip("numpy")

from msgflux.utils.pooling import apply_pooling


def test_apply_pooling_mean_2d():
    """Test mean pooling on 2D embeddings."""
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = apply_pooling(embeddings, strategy="mean")
    expected = np.mean(embeddings, axis=0)
    np.testing.assert_array_almost_equal(result, expected)
    assert result.shape == (3,)


def test_apply_pooling_max_2d():
    """Test max pooling on 2D embeddings."""
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = apply_pooling(embeddings, strategy="max")
    expected = np.max(embeddings, axis=0)
    np.testing.assert_array_almost_equal(result, expected)


def test_apply_pooling_cls_2d():
    """Test CLS pooling on 2D embeddings."""
    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    result = apply_pooling(embeddings, strategy="cls")
    expected = embeddings[0, :]
    np.testing.assert_array_almost_equal(result, expected)


def test_apply_pooling_invalid_strategy():
    """Test that invalid pooling strategy raises ValueError."""
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="Unrecognized pooling strategy"):
        apply_pooling(embeddings, strategy="invalid")
