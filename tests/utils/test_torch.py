"""Tests for msgflux.utils.torch module."""

import pytest

torch = pytest.importorskip("torch")

from msgflux.utils.torch import TORCH_DTYPE_MAP


def test_torch_dtype_map_contains_expected_types():
    """Test that TORCH_DTYPE_MAP contains expected dtype mappings."""
    assert "bfloat16" in TORCH_DTYPE_MAP
    assert "float16" in TORCH_DTYPE_MAP
    assert "float32" in TORCH_DTYPE_MAP


def test_torch_dtype_map_bfloat16():
    """Test bfloat16 dtype mapping."""
    assert TORCH_DTYPE_MAP["bfloat16"] == torch.bfloat16


def test_torch_dtype_map_all_values_are_torch_dtypes():
    """Test that all values in TORCH_DTYPE_MAP are valid torch dtypes."""
    for dtype_name, dtype_value in TORCH_DTYPE_MAP.items():
        assert isinstance(dtype_value, torch.dtype)
