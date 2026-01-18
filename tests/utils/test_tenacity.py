"""Tests for msgflux.utils.tenacity module."""

import pytest

from msgflux.utils.tenacity import model_retry, tool_retry


def test_model_retry_decorator_exists():
    """Test that model_retry decorator exists and is callable."""
    assert callable(model_retry)


def test_tool_retry_decorator_exists():
    """Test that tool_retry decorator exists and is callable."""
    assert callable(tool_retry)


def test_model_retry_success():
    """Test model_retry with a successful function."""
    call_count = 0

    @model_retry
    def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"

    result = successful_function()
    assert result == "success"
    assert call_count == 1


def test_tool_retry_success():
    """Test tool_retry with a successful function."""
    call_count = 0

    @tool_retry
    def successful_function():
        nonlocal call_count
        call_count += 1
        return "success"

    result = successful_function()
    assert result == "success"
    assert call_count == 1
