"""Tests for msgflux.utils.tenacity module."""

import pytest

from msgflux.utils.tenacity import (
    build_model_retry,
    build_tool_retry,
    model_retry,
    tool_retry,
)

# --- Backward compatibility ---


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


# --- build_tool_retry ---


def test_build_tool_retry_default():
    """Test build_tool_retry with default settings."""
    dec = build_tool_retry()
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    assert fn() == "ok"
    assert call_count == 1


def test_build_tool_retry_retries_on_failure():
    """Test that build_tool_retry retries on failure."""
    dec = build_tool_retry(attempts=3)
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("fail")
        return "ok"

    assert fn() == "ok"
    assert call_count == 3


def test_build_tool_retry_disabled():
    """Test build_tool_retry with enabled=False returns no-op."""
    dec = build_tool_retry(enabled=False)
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        fn()
    assert call_count == 1  # No retry


def test_build_tool_retry_custom_attempts():
    """Test build_tool_retry with custom attempts."""
    dec = build_tool_retry(attempts=2)
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        fn()
    assert call_count == 2


# --- build_model_retry ---


def test_build_model_retry_default():
    """Test build_model_retry with default settings."""
    dec = build_model_retry()
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        return "ok"

    assert fn() == "ok"
    assert call_count == 1


def test_build_model_retry_disabled():
    """Test build_model_retry with enabled=False returns no-op."""
    dec = build_model_retry(enabled=False)
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    with pytest.raises(ValueError, match="fail"):
        fn()
    assert call_count == 1  # No retry


def test_build_model_retry_custom_attempts():
    """Test build_model_retry with custom attempts."""
    dec = build_model_retry(attempts=3)
    call_count = 0

    @dec
    def fn():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("fail")
        return "ok"

    assert fn() == "ok"
    assert call_count == 3
