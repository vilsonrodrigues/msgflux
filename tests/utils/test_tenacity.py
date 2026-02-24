"""Tests for msgflux.utils.tenacity module."""

import pytest
from tenacity import retry, stop_after_attempt

from msgflux.utils.tenacity import apply_retry, default_model_retry, default_tool_retry


class TestDefaults:
    """Test default retry decorators."""

    def test_default_tool_retry_exists(self):
        assert callable(default_tool_retry)

    def test_default_model_retry_exists(self):
        assert callable(default_model_retry)

    def test_default_tool_retry_success(self):
        call_count = 0

        @default_tool_retry
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1

    def test_default_model_retry_success(self):
        call_count = 0

        @default_model_retry
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert fn() == "ok"
        assert call_count == 1


class TestApplyRetry:
    """Test apply_retry helper."""

    def test_none_uses_default(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("fail")
            return "ok"

        wrapped = apply_retry(
            fn, None, default=retry(reraise=True, stop=stop_after_attempt(3))
        )
        assert wrapped() == "ok"
        assert call_count == 2

    def test_false_disables_retry(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        wrapped = apply_retry(fn, False, default=default_tool_retry)
        with pytest.raises(ValueError, match="fail"):
            wrapped()
        assert call_count == 1

    def test_custom_decorator(self):
        call_count = 0
        custom = retry(reraise=True, stop=stop_after_attempt(2))

        def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        wrapped = apply_retry(fn, custom, default=default_tool_retry)
        with pytest.raises(ValueError, match="fail"):
            wrapped()
        assert call_count == 2

    def test_success_no_retry(self):
        call_count = 0

        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        wrapped = apply_retry(fn, None, default=default_tool_retry)
        assert wrapped() == "ok"
        assert call_count == 1
