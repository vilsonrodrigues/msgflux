"""Tests for msgflux.utils.logging module."""

import logging

from msgflux.utils.logging import NewLineFormatter


def test_newline_formatter_single_line():
    """Test NewLineFormatter with single line messages."""
    formatter = NewLineFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Single line message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert formatted == "INFO - Single line message"
    assert "\r\n" not in formatted


def test_newline_formatter_multi_line():
    """Test NewLineFormatter with multi-line messages."""
    formatter = NewLineFormatter("%(levelname)s - %(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Line 1\nLine 2\nLine 3",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "\r\n" in formatted
    lines = formatted.split("\r\n")
    assert len(lines) == 3
