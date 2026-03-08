"""Tests for MCP log levels."""

import pytest

from msgflux.protocols.mcp.loglevels import LogLevel


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_loglevel_debug(self):
        """Test DEBUG log level."""
        assert LogLevel.DEBUG.value == "debug"

    def test_loglevel_info(self):
        """Test INFO log level."""
        assert LogLevel.INFO.value == "info"

    def test_loglevel_notice(self):
        """Test NOTICE log level."""
        assert LogLevel.NOTICE.value == "notice"

    def test_loglevel_warning(self):
        """Test WARNING log level."""
        assert LogLevel.WARNING.value == "warning"

    def test_loglevel_error(self):
        """Test ERROR log level."""
        assert LogLevel.ERROR.value == "error"

    def test_loglevel_critical(self):
        """Test CRITICAL log level."""
        assert LogLevel.CRITICAL.value == "critical"

    def test_loglevel_alert(self):
        """Test ALERT log level."""
        assert LogLevel.ALERT.value == "alert"

    def test_loglevel_emergency(self):
        """Test EMERGENCY log level."""
        assert LogLevel.EMERGENCY.value == "emergency"

    def test_loglevel_all_values(self):
        """Test that all log levels are defined."""
        expected_levels = {
            "debug",
            "info",
            "notice",
            "warning",
            "error",
            "critical",
            "alert",
            "emergency",
        }
        actual_levels = {level.value for level in LogLevel}
        assert actual_levels == expected_levels

    def test_loglevel_is_enum(self):
        """Test that LogLevel is an enum."""
        from enum import Enum

        assert issubclass(LogLevel, Enum)

    def test_loglevel_string_conversion(self):
        """Test converting LogLevel to string."""
        assert str(LogLevel.INFO.value) == "info"
        assert str(LogLevel.ERROR.value) == "error"
