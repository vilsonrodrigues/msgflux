"""Tests for msgflux.telemetry.config module."""

import os
import pytest
from unittest.mock import patch

from msgflux.telemetry.config import MsgTraceSettings, configure_msgtrace


class TestMsgTraceSettings:
    """Test suite for MsgTraceSettings configuration class."""

    def test_settings_initialization_defaults(self):
        """Test MsgTraceSettings initialization with default values."""
        settings = MsgTraceSettings()
        
        assert settings.requires_trace is False
        assert settings.span_exporter_type == "console"
        assert settings.otlp_endpoint == ""

    def test_settings_from_env_variables(self, monkeypatch):
        """Test MsgTraceSettings loads from environment variables."""
        monkeypatch.setenv("MSGTRACE_REQUIRES_TRACE", "true")
        monkeypatch.setenv("MSGTRACE_SPAN_EXPORTER_TYPE", "otlp")
        monkeypatch.setenv("MSGTRACE_OTLP_ENDPOINT", "http://localhost:4317")
        
        settings = MsgTraceSettings()
        
        assert settings.requires_trace is True
        assert settings.span_exporter_type == "otlp"
        assert settings.otlp_endpoint == "http://localhost:4317"

    def test_settings_requires_trace_false(self, monkeypatch):
        """Test requires_trace with false value."""
        monkeypatch.setenv("MSGTRACE_REQUIRES_TRACE", "false")
        
        settings = MsgTraceSettings()
        
        assert settings.requires_trace is False


class TestConfigureMsgtrace:
    """Test suite for configure_msgtrace function."""

    def test_configure_requires_trace(self):
        """Test configuring requires_trace parameter."""
        configure_msgtrace(requires_trace=True)
        
        assert os.environ.get("MSGTRACE_REQUIRES_TRACE") == "true"

    def test_configure_span_exporter_type(self):
        """Test configuring span_exporter_type parameter."""
        configure_msgtrace(span_exporter_type="otlp")
        
        assert os.environ.get("MSGTRACE_SPAN_EXPORTER_TYPE") == "otlp"

    def test_configure_otlp_endpoint(self):
        """Test configuring otlp_endpoint parameter."""
        configure_msgtrace(otlp_endpoint="http://localhost:4317")
        
        assert os.environ.get("MSGTRACE_OTLP_ENDPOINT") == "http://localhost:4317"
