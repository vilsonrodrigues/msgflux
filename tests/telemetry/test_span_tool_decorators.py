"""Tests for msgflux.telemetry.span tool decorators."""

import pytest
from unittest.mock import Mock, patch

from msgflux.telemetry.span import set_tool_attributes, aset_tool_attributes


class MockTool:
    """Mock tool class for testing."""

    def __init__(self, name, description=None):
        self.name = name
        self.description = description


class TestSetToolAttributes:
    """Test suite for set_tool_attributes decorator."""

    @patch("msgflux.telemetry.span.trace.get_current_span")
    @patch("msgflux.telemetry.span.MsgTraceAttributes")
    @patch("msgflux.telemetry.span.envs")
    def test_decorator_with_local_execution(self, mock_envs, mock_attrs, mock_get_span):
        """Test decorator with local execution type."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span
        mock_envs.telemetry_capture_tool_call_responses = True

        tool = MockTool(name="test_tool", description="Test description")

        @set_tool_attributes(execution_type="local")
        def execute(self, arg1=None):
            return f"result: {arg1}"

        result = execute(tool, arg1="value1")

        assert result == "result: value1"
        mock_attrs.set_operation_name.assert_called_with("tool")
        mock_attrs.set_tool_name.assert_called_with("test_tool")
        mock_attrs.set_tool_description.assert_called_with("Test description")
        mock_attrs.set_tool_execution_type.assert_called_with("local")

    @patch("msgflux.telemetry.span.trace.get_current_span")
    @patch("msgflux.telemetry.span.MsgTraceAttributes")
    @patch("msgflux.telemetry.span.envs")
    def test_decorator_with_remote_execution_and_protocol(
        self, mock_envs, mock_attrs, mock_get_span
    ):
        """Test decorator with remote execution and MCP protocol."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span
        mock_envs.telemetry_capture_tool_call_responses = True

        tool = MockTool(name="remote_tool")

        @set_tool_attributes(execution_type="remote", protocol="mcp")
        def execute(self, **kwargs):
            return "remote_result"

        result = execute(tool)

        assert result == "remote_result"
        mock_attrs.set_tool_execution_type.assert_called_with("remote")
        mock_attrs.set_tool_protocol.assert_called_with("mcp")

    @patch("msgflux.telemetry.span.trace.get_current_span")
    @patch("msgflux.telemetry.span.MsgTraceAttributes")
    @patch("msgflux.telemetry.span.envs")
    def test_decorator_with_tool_call_id(self, mock_envs, mock_attrs, mock_get_span):
        """Test decorator captures tool_call_id."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span
        mock_envs.telemetry_capture_tool_call_responses = True

        tool = MockTool(name="id_tool")

        @set_tool_attributes(execution_type="local")
        def execute(self, **kwargs):
            return "result"

        result = execute(tool, tool_call_id="call_123")

        assert result == "result"
        mock_attrs.set_tool_call_id.assert_called_with("call_123")

    @patch("msgflux.telemetry.span.trace.get_current_span")
    def test_decorator_when_span_not_recording(self, mock_get_span):
        """Test decorator when span is not recording."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        tool = MockTool(name="test_tool")

        @set_tool_attributes(execution_type="local")
        def execute(self):
            return "result"

        result = execute(tool)

        # Should still return result but not set attributes
        assert result == "result"


class TestAsetToolAttributes:
    """Test suite for aset_tool_attributes async decorator."""

    @pytest.mark.asyncio
    @patch("msgflux.telemetry.span.trace.get_current_span")
    @patch("msgflux.telemetry.span.MsgTraceAttributes")
    @patch("msgflux.telemetry.span.envs")
    async def test_async_decorator_with_local_execution(
        self, mock_envs, mock_attrs, mock_get_span
    ):
        """Test async decorator with local execution type."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span
        mock_envs.telemetry_capture_tool_call_responses = True

        tool = MockTool(name="async_tool", description="Async description")

        @aset_tool_attributes(execution_type="local")
        async def execute(self, arg1=None):
            return f"async_result: {arg1}"

        result = await execute(tool, arg1="async_value")

        assert result == "async_result: async_value"
        mock_attrs.set_operation_name.assert_called_with("tool")
        mock_attrs.set_tool_name.assert_called_with("async_tool")
        mock_attrs.set_tool_description.assert_called_with("Async description")
        mock_attrs.set_tool_execution_type.assert_called_with("local")

    @pytest.mark.asyncio
    @patch("msgflux.telemetry.span.trace.get_current_span")
    @patch("msgflux.telemetry.span.MsgTraceAttributes")
    @patch("msgflux.telemetry.span.envs")
    async def test_async_decorator_with_protocol(
        self, mock_envs, mock_attrs, mock_get_span
    ):
        """Test async decorator with protocol."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span
        mock_envs.telemetry_capture_tool_call_responses = True

        tool = MockTool(name="mcp_tool")

        @aset_tool_attributes(execution_type="remote", protocol="mcp")
        async def execute(self):
            return "mcp_result"

        result = await execute(tool)

        assert result == "mcp_result"
        mock_attrs.set_tool_protocol.assert_called_with("mcp")

    @pytest.mark.asyncio
    @patch("msgflux.telemetry.span.trace.get_current_span")
    async def test_async_decorator_when_span_not_recording(self, mock_get_span):
        """Test async decorator when span is not recording."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        tool = MockTool(name="test_tool")

        @aset_tool_attributes(execution_type="local")
        async def execute(self):
            return "result"

        result = await execute(tool)

        # Decorator should still work without recording
        assert result == "result"
