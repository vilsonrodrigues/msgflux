"""Tests for MCP exceptions."""

import pytest

from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)


class TestMCPError:
    """Tests for base MCPError exception."""

    def test_raise_base_exception(self):
        """Test raising base MCP exception."""
        with pytest.raises(MCPError) as exc_info:
            raise MCPError("Base error message")
        assert str(exc_info.value) == "Base error message"

    def test_inheritance_from_exception(self):
        """Test that MCPError inherits from Exception."""
        error = MCPError("Test error")
        assert isinstance(error, Exception)


class TestMCPTimeoutError:
    """Tests for MCPTimeoutError exception."""

    def test_raise_timeout_error(self):
        """Test raising timeout error."""
        with pytest.raises(MCPTimeoutError) as exc_info:
            raise MCPTimeoutError("Operation timed out after 30s")
        assert "timed out" in str(exc_info.value)

    def test_inherits_from_mcp_error(self):
        """Test that MCPTimeoutError inherits from MCPError."""
        error = MCPTimeoutError("Timeout")
        assert isinstance(error, MCPError)
        assert isinstance(error, Exception)


class TestMCPToolError:
    """Tests for MCPToolError exception."""

    def test_raise_tool_error(self):
        """Test raising tool error."""
        with pytest.raises(MCPToolError) as exc_info:
            raise MCPToolError("Tool execution failed")
        assert "Tool execution failed" in str(exc_info.value)

    def test_inherits_from_mcp_error(self):
        """Test that MCPToolError inherits from MCPError."""
        error = MCPToolError("Tool failed")
        assert isinstance(error, MCPError)


class TestMCPConnectionError:
    """Tests for MCPConnectionError exception."""

    def test_raise_connection_error(self):
        """Test raising connection error."""
        with pytest.raises(MCPConnectionError) as exc_info:
            raise MCPConnectionError("Failed to connect to server")
        assert "Failed to connect" in str(exc_info.value)

    def test_inherits_from_mcp_error(self):
        """Test that MCPConnectionError inherits from MCPError."""
        error = MCPConnectionError("Connection failed")
        assert isinstance(error, MCPError)
