"""Live integration tests for MCPClient with a real FastMCP server.

The server script (mcp_simple_server.py) is launched as a subprocess by
StdioTransport using `uv run`, which installs fastmcp automatically via
the inline `# /// script` dependency metadata in that file.
"""
import sys
from pathlib import Path

import pytest

from msgflux.protocols.mcp import MCPClient

SERVER_SCRIPT = str(Path(__file__).parent / "mcp_simple_server.py")


def make_client(timeout: float = 15.0) -> MCPClient:
    """Create an MCPClient pointing at the local test server."""
    return MCPClient.from_stdio(
        command="uv",
        args=["run", SERVER_SCRIPT],
        timeout=timeout,
    )


@pytest.mark.asyncio
async def test_connect_and_list_tools():
    """Connect to the server and verify the expected tools are listed."""
    async with make_client() as client:
        tools = await client.list_tools()

        names = {t.name for t in tools}
        assert "add" in names
        assert "echo" in names
        assert "divide" in names


@pytest.mark.asyncio
async def test_call_add():
    """Call the add tool and verify the result."""
    async with make_client() as client:
        result = await client.call_tool("add", {"a": 3, "b": 4})

        assert not result.isError
        assert "7" in result.content[0].text


@pytest.mark.asyncio
async def test_call_echo():
    """Call the echo tool and verify the message is returned unchanged."""
    async with make_client() as client:
        result = await client.call_tool("echo", {"message": "hello msgflux"})

        assert not result.isError
        assert "hello msgflux" in result.content[0].text


@pytest.mark.asyncio
async def test_tool_error_returned_as_result():
    """A tool that raises should come back as isError=True, not an exception."""
    async with make_client() as client:
        result = await client.call_tool("divide", {"a": 1.0, "b": 0.0})

        assert result.isError
        assert result.content  # error message present


@pytest.mark.asyncio
async def test_ping():
    """Verify ping returns True when server is running."""
    async with make_client() as client:
        assert await client.ping() is True


@pytest.mark.asyncio
async def test_list_tools_cache():
    """Second call with use_cache=True should return same objects without a round-trip."""
    async with make_client() as client:
        tools_first = await client.list_tools(use_cache=False)
        tools_cached = await client.list_tools(use_cache=True)

        assert [t.name for t in tools_first] == [t.name for t in tools_cached]
