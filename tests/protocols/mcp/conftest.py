"""Shared fixtures for MCP protocol live tests."""

from pathlib import Path

import pytest_asyncio

from msgflux.protocols.mcp import MCPClient

_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_simple_server.py")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_mcp_client():
    """Shared MCP client that spawns the test server once per module.

    Using module scope reduces subprocess startup overhead from ~1.5s per test
    to a single startup for the entire test_mcp_live.py module.
    """
    async with MCPClient.from_stdio(
        command="uv",
        args=["run", _SERVER_SCRIPT],
        timeout=15.0,
    ) as client:
        yield client
