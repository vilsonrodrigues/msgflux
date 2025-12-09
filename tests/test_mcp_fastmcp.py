"""Test MCP client with local FastMCP server."""
import asyncio
from msgflux.protocols.mcp import MCPClient


async def main():
    print("Starting MCP client test...")

    client = MCPClient.from_http(
        base_url="http://127.0.0.1:8000/mcp"
    )

    try:
        async with client:
            print("✅ Connected successfully!")

            # List tools
            print("\n📋 Listing tools...")
            tools = await client.list_tools()
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description or '(no description)'}")
                print(f"    Input schema: {tool.inputSchema}")

            # Call a tool
            print("\n🔧 Calling 'greet' tool...")
            result = await client.call_tool("greet", {"name": "FastMCP"})
            print(f"Result: {result.content}")
            print(f"Is error: {result.isError}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
