"""Tests for MCP integration functions."""

import pytest

from msgflux.protocols.mcp.integration import (
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.protocols.mcp.types import MCPContent, MCPTool, MCPToolResult


class TestFilterTools:
    """Tests for filter_tools function."""

    @pytest.fixture
    def sample_tools(self):
        """Sample MCP tools for testing."""
        return [
            MCPTool(name="read_file", description="Read file", inputSchema={}),
            MCPTool(name="write_file", description="Write file", inputSchema={}),
            MCPTool(name="delete_file", description="Delete file", inputSchema={}),
            MCPTool(name="list_files", description="List files", inputSchema={}),
        ]

    def test_filter_no_filters(self, sample_tools):
        """Test with no filters - should return all tools."""
        result = filter_tools(sample_tools)
        assert len(result) == 4
        assert result == sample_tools

    def test_filter_include_only(self, sample_tools):
        """Test with include_tools filter."""
        result = filter_tools(sample_tools, include_tools=["read_file", "write_file"])
        assert len(result) == 2
        assert result[0].name == "read_file"
        assert result[1].name == "write_file"

    def test_filter_exclude_only(self, sample_tools):
        """Test with exclude_tools filter."""
        result = filter_tools(sample_tools, exclude_tools=["delete_file"])
        assert len(result) == 3
        tool_names = [t.name for t in result]
        assert "delete_file" not in tool_names
        assert "read_file" in tool_names

    def test_filter_include_takes_priority(self, sample_tools):
        """Test that include_tools takes priority over exclude_tools."""
        result = filter_tools(
            sample_tools,
            include_tools=["read_file", "write_file"],
            exclude_tools=["write_file"]  # Should be ignored
        )
        assert len(result) == 2
        assert result[1].name == "write_file"  # Not excluded

    def test_filter_include_nonexistent(self, sample_tools):
        """Test including non-existent tools."""
        result = filter_tools(sample_tools, include_tools=["nonexistent"])
        assert len(result) == 0

    def test_filter_exclude_all(self, sample_tools):
        """Test excluding all tools."""
        all_names = [t.name for t in sample_tools]
        result = filter_tools(sample_tools, exclude_tools=all_names)
        assert len(result) == 0


class TestConvertMCPSchema:
    """Tests for convert_mcp_schema_to_tool_schema function."""

    def test_convert_basic_schema(self):
        """Test converting basic MCP tool schema."""
        mcp_tool = MCPTool(
            name="read_file",
            description="Read a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        )

        schema = convert_mcp_schema_to_tool_schema(mcp_tool)

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "read_file"
        assert schema["function"]["description"] == "Read a file"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_convert_with_namespace(self):
        """Test converting schema with namespace."""
        mcp_tool = MCPTool(
            name="list_files",
            description="List files",
            inputSchema={}
        )

        schema = convert_mcp_schema_to_tool_schema(mcp_tool, namespace="fs")

        assert schema["function"]["name"] == "fs__list_files"
        assert schema["function"]["description"] == "List files"

    def test_convert_empty_schema(self):
        """Test converting tool with empty input schema."""
        mcp_tool = MCPTool(
            name="simple_tool",
            description="Simple tool",
            inputSchema={}
        )

        schema = convert_mcp_schema_to_tool_schema(mcp_tool)

        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {}

    def test_convert_complex_schema(self):
        """Test converting complex schema with nested properties."""
        mcp_tool = MCPTool(
            name="search",
            description="Search files",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "case_sensitive": {"type": "boolean"},
                            "max_results": {"type": "integer"}
                        }
                    }
                },
                "required": ["query"]
            }
        )

        schema = convert_mcp_schema_to_tool_schema(mcp_tool)

        assert "options" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["properties"]["options"]["type"] == "object"


class TestExtractToolResultText:
    """Tests for extract_tool_result_text function."""

    def test_extract_single_text_content(self):
        """Test extracting text from single text content."""
        result = MCPToolResult(
            content=[MCPContent(type="text", text="Hello, world!")],
            isError=False
        )

        text = extract_tool_result_text(result)
        assert text == "Hello, world!"

    def test_extract_multiple_text_contents(self):
        """Test extracting text from multiple contents."""
        result = MCPToolResult(
            content=[
                MCPContent(type="text", text="Part 1"),
                MCPContent(type="text", text="Part 2"),
                MCPContent(type="text", text="Part 3")
            ],
            isError=False
        )

        text = extract_tool_result_text(result)
        assert text == "Part 1\nPart 2\nPart 3"

    def test_extract_resource_content(self):
        """Test extracting data from resource content."""
        result = MCPToolResult(
            content=[MCPContent(type="resource", data="base64data")],
            isError=False
        )

        text = extract_tool_result_text(result)
        assert text == "base64data"

    def test_extract_mixed_content(self):
        """Test extracting from mixed content types."""
        result = MCPToolResult(
            content=[
                MCPContent(type="text", text="Text part"),
                MCPContent(type="resource", data="Data part"),
                MCPContent(type="text", text="More text")
            ],
            isError=False
        )

        text = extract_tool_result_text(result)
        assert "Text part" in text
        assert "Data part" in text
        assert "More text" in text

    def test_extract_empty_content(self):
        """Test extracting from empty content."""
        result = MCPToolResult(content=[], isError=False)

        text = extract_tool_result_text(result)
        assert text == "Tool executed successfully"

    def test_extract_from_string(self):
        """Test extracting from non-MCPToolResult."""
        text = extract_tool_result_text("Just a string")
        assert text == "Just a string"

    def test_extract_ignores_none_values(self):
        """Test that None values are ignored."""
        result = MCPToolResult(
            content=[
                MCPContent(type="text", text="Valid text"),
                MCPContent(type="text", text=None),
                MCPContent(type="resource", data=None)
            ],
            isError=False
        )

        text = extract_tool_result_text(result)
        assert text == "Valid text"
