"""Tests for MCP types."""

import pytest

from msgflux.protocols.mcp.types import (
    MCPContent,
    MCPPrompt,
    MCPResource,
    MCPTool,
    MCPToolResult,
)


class TestMCPResource:
    """Tests for MCPResource dataclass."""

    def test_create_resource(self):
        """Test creating a basic resource."""
        resource = MCPResource(
            uri="file:///test.txt",
            name="test_file",
            description="A test file"
        )

        assert resource.uri == "file:///test.txt"
        assert resource.name == "test_file"
        assert resource.description == "A test file"
        assert resource.mimeType is None
        assert resource.annotations is None

    def test_create_resource_with_metadata(self):
        """Test creating resource with all metadata."""
        resource = MCPResource(
            uri="file:///data.json",
            name="data_file",
            description="Data file",
            mimeType="application/json",
            annotations={"size": 1024, "readonly": True}
        )

        assert resource.mimeType == "application/json"
        assert resource.annotations["size"] == 1024
        assert resource.annotations["readonly"] is True


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_create_tool(self):
        """Test creating a basic tool."""
        tool = MCPTool(
            name="read_file",
            description="Read a file from disk",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        )

        assert tool.name == "read_file"
        assert tool.description == "Read a file from disk"
        assert "properties" in tool.inputSchema
        assert "path" in tool.inputSchema["properties"]

    def test_create_tool_empty_schema(self):
        """Test creating tool with empty schema."""
        tool = MCPTool(
            name="simple_tool",
            description="Simple tool",
            inputSchema={}
        )

        assert tool.name == "simple_tool"
        assert tool.inputSchema == {}


class TestMCPPrompt:
    """Tests for MCPPrompt dataclass."""

    def test_create_prompt(self):
        """Test creating a basic prompt."""
        prompt = MCPPrompt(
            name="code_review",
            description="Review code for issues"
        )

        assert prompt.name == "code_review"
        assert prompt.description == "Review code for issues"
        assert prompt.arguments is None

    def test_create_prompt_with_arguments(self):
        """Test creating prompt with arguments."""
        prompt = MCPPrompt(
            name="translate",
            description="Translate text",
            arguments=[
                {"name": "text", "type": "string", "required": True},
                {"name": "target_lang", "type": "string", "required": True}
            ]
        )

        assert len(prompt.arguments) == 2
        assert prompt.arguments[0]["name"] == "text"


class TestMCPContent:
    """Tests for MCPContent dataclass."""

    def test_create_text_content(self):
        """Test creating text content."""
        content = MCPContent(
            type="text",
            text="Hello, world!"
        )

        assert content.type == "text"
        assert content.text == "Hello, world!"
        assert content.data is None
        assert content.mimeType is None

    def test_create_resource_content(self):
        """Test creating resource content."""
        content = MCPContent(
            type="resource",
            data="base64encodeddata",
            mimeType="image/png"
        )

        assert content.type == "resource"
        assert content.data == "base64encodeddata"
        assert content.mimeType == "image/png"


class TestMCPToolResult:
    """Tests for MCPToolResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful tool result."""
        content = [MCPContent(type="text", text="Success")]
        result = MCPToolResult(content=content, isError=False)

        assert len(result.content) == 1
        assert result.content[0].text == "Success"
        assert result.isError is False

    def test_create_error_result(self):
        """Test creating error tool result."""
        content = [MCPContent(type="text", text="Error: File not found")]
        result = MCPToolResult(content=content, isError=True)

        assert result.isError is True
        assert "Error" in result.content[0].text

    def test_create_result_multiple_contents(self):
        """Test creating result with multiple contents."""
        contents = [
            MCPContent(type="text", text="Part 1"),
            MCPContent(type="text", text="Part 2"),
            MCPContent(type="resource", data="data123")
        ]
        result = MCPToolResult(content=contents)

        assert len(result.content) == 3
        assert result.content[2].type == "resource"
