"""Streaming types and utilities for model responses.

This module provides types for real-time streaming of model responses,
supporting both text-only and tool-call scenarios.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class StreamChunk:
    """Individual chunk from a streaming response.

    Types:
    - text: Textual content to display to the user
    - reasoning: Reasoning/thinking content (if model supports)
    - tool_call: Complete tool call ready for execution
    - usage: Token usage statistics (usually final chunk)
    """

    type: Literal["text", "reasoning", "tool_call", "usage"]
    content: Optional[str] = None
    tool_call: Optional["ToolCallComplete"] = None
    usage: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallBuffer:
    """Buffer for accumulating tool call during streaming.

    Used internally to collect partial tool call data until complete.
    """

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""


@dataclass
class ToolCallComplete:
    """Complete tool call ready for execution.

    Emitted when a tool call is fully received from the stream.
    """

    id: str
    name: str
    arguments: str  # JSON string

    def get_params(self) -> Dict[str, Any]:
        """Parse arguments JSON to dict.

        Returns:
            Parsed arguments as dictionary.
        """
        return json.loads(self.arguments) if self.arguments else {}


@dataclass
class StreamState:
    """State tracking for streaming responses.

    Tracks accumulated content, tool calls, and metadata during streaming.
    """

    text_buffer: str = ""
    reasoning_buffer: str = ""
    tool_buffers: Dict[int, ToolCallBuffer] = field(default_factory=dict)
    completed_tool_calls: List[ToolCallComplete] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    def get_full_text(self) -> str:
        """Get accumulated text content."""
        return self.text_buffer

    def get_full_reasoning(self) -> str:
        """Get accumulated reasoning content."""
        return self.reasoning_buffer

    def has_tool_calls(self) -> bool:
        """Check if any tool calls were received."""
        return len(self.completed_tool_calls) > 0


# ============================================================================
# XML Streaming Parser for AgentStreamer
# ============================================================================


class ParserState(Enum):
    """States for the incremental XML parser."""

    TEXT = auto()  # Streaming text normally
    MAYBE_TAG = auto()  # Saw "<", waiting to confirm
    IN_TAG = auto()  # Inside a tag, accumulating
    IN_TOOL_BLOCK = auto()  # Inside tool calls block


@dataclass
class ParsedToolCall:
    """A tool call parsed from XML."""

    name: str
    params: Dict[str, Any]


class StreamingXMLParser:
    """Incremental XML parser for detecting tool calls in streamed text.

    This parser processes text token-by-token and detects tool call blocks
    marked with XML tags. It's designed for use with AgentStreamer where
    the model generates text with embedded tool calls.

    The expected XML structure uses tags like:
    - Opening tag for function calls block
    - invoke tags with name attribute for each tool
    - parameter tags with name attribute for each argument

    Example:
        >>> parser = StreamingXMLParser()
        >>> text, tools = parser.feed("Hello ")
        >>> print(text)  # "Hello "
        >>> print(tools)  # []
    """

    # Tag patterns - using constants to avoid XML interpretation
    FUNCTION_CALLS_OPEN = "function_calls"
    FUNCTION_CALLS_CLOSE = "/function_calls"
    INVOKE_PATTERN = r'invoke\s+name="([^"]+)"'
    PARAMETER_PATTERN = r'parameter\s+name="([^"]+)"'

    def __init__(self):
        """Initialize the parser."""
        self.state = ParserState.TEXT
        self.buffer = ""
        self.tag_buffer = ""
        self.tool_block_buffer = ""
        self.depth = 0

    def reset(self):
        """Reset parser state for new stream."""
        self.state = ParserState.TEXT
        self.buffer = ""
        self.tag_buffer = ""
        self.tool_block_buffer = ""
        self.depth = 0

    def feed(self, token: str) -> Tuple[str, List[ParsedToolCall]]:
        """Process a token and return text to stream and detected tools.

        Args:
            token: The next token from the stream.

        Returns:
            Tuple of (text_to_stream, list_of_detected_tools)
        """
        text_output = ""
        detected_tools: List[ParsedToolCall] = []

        for char in token:
            result = self._process_char(char)
            if result:
                if result["type"] == "text":
                    text_output += result["content"]
                elif result["type"] == "tools":
                    detected_tools.extend(result["content"])

        return text_output, detected_tools

    def _process_char(self, char: str) -> Optional[Dict[str, Any]]:
        """Process a single character based on current state."""
        if self.state == ParserState.TEXT:
            return self._handle_text(char)
        elif self.state == ParserState.MAYBE_TAG:
            return self._handle_maybe_tag(char)
        elif self.state == ParserState.IN_TAG:
            return self._handle_in_tag(char)
        elif self.state == ParserState.IN_TOOL_BLOCK:
            return self._handle_tool_block(char)
        return None

    def _handle_text(self, char: str) -> Optional[Dict[str, Any]]:
        """Handle character in TEXT state."""
        if char == "<":
            self.state = ParserState.MAYBE_TAG
            self.tag_buffer = "<"
            return None
        return {"type": "text", "content": char}

    def _handle_maybe_tag(self, char: str) -> Optional[Dict[str, Any]]:
        """Handle character in MAYBE_TAG state."""
        self.tag_buffer += char

        # Check if this looks like a valid tag start
        if len(self.tag_buffer) == 2:
            # Could be start of tag name or closing tag
            if char.isalpha() or char == "/" or char == "!":
                self.state = ParserState.IN_TAG
                return None
            else:
                # Not a tag, flush buffer as text
                self.state = ParserState.TEXT
                result = {"type": "text", "content": self.tag_buffer}
                self.tag_buffer = ""
                return result

        return None

    def _handle_in_tag(self, char: str) -> Optional[Dict[str, Any]]:
        """Handle character in IN_TAG state."""
        self.tag_buffer += char

        if char == ">":
            # Tag complete, check what kind
            tag_content = self.tag_buffer[1:-1].strip()  # Remove < and >

            if tag_content.startswith(self.FUNCTION_CALLS_OPEN):
                # Start of tool calls block
                self.state = ParserState.IN_TOOL_BLOCK
                self.tool_block_buffer = self.tag_buffer
                self.depth = 1
                self.tag_buffer = ""
                return None

            # Not a tool block tag, flush as text
            self.state = ParserState.TEXT
            result = {"type": "text", "content": self.tag_buffer}
            self.tag_buffer = ""
            return result

        return None

    def _handle_tool_block(self, char: str) -> Optional[Dict[str, Any]]:
        """Handle character in IN_TOOL_BLOCK state."""
        self.tool_block_buffer += char

        # Check for closing tag
        close_tag = "<" + self.FUNCTION_CALLS_CLOSE + ">"
        if self.tool_block_buffer.endswith(close_tag):
            # Block complete, parse and return tools
            tools = self._parse_tool_block(self.tool_block_buffer)
            self.state = ParserState.TEXT
            self.tool_block_buffer = ""
            return {"type": "tools", "content": tools}

        return None

    def _parse_tool_block(self, xml_block: str) -> List[ParsedToolCall]:
        """Parse XML block and extract tool calls.

        Args:
            xml_block: Complete XML block containing tool calls.

        Returns:
            List of ParsedToolCall objects.
        """
        tools = []

        # Find all invoke tags
        invoke_pattern = r'<invoke\s+name="([^"]+)">(.*?)</invoke>'
        invoke_matches = re.findall(invoke_pattern, xml_block, re.DOTALL)

        for tool_name, invoke_content in invoke_matches:
            params = {}

            # Find all parameter tags within this invoke
            param_pattern = r'<parameter\s+name="([^"]+)">([^<]*)</parameter>'
            param_matches = re.findall(param_pattern, invoke_content)

            for param_name, param_value in param_matches:
                # Try to parse as JSON, fallback to string
                try:
                    params[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    params[param_name] = param_value.strip()

            tools.append(ParsedToolCall(name=tool_name, params=params))

        return tools

    def flush(self) -> str:
        """Flush any remaining buffered content.

        Call this when the stream ends to get any remaining text.

        Returns:
            Any remaining buffered text.
        """
        result = ""

        if self.tag_buffer:
            result += self.tag_buffer
            self.tag_buffer = ""

        if self.tool_block_buffer:
            # Incomplete tool block, return as text
            result += self.tool_block_buffer
            self.tool_block_buffer = ""

        self.state = ParserState.TEXT
        return result
