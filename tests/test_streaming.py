"""Tests for streaming module."""

from msgflux.models.streaming import (
    ParsedToolCall,
    ParserState,
    StreamChunk,
    StreamingXMLParser,
    StreamState,
    ToolCallBuffer,
    ToolCallComplete,
)


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_text_chunk(self):
        """Test creating a text chunk."""
        chunk = StreamChunk(type="text", content="Hello")
        assert chunk.type == "text"
        assert chunk.content == "Hello"
        assert chunk.tool_call is None
        assert chunk.usage is None

    def test_tool_call_chunk(self):
        """Test creating a tool call chunk."""
        tool = ToolCallComplete(id="123", name="search", arguments='{"q": "test"}')
        chunk = StreamChunk(type="tool_call", tool_call=tool)
        assert chunk.type == "tool_call"
        assert chunk.tool_call.name == "search"

    def test_usage_chunk(self):
        """Test creating a usage chunk."""
        chunk = StreamChunk(type="usage", usage={"input_tokens": 100})
        assert chunk.type == "usage"
        assert chunk.usage["input_tokens"] == 100


class TestToolCallComplete:
    """Tests for ToolCallComplete dataclass."""

    def test_get_params_valid_json(self):
        """Test parsing valid JSON arguments."""
        tool = ToolCallComplete(
            id="123", name="search", arguments='{"query": "test", "limit": 10}'
        )
        params = tool.get_params()
        assert params == {"query": "test", "limit": 10}

    def test_get_params_empty(self):
        """Test parsing empty arguments."""
        tool = ToolCallComplete(id="123", name="search", arguments="")
        params = tool.get_params()
        assert params == {}


class TestToolCallBuffer:
    """Tests for ToolCallBuffer dataclass."""

    def test_buffer_creation(self):
        """Test creating a buffer."""
        buffer = ToolCallBuffer(index=0, id="123", name="search", arguments="")
        assert buffer.index == 0
        assert buffer.id == "123"
        assert buffer.name == "search"
        assert buffer.arguments == ""


class TestStreamState:
    """Tests for StreamState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = StreamState()
        assert state.text_buffer == ""
        assert state.reasoning_buffer == ""
        assert state.tool_buffers == {}
        assert state.completed_tool_calls == []
        assert state.usage is None
        assert state.finish_reason is None

    def test_has_tool_calls(self):
        """Test has_tool_calls method."""
        state = StreamState()
        assert state.has_tool_calls() is False

        state.completed_tool_calls.append(
            ToolCallComplete(id="1", name="test", arguments="{}")
        )
        assert state.has_tool_calls() is True


class TestStreamingXMLParser:
    """Tests for StreamingXMLParser."""

    def test_parser_initial_state(self):
        """Test parser starts in TEXT state."""
        parser = StreamingXMLParser()
        assert parser.state == ParserState.TEXT

    def test_plain_text(self):
        """Test parsing plain text without tags."""
        parser = StreamingXMLParser()
        text, tools = parser.feed("Hello, world!")
        assert text == "Hello, world!"
        assert tools == []

    def test_plain_text_with_less_than(self):
        """Test parsing text with < that's not a tag."""
        parser = StreamingXMLParser()
        text, tools = parser.feed("5 < 10")
        assert text == "5 < 10"
        assert tools == []

    def test_non_tool_tag(self):
        """Test parsing a non-tool XML tag."""
        parser = StreamingXMLParser()
        text, tools = parser.feed("Hello <b>world</b>!")
        assert text == "Hello <b>world</b>!"
        assert tools == []

    def test_tool_call_single(self):
        """Test parsing a single tool call."""
        parser = StreamingXMLParser()
        xml = (
            '<function_calls>'
            '<invoke name="search">'
            '<parameter name="query">test</parameter>'
            '</invoke>'
            '</function_calls>'
        )
        text, tools = parser.feed("Let me search. " + xml)
        assert text == "Let me search. "
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].params == {"query": "test"}

    def test_tool_call_multiple_params(self):
        """Test parsing tool call with multiple parameters."""
        parser = StreamingXMLParser()
        xml = (
            '<function_calls>'
            '<invoke name="search">'
            '<parameter name="query">python</parameter>'
            '<parameter name="limit">10</parameter>'
            '</invoke>'
            '</function_calls>'
        )
        text, tools = parser.feed(xml)
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].params["query"] == "python"
        assert tools[0].params["limit"] == 10  # Should be parsed as int

    def test_tool_call_multiple_tools(self):
        """Test parsing multiple tool calls."""
        parser = StreamingXMLParser()
        xml = (
            '<function_calls>'
            '<invoke name="search"><parameter name="q">a</parameter></invoke>'
            '<invoke name="calculate"><parameter name="expr">1+1</parameter></invoke>'
            '</function_calls>'
        )
        text, tools = parser.feed(xml)
        assert len(tools) == 2
        assert tools[0].name == "search"
        assert tools[1].name == "calculate"

    def test_incremental_parsing(self):
        """Test parsing token by token."""
        parser = StreamingXMLParser()

        # Feed character by character
        full_text = (
            'Hi <function_calls><invoke name="test">'
            '<parameter name="x">1</parameter></invoke></function_calls> done'
        )

        all_text = ""
        all_tools = []

        for char in full_text:
            text, tools = parser.feed(char)
            all_text += text
            all_tools.extend(tools)

        # Flush remaining
        all_text += parser.flush()

        assert "Hi " in all_text
        assert " done" in all_text
        assert len(all_tools) == 1
        assert all_tools[0].name == "test"

    def test_reset(self):
        """Test parser reset."""
        parser = StreamingXMLParser()
        parser.feed("some text")
        parser.state = ParserState.IN_TOOL_BLOCK
        parser.tool_block_buffer = "partial"

        parser.reset()

        assert parser.state == ParserState.TEXT
        assert parser.tool_block_buffer == ""

    def test_flush_incomplete_tag(self):
        """Test flushing incomplete tag buffer."""
        parser = StreamingXMLParser()
        parser.feed("Hello <incomplete")
        remaining = parser.flush()
        assert "<incomplete" in remaining

    def test_json_value_parsing(self):
        """Test that JSON values are parsed correctly."""
        parser = StreamingXMLParser()
        xml = (
            '<function_calls>'
            '<invoke name="test">'
            '<parameter name="number">42</parameter>'
            '<parameter name="float">3.14</parameter>'
            '<parameter name="bool">true</parameter>'
            '<parameter name="string">hello</parameter>'
            '</invoke>'
            '</function_calls>'
        )
        text, tools = parser.feed(xml)
        params = tools[0].params
        assert params["number"] == 42
        assert params["float"] == 3.14
        assert params["bool"] is True
        assert params["string"] == "hello"


class TestParsedToolCall:
    """Tests for ParsedToolCall dataclass."""

    def test_creation(self):
        """Test creating a parsed tool call."""
        tool = ParsedToolCall(name="search", params={"q": "test"})
        assert tool.name == "search"
        assert tool.params == {"q": "test"}


class TestParserState:
    """Tests for ParserState enum."""

    def test_states_exist(self):
        """Test all states exist."""
        assert ParserState.TEXT
        assert ParserState.MAYBE_TAG
        assert ParserState.IN_TAG
        assert ParserState.IN_TOOL_BLOCK
