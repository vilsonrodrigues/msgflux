"""Tests for msgflux.telemetry.span agent decorators."""

import pytest
from unittest.mock import Mock, patch

from msgflux.telemetry.span import set_agent_attributes, aset_agent_attributes
from msgflux.models.response import ModelStreamResponse


class MockAgent:
    """Mock agent class for testing."""
    def __init__(self, name, description=None):
        self.name = name
        self.description = description


class TestSetAgentAttributes:
    """Test suite for set_agent_attributes decorator."""

    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    def test_decorator_sets_agent_name_and_id(self, mock_attrs, mock_get_span):
        """Test that decorator sets agent name and generates ID."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="test_agent", description="Test agent")

        @set_agent_attributes
        def forward(self, query):
            return f"Response to: {query}"

        result = forward(agent, "test query")

        assert result == "Response to: test query"
        mock_attrs.set_agent_name.assert_called_with("test_agent")
        # Verify agent ID was set (should start with "asst_")
        call_args = mock_attrs.set_agent_id.call_args
        assert call_args is not None
        agent_id = call_args[0][0]
        assert agent_id.startswith("asst_")
        assert len(agent_id) == 29  # "asst_" + 24 hex chars

    @patch('msgflux.telemetry.span.trace.get_current_span')
    def test_decorator_sets_agent_description(self, mock_get_span):
        """Test that decorator sets agent description attribute."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="described_agent", description="Agent with description")

        @set_agent_attributes
        def forward(self):
            return "result"

        result = forward(agent)

        assert result == "result"
        # Use assert_any_call since set_attribute is called multiple times
        mock_span.set_attribute.assert_any_call(
            "gen_ai.agent.description", "Agent with description"
        )

    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    def test_decorator_captures_non_stream_response(self, mock_attrs, mock_get_span):
        """Test that decorator captures non-streaming responses."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="response_agent")

        @set_agent_attributes
        def forward(self):
            return "Agent response text"

        result = forward(agent)

        assert result == "Agent response text"
        mock_attrs.set_agent_response.assert_called_with("Agent response text")

    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    def test_decorator_skips_stream_response(self, mock_attrs, mock_get_span):
        """Test that decorator does not capture ModelStreamResponse."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="stream_agent")
        stream_response = ModelStreamResponse()

        @set_agent_attributes
        def forward(self):
            return stream_response

        result = forward(agent)

        assert result == stream_response
        # Should NOT call set_agent_response for streaming
        mock_attrs.set_agent_response.assert_not_called()

    @patch('msgflux.telemetry.span.trace.get_current_span')
    def test_decorator_when_span_not_recording(self, mock_get_span):
        """Test decorator when span is not recording."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="test_agent")

        @set_agent_attributes
        def forward(self):
            return "result"

        result = forward(agent)

        # Should still return result but not set attributes
        assert result == "result"

    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    def test_decorator_without_description(self, mock_attrs, mock_get_span):
        """Test decorator when agent has no description."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="no_desc_agent", description=None)

        @set_agent_attributes
        def forward(self):
            return "result"

        result = forward(agent)

        assert result == "result"
        # Should still set name but not description
        mock_attrs.set_agent_name.assert_called_with("no_desc_agent")


class TestAsetAgentAttributes:
    """Test suite for aset_agent_attributes async decorator."""

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    async def test_async_decorator_sets_agent_name_and_id(self, mock_attrs, mock_get_span):
        """Test that async decorator sets agent name and generates ID."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="async_agent", description="Async test agent")

        @aset_agent_attributes
        async def aforward(self, query):
            return f"Async response to: {query}"

        result = await aforward(agent, "async query")

        assert result == "Async response to: async query"
        mock_attrs.set_agent_name.assert_called_with("async_agent")
        # Verify agent ID was set
        call_args = mock_attrs.set_agent_id.call_args
        assert call_args is not None
        agent_id = call_args[0][0]
        assert agent_id.startswith("asst_")

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.trace.get_current_span')
    async def test_async_decorator_sets_agent_description(self, mock_get_span):
        """Test that async decorator sets agent description attribute."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="async_described_agent", description="Async description")

        @aset_agent_attributes
        async def aforward(self):
            return "result"

        result = await aforward(agent)

        assert result == "result"
        # Use assert_any_call since set_attribute is called multiple times
        mock_span.set_attribute.assert_any_call(
            "gen_ai.agent.description", "Async description"
        )

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    async def test_async_decorator_captures_non_stream_response(self, mock_attrs, mock_get_span):
        """Test that async decorator captures non-streaming responses."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="async_response_agent")

        @aset_agent_attributes
        async def aforward(self):
            return "Async agent response"

        result = await aforward(agent)

        assert result == "Async agent response"
        mock_attrs.set_agent_response.assert_called_with("Async agent response")

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.trace.get_current_span')
    @patch('msgflux.telemetry.span.MsgTraceAttributes')
    async def test_async_decorator_skips_stream_response(self, mock_attrs, mock_get_span):
        """Test that async decorator does not capture ModelStreamResponse."""
        mock_span = Mock()
        mock_span.is_recording.return_value = True
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="async_stream_agent")
        stream_response = ModelStreamResponse()

        @aset_agent_attributes
        async def aforward(self):
            return stream_response

        result = await aforward(agent)

        assert result == stream_response
        # Should NOT call set_agent_response for streaming
        mock_attrs.set_agent_response.assert_not_called()

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.trace.get_current_span')
    async def test_async_decorator_when_span_not_recording(self, mock_get_span):
        """Test async decorator when span is not recording."""
        mock_span = Mock()
        mock_span.is_recording.return_value = False
        mock_get_span.return_value = mock_span

        agent = MockAgent(name="test_agent")

        @aset_agent_attributes
        async def aforward(self):
            return "result"

        result = await aforward(agent)

        # Decorator should still work without recording
        assert result == "result"
