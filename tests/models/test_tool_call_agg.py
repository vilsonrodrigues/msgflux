"""Tests for msgflux.models.tool_call_agg module."""

import json

import pytest

from msgflux.models.tool_call_agg import ToolCallAggregator


class TestToolCallAggregatorBasics:
    """Test suite for basic ToolCallAggregator functionality."""

    def test_aggregator_initialization(self):
        """Test ToolCallAggregator initialization."""
        agg = ToolCallAggregator()
        assert agg.reasoning is None
        assert len(agg.tool_calls) == 0

    def test_aggregator_initialization_with_reasoning(self):
        """Test ToolCallAggregator initialization with reasoning."""
        agg = ToolCallAggregator(reasoning="Let me think about this...")
        assert agg.reasoning == "Let me think about this..."
        assert len(agg.tool_calls) == 0

    def test_process_single_tool_call(self):
        """Test processing a single complete tool call."""
        agg = ToolCallAggregator()
        agg.process(
            call_index=0,
            tool_id="call_123",
            name="get_weather",
            arguments='{"location": "NYC"}',
        )

        assert len(agg.tool_calls) == 1
        assert 0 in agg.tool_calls
        assert agg.tool_calls[0]["id"] == "call_123"
        assert agg.tool_calls[0]["name"] == "get_weather"
        assert agg.tool_calls[0]["arguments"] == '{"location": "NYC"}'

    def test_process_incremental_arguments(self):
        """Test processing tool call with incremental arguments (streaming)."""
        agg = ToolCallAggregator()

        # First chunk
        agg.process(call_index=0, tool_id="", name="calculate", arguments='{"op')

        # Second chunk
        agg.process(
            call_index=0, tool_id="call_456", name="", arguments='eration": "add'
        )

        # Third chunk
        agg.process(call_index=0, tool_id="", name="", arguments='", "a": 5, "b": 3}')

        assert len(agg.tool_calls) == 1
        assert agg.tool_calls[0]["id"] == "call_456"
        assert agg.tool_calls[0]["name"] == "calculate"
        assert agg.tool_calls[0]["arguments"] == '{"operation": "add", "a": 5, "b": 3}'

    def test_thought_signature_preserved_in_get_messages(self):
        """Test that thought_signature is preserved through get_messages."""
        agg = ToolCallAggregator()
        agg.process(
            call_index=0,
            tool_id="call_1",
            name="get_weather",
            arguments='{"city": "NYC"}',
        )
        agg.tool_calls[0]["thought_signature"] = "sig_from_gemini"

        agg.insert_results({"call_1": "sunny and warm"})
        messages = agg.get_messages()

        # First message: assistant with tool_calls
        tc = messages[0]["tool_calls"][0]
        assert tc["thought_signature"] == "sig_from_gemini"

    def test_thought_signature_absent_when_not_set(self):
        """Test that thought_signature is not added when not present."""
        agg = ToolCallAggregator()
        agg.process(
            call_index=0,
            tool_id="call_1",
            name="get_weather",
            arguments='{"city": "NYC"}',
        )

        agg.insert_results({"call_1": "sunny"})
        messages = agg.get_messages()

        tc = messages[0]["tool_calls"][0]
        assert "thought_signature" not in tc
