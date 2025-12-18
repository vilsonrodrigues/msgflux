"""Tests for msgflux.telemetry.span tool decorators."""

import pytest
from unittest.mock import Mock, patch

from msgflux.telemetry.span import set_tool_attributes, aset_tool_attributes


class TestSetToolAttributes:
    """Test suite for set_tool_attributes decorator."""

    @patch('msgflux.telemetry.span.get_current_span')
    def test_decorator_sets_tool_name(self, mock_get_span):
        """Test that decorator sets tool name attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @set_tool_attributes(name="test_tool")
        def test_function():
            return "result"
        
        result = test_function()
        
        assert result == "result"
        mock_span.set_attribute.assert_called()

    @patch('msgflux.telemetry.span.get_current_span')
    def test_decorator_sets_tool_description(self, mock_get_span):
        """Test that decorator sets tool description attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @set_tool_attributes(name="test_tool", description="Test tool description")
        def test_function():
            return "result"
        
        result = test_function()
        
        assert result == "result"


class TestAsetToolAttributes:
    """Test suite for aset_tool_attributes async decorator."""

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.get_current_span')
    async def test_async_decorator_sets_tool_name(self, mock_get_span):
        """Test that async decorator sets tool name attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @aset_tool_attributes(name="test_async_tool")
        async def test_async_function():
            return "async_result"
        
        result = await test_async_function()
        
        assert result == "async_result"
        mock_span.set_attribute.assert_called()
