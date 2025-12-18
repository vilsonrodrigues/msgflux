"""Tests for msgflux.telemetry.span agent decorators."""

import pytest
from unittest.mock import Mock, patch

from msgflux.telemetry.span import set_agent_attributes, aset_agent_attributes


class TestSetAgentAttributes:
    """Test suite for set_agent_attributes decorator."""

    @patch('msgflux.telemetry.span.get_current_span')
    def test_decorator_sets_agent_name(self, mock_get_span):
        """Test that decorator sets agent name attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @set_agent_attributes(name="test_agent")
        def test_agent_function():
            return "agent_result"
        
        result = test_agent_function()
        
        assert result == "agent_result"
        mock_span.set_attribute.assert_called()

    @patch('msgflux.telemetry.span.get_current_span')
    def test_decorator_sets_agent_description(self, mock_get_span):
        """Test that decorator sets agent description attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @set_agent_attributes(name="test_agent", description="Test agent description")
        def test_agent_function():
            return "agent_result"
        
        result = test_agent_function()
        
        assert result == "agent_result"


class TestAsetAgentAttributes:
    """Test suite for aset_agent_attributes async decorator."""

    @pytest.mark.asyncio
    @patch('msgflux.telemetry.span.get_current_span')
    async def test_async_decorator_sets_agent_name(self, mock_get_span):
        """Test that async decorator sets agent name attribute."""
        mock_span = Mock()
        mock_get_span.return_value = mock_span
        
        @aset_agent_attributes(name="test_async_agent")
        async def test_async_agent_function():
            return "async_agent_result"
        
        result = await test_async_agent_function()
        
        assert result == "async_agent_result"
        mock_span.set_attribute.assert_called()
