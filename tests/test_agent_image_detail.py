"""Test Agent image_detail parameter integration."""
import pytest
from unittest.mock import Mock, patch
from msgflux.nn.modules.agent import Agent
from msgflux.message import Message


class TestAgentImageDetail:
    """Test suite for Agent image_detail parameter."""

    def test_agent_init_with_image_detail_high(self):
        """Test Agent initialization with image_detail='high'."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            image_detail="high"
        )

        assert agent.image_detail == "high"

    def test_agent_init_with_image_detail_low(self):
        """Test Agent initialization with image_detail='low'."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            image_detail="low"
        )

        assert agent.image_detail == "low"

    def test_agent_init_without_image_detail(self):
        """Test Agent initialization without image_detail (defaults to None)."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model
        )

        assert agent.image_detail is None

    def test_agent_init_with_invalid_image_detail(self):
        """Test Agent initialization with invalid image_detail raises ValueError."""
        model = Mock()
        model.model_type = "chat_completion"

        with pytest.raises(ValueError, match="image_detail"):
            Agent(
                name="test_agent",
                model=model,
                image_detail="invalid"
            )

    @patch('msgflux.nn.modules.agent.ChatBlock')
    def test_format_image_input_passes_detail_high(self, mock_chatblock):
        """Test that _format_image_input passes detail='high' to ChatBlock.image."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            image_detail="high"
        )

        # Mock the return value
        mock_chatblock.image.return_value = {
            "type": "image_url",
            "image_url": {
                "url": "http://example.com/image.jpg",
                "detail": "high"
            }
        }

        # Mock _prepare_data_uri to return a URL
        with patch.object(agent, '_prepare_data_uri', return_value="http://example.com/image.jpg"):
            result = agent._format_image_input("http://example.com/image.jpg")

        # Verify ChatBlock.image was called with detail="high"
        mock_chatblock.image.assert_called_once_with(
            "http://example.com/image.jpg",
            detail="high"
        )

    @patch('msgflux.nn.modules.agent.ChatBlock')
    def test_format_image_input_passes_detail_low(self, mock_chatblock):
        """Test that _format_image_input passes detail='low' to ChatBlock.image."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            image_detail="low"
        )

        # Mock the return value
        mock_chatblock.image.return_value = {
            "type": "image_url",
            "image_url": {
                "url": "http://example.com/image.jpg",
                "detail": "low"
            }
        }

        # Mock _prepare_data_uri to return a URL
        with patch.object(agent, '_prepare_data_uri', return_value="http://example.com/image.jpg"):
            result = agent._format_image_input("http://example.com/image.jpg")

        # Verify ChatBlock.image was called with detail="low"
        mock_chatblock.image.assert_called_once_with(
            "http://example.com/image.jpg",
            detail="low"
        )

    @patch('msgflux.nn.modules.agent.ChatBlock')
    def test_format_image_input_passes_detail_none(self, mock_chatblock):
        """Test that _format_image_input passes detail=None when not set."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model
        )

        # Mock the return value
        mock_chatblock.image.return_value = {
            "type": "image_url",
            "image_url": {
                "url": "http://example.com/image.jpg"
            }
        }

        # Mock _prepare_data_uri to return a URL
        with patch.object(agent, '_prepare_data_uri', return_value="http://example.com/image.jpg"):
            result = agent._format_image_input("http://example.com/image.jpg")

        # Verify ChatBlock.image was called with detail=None
        mock_chatblock.image.assert_called_once_with(
            "http://example.com/image.jpg",
            detail=None
        )
