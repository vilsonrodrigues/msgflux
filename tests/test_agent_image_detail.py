"""Test Agent image_detail parameter integration."""

from unittest.mock import Mock, patch

import pytest

from msgflux.core.message import Message
from msgflux.nn.modules.agent import Agent


class TestAgentImageDetail:
    """Test suite for Agent image_detail parameter."""

    def test_agent_init_with_image_detail_high(self):
        """Test Agent initialization with image_detail='high'."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            config={"image_block_kwargs": {"detail": "high"}},
        )

        assert agent.config["image_block_kwargs"]["detail"] == "high"

    def test_agent_init_with_image_detail_low(self):
        """Test Agent initialization with image_detail='low'."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            config={"image_block_kwargs": {"detail": "low"}},
        )

        assert agent.config["image_block_kwargs"]["detail"] == "low"

    def test_agent_init_without_image_detail(self):
        """Test Agent initialization without image_detail (defaults to None)."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(name="test_agent", model=model)

        assert agent.config.get("image_block_kwargs") is None

    def test_agent_init_with_invalid_image_detail(self):
        """Test Agent initialization with invalid image_detail - Agent doesn't validate, just passes to ChatBlock."""
        model = Mock()
        model.model_type = "chat_completion"

        # O Agent não valida o valor de detail, apenas passa para o ChatBlock
        agent = Agent(
            name="test_agent",
            model=model,
            config={"image_block_kwargs": {"detail": "invalid"}},
        )

        # O agente deve ser criado sem erro, o ChatBlock é quem validará o valor
        assert agent.config["image_block_kwargs"]["detail"] == "invalid"

    def test_format_image_input_passes_detail_high(self):
        """Test that _format_image_input passes detail='high' via image_block_kwargs."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            config={"image_block_kwargs": {"detail": "high"}},
        )

        result = agent._format_image_input("http://example.com/image.jpg")

        assert result["image_url"]["detail"] == "high"

    def test_format_image_input_passes_detail_low(self):
        """Test that _format_image_input passes detail='low' via image_block_kwargs."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(
            name="test_agent",
            model=model,
            config={"image_block_kwargs": {"detail": "low"}},
        )

        result = agent._format_image_input("http://example.com/image.jpg")

        assert result["image_url"]["detail"] == "low"

    def test_format_image_input_passes_detail_none(self):
        """Test that _format_image_input has no detail when not set."""
        model = Mock()
        model.model_type = "chat_completion"

        agent = Agent(name="test_agent", model=model)

        result = agent._format_image_input("http://example.com/image.jpg")

        assert "detail" not in result["image_url"]
