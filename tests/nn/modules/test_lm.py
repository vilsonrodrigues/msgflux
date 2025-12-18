"""Tests for msgflux.nn.modules.lm module."""

import pytest
from unittest.mock import Mock, MagicMock

from msgflux.nn.modules.lm import LM
from msgflux.models.response import ModelResponse


class TestLM:
    """Test suite for LM (Language Model) module."""

    def test_lm_initialization(self):
        """Test LM basic initialization."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        lm = LM(model=mock_model)

        assert lm.model is mock_model

    def test_lm_forward_basic(self):
        """Test LM forward method with basic prompt."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"
        mock_response = ModelResponse()
        mock_response.data = "Generated text response"
        mock_model.return_value = mock_response

        lm = LM(model=mock_model)
        result = lm(messages=[{"role": "user", "content": "Hello"}])

        assert result.data == "Generated text response"
        mock_model.assert_called_once()

    def test_lm_invalid_model_type(self):
        """Test LM raises TypeError for invalid model type."""
        mock_model = Mock()
        mock_model.model_type = "text_embedding"

        with pytest.raises(TypeError, match="must be a `chat_completion` model"):
            LM(model=mock_model)

    def test_lm_inheritance_from_module(self):
        """Test that LM inherits from Module."""
        from msgflux.nn.modules.module import Module
        assert issubclass(LM, Module)
