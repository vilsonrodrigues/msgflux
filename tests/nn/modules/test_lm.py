"""Tests for msgflux.nn.modules.generator module."""

from unittest.mock import Mock

from msgflux.nn.modules.generator import Generator
from msgflux.models.response import ModelResponse


class TestGenerator:
    """Test suite for Generator (universal model wrapper) module."""

    def test_initialization(self):
        """Test Generator basic initialization."""
        mock_model = Mock()
        gen = Generator(model=mock_model)

        assert gen.model is mock_model

    def test_forward(self):
        """Test Generator forward method."""
        mock_model = Mock()
        mock_response = ModelResponse()
        mock_response.data = "response"
        mock_model.return_value = mock_response

        gen = Generator(model=mock_model)
        result = gen(messages=[{"role": "user", "content": "Hello"}])

        assert result.data == "response"
        mock_model.assert_called_once()

    def test_inheritance_from_module(self):
        """Test that Generator inherits from Module."""
        from msgflux.nn.modules.module import Module

        assert issubclass(Generator, Module)
