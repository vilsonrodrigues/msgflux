"""Tests for msgflux.nn.modules.embedder module."""

import pytest
from unittest.mock import Mock

from msgflux.nn.modules.embedder import Embedder


class TestEmbedder:
    """Test suite for Embedder module."""

    def test_embedder_initialization(self):
        """Test Embedder basic initialization."""
        mock_model = Mock()
        mock_model.model_type = "text_embedder"
        embedder = Embedder(model=mock_model)

        assert embedder.model is mock_model

    def test_embedder_initialization_with_config(self):
        """Test Embedder initialization with configuration."""
        mock_model = Mock()
        mock_model.model_type = "text_embedder"
        config = {"dimensions": 1536, "encoding_format": "float"}
        embedder = Embedder(model=mock_model, config=config)

        # Config is stored in dotdict buffer
        assert embedder._buffers["config"]["dimensions"] == 1536

    def test_embedder_inheritance_from_module(self):
        """Test that Embedder inherits from Module."""
        from msgflux.nn.modules.module import Module
        assert issubclass(Embedder, Module)

    def test_embedder_with_pooling_strategy(self):
        """Test Embedder with pooling strategy."""
        mock_model = Mock()
        mock_model.model_type = "audio_embedder"
        config = {"pooling_strategy": "mean"}
        embedder = Embedder(model=mock_model, config=config)

        assert embedder._buffers["config"]["pooling_strategy"] == "mean"

    def test_embedder_invalid_model_type(self):
        """Test Embedder raises TypeError for invalid model type."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        with pytest.raises(TypeError, match="requires be `embedder` model"):
            Embedder(model=mock_model)
