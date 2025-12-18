"""Tests for msgflux.nn.modules.retriever module."""

import pytest
from unittest.mock import Mock, MagicMock

from msgflux.nn.modules.retriever import Retriever
from msgflux.data.retrievers.types import LexicalRetriever


class MockLexicalRetriever(LexicalRetriever):
    """Mock lexical retriever for testing."""

    def __call__(self, query, **kwargs):
        return []


class TestRetriever:
    """Test suite for Retriever module."""

    def test_retriever_initialization(self):
        """Test Retriever basic initialization."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        assert ret.retriever is mock_retriever

    def test_retriever_initialization_with_config(self):
        """Test Retriever initialization with configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 5, "threshold": 0.7}
        ret = Retriever(retriever=mock_retriever, config=config)

        assert ret.retriever is mock_retriever
        assert ret._buffers["config"]["top_k"] == 5

    def test_retriever_inheritance_from_module(self):
        """Test that Retriever inherits from Module."""
        from msgflux.nn.modules.module import Module
        assert issubclass(Retriever, Module)

    def test_retriever_with_top_k(self):
        """Test Retriever with top_k configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 3}
        ret = Retriever(retriever=mock_retriever, config=config)

        assert ret._buffers["config"]["top_k"] == 3
