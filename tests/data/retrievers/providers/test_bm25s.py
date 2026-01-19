"""Tests for BM25SLexicalRetriever."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestBM25SLexicalRetriever:
    """Tests for BM25SLexicalRetriever."""

    @pytest.fixture
    def mock_bm25s(self):
        """Mock the bm25s module."""
        mock_module = MagicMock()

        # Mock tokenize function
        def mock_tokenize(texts, stopwords=None, return_ids=False):
            # Return tokenized texts
            return [text.lower().split() for text in texts]

        mock_module.tokenize = mock_tokenize

        # Mock BM25 class
        mock_bm25_instance = MagicMock()

        # Mock retrieve method
        def mock_retrieve(query_tokens, k=10):
            # Return mock doc_ids and scores
            n_docs = 3
            doc_ids = np.array([[0, 1, 2][:k]])
            scores = np.array([[2.5, 1.8, 0.5][:k]])
            return doc_ids, scores

        mock_bm25_instance.retrieve = mock_retrieve
        mock_bm25_class = MagicMock(return_value=mock_bm25_instance)
        mock_module.BM25 = mock_bm25_class

        with patch.dict("sys.modules", {"bm25s": mock_module, "numpy": np}):
            # Re-import to get the mocked version
            from msgflux.data.retrievers.providers import bm25s

            # Reload the module
            import importlib

            importlib.reload(bm25s)
            yield bm25s.BM25SLexicalRetriever

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without bm25s."""
        with patch.dict("sys.modules", {"bm25s": None}):
            # Force reimport
            if "msgflux.data.retrievers.providers.bm25s" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.bm25s"]

            from msgflux.data.retrievers.providers.bm25s import (
                BM25SLexicalRetriever,
            )

            with pytest.raises(ImportError) as exc_info:
                BM25SLexicalRetriever()

            assert "bm25s" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_with_defaults(self, mock_bm25s):
        """Test initialization with default parameters."""
        retriever = mock_bm25s()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.method == "lucene"
        assert retriever.stopwords is None

    def test_init_with_custom_params(self, mock_bm25s):
        """Test initialization with custom parameters."""
        retriever = mock_bm25s(k1=2.0, b=0.5, method="robertson", stopwords="en")
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5
        assert retriever.method == "robertson"
        assert retriever.stopwords == "en"

    def test_init_with_none_applies_defaults(self, mock_bm25s):
        """Test that None values are replaced with defaults."""
        retriever = mock_bm25s(k1=None, b=None, method=None)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.method == "lucene"

    def test_add_documents(self, mock_bm25s):
        """Test adding documents to the index."""
        retriever = mock_bm25s()
        documents = ["hello world", "python programming"]
        retriever.add(documents)

        assert len(retriever.documents) == 2
        assert retriever.bm25 is not None

    def test_search_with_return_score(self, mock_bm25s):
        """Test that scores are returned when requested."""
        retriever = mock_bm25s()
        documents = ["python programming", "java programming", "machine learning"]
        retriever.add(documents)

        results = retriever(["python"], top_k=2, return_score=True)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) > 0
        first_result = results.data[0]["results"][0]
        assert hasattr(first_result, "score")
        assert isinstance(first_result.score, float)

    @pytest.mark.asyncio
    async def test_acall_single_query(self, mock_bm25s):
        """Test async search with a single query."""
        retriever = mock_bm25s()
        documents = ["python programming", "java programming"]
        retriever.add(documents)

        results = await retriever.acall(["python"], top_k=1)

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_bm25s):
        """Test async search applies defaults correctly."""
        retriever = mock_bm25s()
        documents = ["python", "java", "rust"]
        retriever.add(documents)

        results = await retriever.acall("python")

        assert len(results.data) == 1
