"""Tests for RankBM25LexicalRetriever."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


class TestRankBM25LexicalRetriever:
    """Tests for RankBM25LexicalRetriever."""

    @pytest.fixture
    def mock_rank_bm25(self):
        """Mock the rank_bm25 module."""
        mock_bm25 = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.get_scores.return_value = np.array([2.5, 1.8, 0.5])
        mock_bm25.BM25Okapi.return_value = mock_bm25_instance

        with patch.dict("sys.modules", {"rank_bm25": mock_bm25}):
            # Re-import to get the mocked version
            from msgflux.data.retrievers.providers import rank_bm25

            # Reload the module
            import importlib

            importlib.reload(rank_bm25)
            yield rank_bm25.RankBM25LexicalRetriever

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without rank_bm25."""
        with patch.dict("sys.modules", {"rank_bm25": None}):
            # Force reimport
            if "msgflux.data.retrievers.providers.rank_bm25" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.rank_bm25"]

            from msgflux.data.retrievers.providers.rank_bm25 import (
                RankBM25LexicalRetriever,
            )

            with pytest.raises(ImportError) as exc_info:
                RankBM25LexicalRetriever()

            assert "rank_bm25" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_with_defaults(self, mock_rank_bm25):
        """Test initialization with default parameters."""
        retriever = mock_rank_bm25()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75

    def test_init_with_custom_params(self, mock_rank_bm25):
        """Test initialization with custom parameters."""
        retriever = mock_rank_bm25(k1=2.0, b=0.5)
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5

    def test_init_with_none_applies_defaults(self, mock_rank_bm25):
        """Test that None values are replaced with defaults."""
        retriever = mock_rank_bm25(k1=None, b=None)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75

    def test_add_documents(self, mock_rank_bm25):
        """Test adding documents to the index."""
        retriever = mock_rank_bm25()
        documents = ["hello world", "python programming"]
        retriever.add(documents)

        assert len(retriever.documents) == 2
        assert retriever.bm25 is not None

    def test_tokenize(self, mock_rank_bm25):
        """Test tokenization."""
        retriever = mock_rank_bm25()
        tokens = retriever._tokenize("Hello World Python")

        assert tokens == ["hello", "world", "python"]

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_rank_bm25):
        """Test async search applies defaults correctly."""
        retriever = mock_rank_bm25()
        documents = ["python", "java", "rust"]
        retriever.add(documents)

        results = await retriever.acall("python")

        assert len(results.data) == 1
