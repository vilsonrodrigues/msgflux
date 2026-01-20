"""Tests for BM25SLexicalRetriever."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Try to import the real module
try:
    import bm25s

    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

# Import the retriever only if bm25s is available
if HAS_BM25S:
    from msgflux.data.retrievers.providers.bm25s import BM25SLexicalRetriever


class TestBM25SWithoutLibrary:
    """Tests for behavior when bm25s is not installed."""

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without bm25s."""
        with patch.dict("sys.modules", {"bm25s": None}):
            # Force reimport
            if "msgflux.data.retrievers.providers.bm25s" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.bm25s"]

            from msgflux.data.retrievers.providers.bm25s import (
                BM25SLexicalRetriever as RetrieverWithoutLib,
            )

            with pytest.raises(ImportError) as exc_info:
                RetrieverWithoutLib()

            assert "bm25s" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()


@pytest.mark.skipif(not HAS_BM25S, reason="bm25s not installed")
class TestBM25SLexicalRetriever:
    """Tests for BM25SLexicalRetriever with real library."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        retriever = BM25SLexicalRetriever()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.method == "lucene"
        assert retriever.stopwords is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        retriever = BM25SLexicalRetriever(
            k1=2.0, b=0.5, method="robertson", stopwords="en"
        )
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5
        assert retriever.method == "robertson"
        assert retriever.stopwords == "en"

    def test_init_with_none_applies_defaults(self):
        """Test that None values are replaced with defaults."""
        retriever = BM25SLexicalRetriever(k1=None, b=None, method=None)
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.method == "lucene"

    def test_add_documents(self):
        """Test adding documents to the index."""
        retriever = BM25SLexicalRetriever()
        documents = ["hello world", "python programming"]
        retriever.add(documents)

        assert len(retriever.documents) == 2
        assert retriever.bm25 is not None

    def test_search_with_return_score(self):
        """Test that scores are returned when requested."""
        retriever = BM25SLexicalRetriever()
        documents = ["python programming", "java programming", "machine learning"]
        retriever.add(documents)

        results = retriever(["python"], top_k=2, return_score=True)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) > 0
        first_result = results.data[0]["results"][0]
        assert hasattr(first_result, "score")
        assert isinstance(first_result.score, float)

    @pytest.mark.asyncio
    async def test_acall_single_query(self):
        """Test async search with a single query."""
        retriever = BM25SLexicalRetriever()
        documents = ["python programming", "java programming"]
        retriever.add(documents)

        results = await retriever.acall(["python"], top_k=1)

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self):
        """Test async search applies defaults correctly."""
        retriever = BM25SLexicalRetriever()
        documents = ["python", "java", "rust"]
        retriever.add(documents)

        results = await retriever.acall("python")

        assert len(results.data) == 1
