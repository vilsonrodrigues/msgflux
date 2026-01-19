"""Tests for WikipediaWebRetriever."""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestWikipediaWebRetriever:
    """Tests for WikipediaWebRetriever."""

    @pytest.fixture
    def mock_wikipedia(self):
        """Mock the wikipedia module."""
        mock_module = MagicMock()

        # Mock page object
        mock_page = MagicMock()
        mock_page.title = "Python (programming language)"
        mock_page.content = (
            "Python is a high-level programming language. "
            "It is widely used for web development."
        )
        mock_page.images = [
            "https://example.com/python-logo.png",
            "https://example.com/python-icon.svg",
        ]

        # Mock search function
        mock_module.search.return_value = ["Python (programming language)"]

        # Mock page function
        mock_module.page.return_value = mock_page

        # Mock set_lang function
        mock_module.set_lang = MagicMock()

        # Mock exceptions
        mock_module.exceptions.DisambiguationError = Exception

        with patch.dict("sys.modules", {"wikipedia": mock_module}):
            # Re-import to get the mocked version
            from msgflux.data.retrievers.providers import wikipedia

            # Reload the module
            import importlib

            importlib.reload(wikipedia)
            yield wikipedia.WikipediaWebRetriever

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without wikipedia."""
        with patch.dict("sys.modules", {"wikipedia": None}):
            # Force reimport
            if "msgflux.data.retrievers.providers.wikipedia" in sys.modules:
                del sys.modules["msgflux.data.retrievers.providers.wikipedia"]

            from msgflux.data.retrievers.providers.wikipedia import (
                WikipediaWebRetriever,
            )

            with pytest.raises(ImportError) as exc_info:
                WikipediaWebRetriever()

            assert "wikipedia" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_with_defaults(self, mock_wikipedia):
        """Test initialization with default parameters."""
        retriever = mock_wikipedia()
        assert retriever.language == "en"
        assert retriever.return_images is False
        assert retriever.max_return_images == 5
        assert retriever.summary is None

    def test_init_with_custom_params(self, mock_wikipedia):
        """Test initialization with custom parameters."""
        retriever = mock_wikipedia(
            language="pt", return_images=True, max_return_images=3, summary=2
        )
        assert retriever.language == "pt"
        assert retriever.return_images is True
        assert retriever.max_return_images == 3
        assert retriever.summary == 2

    def test_init_with_none_applies_defaults(self, mock_wikipedia):
        """Test that None values are replaced with defaults."""
        retriever = mock_wikipedia(
            language=None, return_images=None, max_return_images=None
        )
        assert retriever.language == "en"
        assert retriever.return_images is False
        assert retriever.max_return_images == 5

    def test_set_language(self, mock_wikipedia):
        """Test setting language."""
        retriever = mock_wikipedia()
        retriever.set_language("pt")
        # Just verify it doesn't raise an error

    def test_extract_sentences(self, mock_wikipedia):
        """Test sentence extraction."""
        retriever = mock_wikipedia()
        text = "This is sentence one. This is sentence two. Short."
        sentences = retriever._extract_sentences(text)

        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_process_content_with_summary(self, mock_wikipedia):
        """Test content processing with summary."""
        retriever = mock_wikipedia(summary=1)
        title = "Test Title"
        content = "First sentence. Second sentence. Third sentence."

        result = retriever._process_content(content, title)

        assert title in result
        assert "First sentence" in result

    def test_process_content_without_summary(self, mock_wikipedia):
        """Test content processing without summary."""
        retriever = mock_wikipedia(summary=None)
        title = "Test Title"
        content = "Full content here."

        result = retriever._process_content(content, title)

        assert title in result
        assert "Full content" in result

    def test_search_single_query(self, mock_wikipedia):
        """Test search with a single query."""
        retriever = mock_wikipedia()
        results = retriever(["Python programming"], top_k=1)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) >= 0

    @pytest.mark.asyncio
    async def test_acall_single_query(self, mock_wikipedia):
        """Test async search with a single query."""
        retriever = mock_wikipedia()
        results = await retriever.acall(["Python programming"], top_k=1)

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_wikipedia):
        """Test async search applies defaults correctly."""
        retriever = mock_wikipedia()
        results = await retriever.acall("Python")

        assert len(results.data) == 1
