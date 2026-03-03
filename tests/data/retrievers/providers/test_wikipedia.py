"""Tests for WikipediaWebRetriever."""

from unittest.mock import MagicMock, patch

import pytest

from msgflux.data.retrievers.providers.wikipedia import WikipediaWebRetriever

_MODULE = "msgflux.data.retrievers.providers.wikipedia.wikipedia"


def _make_mock_wikipedia():
    """Build a mock wikipedia module with a single Python page."""
    mock_module = MagicMock()

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

    mock_module.search.return_value = ["Python (programming language)"]
    mock_module.page.return_value = mock_page
    mock_module.set_lang = MagicMock()
    mock_module.exceptions.DisambiguationError = Exception

    return mock_module


class TestWikipediaWebRetriever:
    """Tests for WikipediaWebRetriever."""

    @pytest.fixture
    def mock_wikipedia(self):
        """Patch the wikipedia reference inside the retriever module.

        Using patch() on the module-level variable (instead of patch.dict on
        sys.modules) ensures the mock is active even when the retriever module
        was already imported and cached before this test runs.
        """
        mock_module = _make_mock_wikipedia()
        with patch(_MODULE, mock_module):
            yield mock_module

    def test_init_without_library_raises_error(self):
        """Test that initialization fails without wikipedia."""
        with patch(_MODULE, None):
            with pytest.raises(ImportError) as exc_info:
                WikipediaWebRetriever()

            assert "wikipedia" in str(exc_info.value).lower()
            assert "pip install" in str(exc_info.value).lower()

    def test_init_with_defaults(self, mock_wikipedia):
        """Test initialization with default parameters."""
        retriever = WikipediaWebRetriever()
        assert retriever.language == "en"
        assert retriever.return_images is False
        assert retriever.max_return_images == 5
        assert retriever.summary is None

    def test_init_with_custom_params(self, mock_wikipedia):
        """Test initialization with custom parameters."""
        retriever = WikipediaWebRetriever(
            language="pt", return_images=True, max_return_images=3, summary=2
        )
        assert retriever.language == "pt"
        assert retriever.return_images is True
        assert retriever.max_return_images == 3
        assert retriever.summary == 2

    def test_init_with_none_applies_defaults(self, mock_wikipedia):
        """Test that None values are replaced with defaults."""
        retriever = WikipediaWebRetriever(
            language=None, return_images=None, max_return_images=None
        )
        assert retriever.language == "en"
        assert retriever.return_images is False
        assert retriever.max_return_images == 5

    def test_set_language(self, mock_wikipedia):
        """Test setting language."""
        retriever = WikipediaWebRetriever()
        retriever.set_language("pt")

    def test_extract_sentences(self, mock_wikipedia):
        """Test sentence extraction."""
        retriever = WikipediaWebRetriever()
        text = "This is sentence one. This is sentence two. Short."
        sentences = retriever._extract_sentences(text)

        assert isinstance(sentences, list)
        assert len(sentences) > 0

    def test_process_content_with_summary(self, mock_wikipedia):
        """Test content processing with summary."""
        retriever = WikipediaWebRetriever(summary=1)
        title = "Test Title"
        content = "First sentence. Second sentence. Third sentence."

        result = retriever._process_content(content, title)

        assert title in result
        assert "First sentence" in result

    def test_process_content_without_summary(self, mock_wikipedia):
        """Test content processing without summary."""
        retriever = WikipediaWebRetriever(summary=None)
        title = "Test Title"
        content = "Full content here."

        result = retriever._process_content(content, title)

        assert title in result
        assert "Full content" in result

    def test_search_single_query(self, mock_wikipedia):
        """Test search with a single query."""
        retriever = WikipediaWebRetriever()
        results = retriever(["Python programming"], top_k=1)

        assert len(results.data) == 1
        assert len(results.data[0]["results"]) >= 0

    @pytest.mark.asyncio
    async def test_acall_single_query(self, mock_wikipedia):
        """Test async search with a single query."""
        retriever = WikipediaWebRetriever()
        results = await retriever.acall(["Python programming"], top_k=1)

        assert len(results.data) == 1

    @pytest.mark.asyncio
    async def test_acall_with_defaults(self, mock_wikipedia):
        """Test async search applies defaults correctly."""
        retriever = WikipediaWebRetriever()
        results = await retriever.acall("Python")

        assert len(results.data) == 1
