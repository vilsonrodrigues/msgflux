import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the crawl4ai module BEFORE importing the retriever
mock_crawl4ai_module = MagicMock()
mock_crawl4ai_module.AsyncWebCrawler = MagicMock()
mock_crawl4ai_module.BrowserConfig = MagicMock()
mock_crawl4ai_module.CacheMode = MagicMock()
mock_crawl4ai_module.CacheMode.ENABLED = "ENABLED"
mock_crawl4ai_module.CacheMode.BYPASS = "BYPASS"
mock_crawl4ai_module.CrawlerRunConfig = MagicMock()
sys.modules["crawl4ai"] = mock_crawl4ai_module

# Reload the retriever module to pick up the mock
if "msgflux.data.retrievers.providers.crawl4ai" in sys.modules:
    importlib.reload(sys.modules["msgflux.data.retrievers.providers.crawl4ai"])

from msgflux.data.retrievers.providers.crawl4ai import Crawl4AIWebRetriever


@pytest.fixture
def mock_crawler():
    crawler_instance = MagicMock()
    crawler_instance.arun = AsyncMock()
    mock_crawl4ai_module.AsyncWebCrawler.return_value.__aenter__ = AsyncMock(
        return_value=crawler_instance
    )
    mock_crawl4ai_module.AsyncWebCrawler.return_value.__aexit__ = AsyncMock(
        return_value=None
    )
    return crawler_instance


@pytest.fixture
def retriever():
    return Crawl4AIWebRetriever()


def test_init_defaults(retriever):
    assert retriever.headless is True
    assert retriever.use_cache is False
    assert retriever.verbose is False


def test_init_custom_params():
    retriever = Crawl4AIWebRetriever(
        headless=False,
        use_cache=True,
        verbose=True,
    )
    assert retriever.headless is False
    assert retriever.use_cache is True
    assert retriever.verbose is True


@pytest.mark.asyncio
async def test_fetch_single_url(mock_crawler):
    retriever = Crawl4AIWebRetriever()

    # Mock result
    mock_result = MagicMock()
    mock_result.markdown = MagicMock()
    mock_result.markdown.raw_markdown = "# Example Page\n\nThis is content."
    mock_result.title = "Example Page"
    mock_result.success = True

    mock_crawler.arun.return_value = mock_result

    results = await retriever.acall("https://example.com")

    assert results.response_type == "web_fetch"
    assert len(results.data) == 1
    assert len(results.data[0].results) == 1

    result_data = results.data[0].results[0]["data"]
    assert result_data["url"] == "https://example.com"
    assert result_data["content"] == "# Example Page\n\nThis is content."
    assert result_data["title"] == "Example Page"
    assert result_data["success"] is True


@pytest.mark.asyncio
async def test_fetch_multiple_urls(mock_crawler):
    retriever = Crawl4AIWebRetriever()

    mock_result = MagicMock()
    mock_result.markdown = MagicMock()
    mock_result.markdown.raw_markdown = "Content"
    mock_result.title = "Page"
    mock_result.success = True

    mock_crawler.arun.return_value = mock_result

    results = await retriever.acall(
        ["https://example.com", "https://python.org"]
    )

    assert results.response_type == "web_fetch"
    assert len(results.data) == 2


@pytest.mark.asyncio
async def test_fetch_with_string_markdown(mock_crawler):
    """Test when markdown is a string instead of object."""
    retriever = Crawl4AIWebRetriever()

    mock_result = MagicMock()
    mock_result.markdown = "# Direct markdown string"
    mock_result.title = "Page"
    mock_result.success = True

    mock_crawler.arun.return_value = mock_result

    results = await retriever.acall("https://example.com")

    result_data = results.data[0].results[0]["data"]
    assert result_data["content"] == "# Direct markdown string"


def test_sync_fetch(mock_crawler):
    retriever = Crawl4AIWebRetriever()

    mock_result = MagicMock()
    mock_result.markdown = "Sync content"
    mock_result.title = "Sync Page"
    mock_result.success = True

    mock_crawler.arun.return_value = mock_result

    results = retriever(["https://example.com"])

    assert results.response_type == "web_fetch"
    assert len(results.data) == 1
