import importlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the perplexity module BEFORE importing the retriever
mock_perplexity_module = MagicMock()
mock_perplexity_module.Perplexity = MagicMock()
mock_perplexity_module.AsyncPerplexity = MagicMock()
sys.modules["perplexity"] = mock_perplexity_module

# Now reload the retriever module to pick up the mock
if "msgflux.data.retrievers.providers.perplexity" in sys.modules:
    importlib.reload(sys.modules["msgflux.data.retrievers.providers.perplexity"])

from msgflux.data.retrievers.providers.perplexity import PerplexityWebRetriever


@pytest.fixture
def mock_clients():
    sync_client = MagicMock()
    async_client = MagicMock()
    async_client.search = MagicMock()
    async_client.search.create = AsyncMock()
    sync_client.search = MagicMock()
    mock_perplexity_module.Perplexity.return_value = sync_client
    mock_perplexity_module.AsyncPerplexity.return_value = async_client
    return sync_client, async_client


@pytest.fixture
def retriever(mock_clients):
    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        return PerplexityWebRetriever()


def test_init_defaults(retriever):
    assert retriever.country is None


def test_init_custom_params(mock_clients):
    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        retriever = PerplexityWebRetriever(country="us")
        assert retriever.country == "us"


@pytest.mark.asyncio
async def test_search(mock_clients):
    sync_client, async_client = mock_clients

    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        retriever = PerplexityWebRetriever()

        # Mock response
        mock_result = MagicMock()
        mock_result.title = "Test Title"
        mock_result.url = "https://example.com"
        mock_result.snippet = "Test snippet"
        mock_result.date = "2024-01-15"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        async_client.search.create.return_value = mock_response

        results = await retriever.acall("test query", top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert len(results.data[0].results) == 1

        result_data = results.data[0].results[0]["data"]
        assert result_data["title"] == "Test Title"
        assert result_data["url"] == "https://example.com"
        assert result_data["content"] == "Test snippet"
        assert result_data["date"] == "2024-01-15"


@pytest.mark.asyncio
async def test_search_with_country_uppercase(mock_clients):
    """Test that country is converted to uppercase."""
    sync_client, async_client = mock_clients

    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        retriever = PerplexityWebRetriever(country="br")

        mock_response = MagicMock()
        mock_response.results = []
        async_client.search.create.return_value = mock_response

        await retriever.acall("query", top_k=5)

        call_kwargs = async_client.search.create.call_args[1]
        assert call_kwargs["country"] == "BR"
        assert call_kwargs["max_results"] == 5


def test_sync_search(mock_clients):
    sync_client, async_client = mock_clients

    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        retriever = PerplexityWebRetriever()

        mock_result = MagicMock()
        mock_result.title = "Sync Test"
        mock_result.url = "https://sync.com"
        mock_result.snippet = "Sync content"

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        sync_client.search.create.return_value = mock_response

        results = retriever(["sync query"], top_k=1)

        assert results.response_type == "web_search"
        assert len(results.data) == 1
        assert results.data[0].results[0]["data"]["title"] == "Sync Test"


def test_init_raises_without_api_key(mock_clients):
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            PerplexityWebRetriever()
