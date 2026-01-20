import asyncio
from os import getenv
from typing import List, Optional

try:
    from perplexity import AsyncPerplexity, Perplexity
except ImportError:
    Perplexity = None
    AsyncPerplexity = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebSearch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class PerplexityWebRetriever(BaseWebSearch, BaseRetriever, WebRetriever):
    """A Perplexity client for retrieving web search results.

    This class interfaces with the Perplexity Search API to provide
    ranked web search results with date information.
    """

    provider = "perplexity"

    def __init__(
        self,
        *,
        country: Optional[str] = None,
    ):
        """Initialize PerplexityWebRetriever.

        Requires the `PERPLEXITY_API_KEY` environment variable to be set.

        Args:
            country:
                ISO 3166-1 alpha-2 country code for localized results
                (e.g., "US", "GB", "BR"). Defaults to None.

        !!! example

            ```python
            retriever = PerplexityWebRetriever(country="US")
            results = retriever(["latest AI news"], top_k=5)
            print(results)
            ```
        """
        if Perplexity is None:
            raise ImportError(
                "The 'perplexity' package is not installed. "
                "Please install it via pip: pip install perplexity"
            )

        api_key = getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError(
                "The Perplexity API key is not available. "
                "Please set `PERPLEXITY_API_KEY`"
            )

        self.client = Perplexity(api_key=api_key)
        self.aclient = AsyncPerplexity(api_key=api_key)
        self.country = country

    def _build_search_kwargs(self, top_k: int) -> dict:
        """Build kwargs for Perplexity search methods."""
        kwargs = {
            "max_results": top_k,
        }

        if self.country:
            kwargs["country"] = self.country.upper()

        return kwargs

    def _parse_results(self, response) -> List[dict]:
        """Parse Perplexity response into standard format."""
        results = []

        if hasattr(response, "results"):
            for result in response.results:
                data = {
                    "title": getattr(result, "title", None),
                    "content": getattr(result, "snippet", None),
                    "url": getattr(result, "url", None),
                }

                # Add date if available
                if hasattr(result, "date") and result.date:
                    data["date"] = result.date

                results.append({"data": data})

        return results

    def _single_search(self, query: str, top_k: int) -> List[dict]:
        """Internal method to search Perplexity for a single query."""
        try:
            kwargs = self._build_search_kwargs(top_k)
            response = self.client.search.create(query=query, **kwargs)
            return self._parse_results(response)
        except Exception as e:
            logger.warning("Perplexity search failed for query '%s': %s", query, e)
            return []

    async def _asingle_search(self, query: str, top_k: int) -> List[dict]:
        """Async internal method to search Perplexity for a single query."""
        try:
            kwargs = self._build_search_kwargs(top_k)
            response = await self.aclient.search.create(query=query, **kwargs)
            return self._parse_results(response)
        except Exception as e:
            logger.warning("Perplexity search failed for query '%s': %s", query, e)
            return []

    def _search(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Synchronous search for multiple queries."""
        results = []
        for query in queries:
            query_results = self._single_search(query, top_k)
            results.append(dotdict({"results": query_results}))
        return results

    async def _asearch(self, queries: List[str], top_k: int) -> List[dotdict]:
        """Async search that runs multiple queries in parallel."""
        tasks = [self._asingle_search(query, top_k) for query in queries]
        query_results = await asyncio.gather(*tasks)
        results = []
        for result in query_results:
            results.append(dotdict({"results": result}))
        return results
