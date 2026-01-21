import asyncio
from typing import List

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
except ImportError:
    AsyncWebCrawler = None
    BrowserConfig = None
    CacheMode = None
    CrawlerRunConfig = None

from msgflux.data.retrievers.base import BaseRetriever, BaseWebFetch
from msgflux.data.retrievers.registry import register_retriever
from msgflux.data.retrievers.types import WebRetriever
from msgflux.dotdict import dotdict
from msgflux.logger import init_logger

logger = init_logger(__name__)


@register_retriever
class Crawl4AIWebRetriever(BaseWebFetch, BaseRetriever, WebRetriever):
    """A Crawl4AI client for fetching and extracting content from web pages.

    This class uses Crawl4AI's async web crawler to fetch pages and
    automatically convert HTML to LLM-friendly Markdown.
    """

    provider = "crawl4ai"

    def __init__(
        self,
        *,
        headless: bool = True,
        use_cache: bool = False,
        verbose: bool = False,
    ):
        """Initialize Crawl4AIWebRetriever.

        Args:
            headless:
                Whether to run browser in headless mode.
                Defaults to True.
            use_cache:
                Whether to use caching for crawled content.
                Defaults to False (fresh content on each request).
            verbose:
                Whether to enable verbose logging.
                Defaults to False.

        !!! example

            ```python
            retriever = Crawl4AIWebRetriever(headless=True)
            results = retriever(["https://example.com"], top_k=1)
            print(results)
            ```
        """
        if AsyncWebCrawler is None:
            raise ImportError(
                "The 'crawl4ai' package is not installed. "
                "Please install it via pip: pip install crawl4ai"
            )

        self.headless = headless
        self.use_cache = use_cache
        self.verbose = verbose

    def _get_browser_config(self) -> "BrowserConfig":
        """Create browser configuration."""
        return BrowserConfig(
            headless=self.headless,
            verbose=self.verbose,
        )

    def _get_run_config(self) -> "CrawlerRunConfig":
        """Create crawler run configuration."""
        cache_mode = CacheMode.ENABLED if self.use_cache else CacheMode.BYPASS
        return CrawlerRunConfig(cache_mode=cache_mode)

    def _parse_result(self, url: str, result) -> dict:
        """Parse Crawl4AI result into standard format."""
        # Get markdown content
        markdown_content = ""
        if hasattr(result, "markdown"):
            if hasattr(result.markdown, "raw_markdown"):
                markdown_content = result.markdown.raw_markdown
            elif isinstance(result.markdown, str):
                markdown_content = result.markdown

        return {
            "data": {
                "url": url,
                "content": markdown_content,
                "title": getattr(result, "title", None),
                "success": getattr(result, "success", True),
            }
        }

    async def _asingle_fetch(self, url: str) -> dict:
        """Async internal method to fetch a single URL."""
        try:
            browser_config = self._get_browser_config()
            run_config = self._get_run_config()

            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                return self._parse_result(url, result)
        except Exception as e:
            logger.warning("Crawl4AI fetch failed for URL '%s': %s", url, e)
            return {
                "data": {
                    "url": url,
                    "content": None,
                    "title": None,
                    "success": False,
                    "error": str(e),
                }
            }

    def _single_fetch(self, url: str) -> dict:
        """Synchronous wrapper for single URL fetch."""
        return asyncio.run(self._asingle_fetch(url))

    def _fetch(self, urls: List[str]) -> List[dotdict]:
        """Synchronous fetch for multiple URLs."""
        results = []
        for url in urls:
            result = self._single_fetch(url)
            results.append(dotdict({"results": [result]}))
        return results

    async def _afetch(self, urls: List[str]) -> List[dotdict]:
        """Async fetch that runs multiple URLs in parallel."""
        tasks = [self._asingle_fetch(url) for url in urls]
        fetch_results = await asyncio.gather(*tasks)
        results = []
        for result in fetch_results:
            results.append(dotdict({"results": [result]}))
        return results
