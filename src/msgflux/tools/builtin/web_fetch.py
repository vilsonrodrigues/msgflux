from urllib.parse import urlparse

try:
    import httpx
except ImportError:
    httpx = None


class WebFetch:
    """Fetches web pages as Markdown via a remote parser endpoint."""

    name = "web_fetch"

    def __init__(
        self,
        web_parser: str = "https://markdown.new/",
        default_headers: dict | None = None,
        timeout: int = 1,
    ):
        """Args:
        web_parser: Parser endpoint URL (must include scheme).
        default_headers: Headers applied to every request.
        timeout: Request timeout in seconds.
        """
        self.web_parser = web_parser.rstrip("/") + "/"
        self.default_headers = default_headers or {}
        self.timeout = timeout

    def _build_url(self, url: str) -> str:
        parsed = urlparse(url)
        target = url if parsed.scheme else f"https://{url}"
        return f"{self.web_parser}{target}"

    def __call__(self, url: str) -> str:
        """Fetch and return web content as a string.

        Args:
            url: Target website URL.

        Returns:
            Parsed response body as a string.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: On network or HTTP failure.
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for WebFetch."
                " Install it with: pip install msgflux[httpx]"
            )

        final_url = self._build_url(url)
        try:
            response = httpx.get(
                final_url,
                headers=self.default_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

    async def acall(self, url: str) -> str:
        """Asynchronously fetch and return web content as a string.

        Args:
            url: Target website URL.

        Returns:
            Parsed response body as a string.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: On network or HTTP failure.
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for WebFetch."
                " Install it with: pip install msgflux[httpx]"
            )

        final_url = self._build_url(url)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(final_url, headers=self.default_headers)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
