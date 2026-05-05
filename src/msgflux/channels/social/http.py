from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import msgspec

from msgflux.channels.exceptions import ChannelError


@dataclass(frozen=True)
class SocialHttpConfig:
    timeout_s: float = 10.0
    connect_timeout_s: Optional[float] = None
    read_timeout_s: Optional[float] = None
    write_timeout_s: Optional[float] = None
    pool_timeout_s: Optional[float] = None
    max_connections: Optional[int] = None
    max_keepalive_connections: Optional[int] = None
    keepalive_expiry_s: Optional[float] = None

    def timeout(self) -> httpx.Timeout:
        granular_timeouts = (
            self.connect_timeout_s,
            self.read_timeout_s,
            self.write_timeout_s,
            self.pool_timeout_s,
        )
        if not any(timeout is not None for timeout in granular_timeouts):
            return httpx.Timeout(self.timeout_s)

        return httpx.Timeout(
            timeout=self.timeout_s,
            connect=_timeout_or_default(self.connect_timeout_s, self.timeout_s),
            read=_timeout_or_default(self.read_timeout_s, self.timeout_s),
            write=_timeout_or_default(self.write_timeout_s, self.timeout_s),
            pool=_timeout_or_default(self.pool_timeout_s, self.timeout_s),
        )

    def limits(self) -> httpx.Limits:
        kwargs: Dict[str, Any] = {}
        if self.max_connections is not None:
            kwargs["max_connections"] = self.max_connections
        if self.max_keepalive_connections is not None:
            kwargs["max_keepalive_connections"] = self.max_keepalive_connections
        if self.keepalive_expiry_s is not None:
            kwargs["keepalive_expiry"] = self.keepalive_expiry_s
        return httpx.Limits(**kwargs)


class SocialHttpClient:
    def __init__(self, config: Optional[SocialHttpConfig] = None) -> None:
        self.config = config or SocialHttpConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout(),
                limits=self.config.limits(),
            )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        response = await self.post(
            url,
            payload,
            headers=headers,
        )
        result = msgspec.json.decode(response.content) if response.content else {}
        if not isinstance(result, Mapping):
            raise ChannelError(f"HTTP response from `{url}` must be a JSON object")
        return dict(result)

    async def post(
        self,
        url: str,
        payload: Dict[str, Any],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> httpx.Response:
        if self._client is None:
            async with httpx.AsyncClient(
                timeout=self.config.timeout(),
                limits=self.config.limits(),
            ) as client:
                return await _post_json(client, url, payload, headers=headers)
        return await _post_json(self._client, url, payload, headers=headers)


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Mapping[str, str]] = None,
) -> httpx.Response:
    response = await client.post(
        url,
        content=msgspec.json.encode(payload),
        headers=dict(headers or {}),
    )
    response.raise_for_status()
    return response


def _timeout_or_default(value: Optional[float], default: float) -> float:
    return default if value is None else value
