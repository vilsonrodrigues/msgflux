"""Direct HTTP transport for provider chat APIs."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from json import JSONDecodeError, loads
from os import getenv
from typing import Any

from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.chat_api import ChatTransport, PreparedChatRequest

try:
    import httpx2 as _httpx2
except ImportError:
    _httpx2 = None

try:
    import httpx as _legacy_httpx
except ImportError:
    _legacy_httpx = None

httpx = _httpx2 or _legacy_httpx
_HTTP_STATUS_ERRORS = tuple(
    module.HTTPStatusError for module in (_httpx2, _legacy_httpx) if module is not None
)
_REQUEST_ERRORS = tuple(
    module.RequestError for module in (_httpx2, _legacy_httpx) if module is not None
)

_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, *range(500, 600)}


class HTTPChatTransport(ChatTransport):
    """Send chat requests directly over JSON HTTP and SSE.

    Clients may be injected for tests or custom networking. Otherwise each
    transport instance lazily creates and owns one synchronous and one
    asynchronous client.
    """

    def __init__(
        self,
        *,
        client: Any = None,
        async_client: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_retry_delay: float = 8.0,
    ) -> None:
        self._client = client
        self._async_client = async_client
        self._owns_client = client is None
        self._owns_async_client = async_client is None
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_delay = max_retry_delay

    def create(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._stream(owner, request)
        response = self._request(owner, request)
        return owner.api_adapter.decode_response(response.json())

    async def acreate(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._astream(owner, request)
        response = await self._arequest(owner, request)
        return owner.api_adapter.decode_response(response.json())

    def close(self, _owner: Any) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    async def aclose(self, _owner: Any) -> None:
        if self._async_client is not None and self._owns_async_client:
            await self._async_client.aclose()
            self._async_client = None

    def _request(self, owner: Any, request: PreparedChatRequest):
        client = self._get_client()
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            owner._raise_if_aborted()
            try:
                response = client.request(
                    request.method,
                    self._url(owner, request),
                    headers=self._headers(owner, request),
                    json=request.json,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    _raise_for_status(response, owner)
                    return response
                if attempt == attempts:
                    _raise_for_status(response, owner)
                    return response
                self._wait(owner, self._retry_delay(attempt, response.headers))
            except _HTTP_STATUS_ERRORS:
                raise
            except _REQUEST_ERRORS:
                if attempt == attempts:
                    raise
                self._wait(owner, self._retry_delay(attempt))
        raise RuntimeError("HTTP retry loop ended without a response")

    async def _arequest(self, owner: Any, request: PreparedChatRequest):
        client = self._get_async_client()
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            owner._raise_if_aborted()
            try:
                response = await client.request(
                    request.method,
                    self._url(owner, request),
                    headers=await self._aheaders(owner, request),
                    json=request.json,
                )
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    _raise_for_status(response, owner)
                    return response
                if attempt == attempts:
                    _raise_for_status(response, owner)
                    return response
                await self._await(
                    owner,
                    self._retry_delay(attempt, response.headers),
                )
            except _HTTP_STATUS_ERRORS:
                raise
            except _REQUEST_ERRORS:
                if attempt == attempts:
                    raise
                await self._await(owner, self._retry_delay(attempt))
        raise RuntimeError("HTTP retry loop ended without a response")

    def _stream(self, owner: Any, request: PreparedChatRequest) -> Iterator[Any]:
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            emitted = False
            owner._raise_if_aborted()
            try:
                with self._get_client().stream(
                    request.method,
                    self._url(owner, request),
                    headers=self._headers(owner, request),
                    json=request.json,
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        if attempt < attempts:
                            self._wait(
                                owner,
                                self._retry_delay(attempt, response.headers),
                            )
                            continue
                    _raise_for_status(response, owner)
                    for payload in _iter_sse_json(response.iter_lines()):
                        owner._raise_if_aborted()
                        emitted = True
                        yield owner.api_adapter.decode_stream_event(payload)
                    return
            except _HTTP_STATUS_ERRORS:
                raise
            except _REQUEST_ERRORS:
                if emitted or attempt == attempts:
                    raise
                self._wait(owner, self._retry_delay(attempt))

    async def _astream(
        self, owner: Any, request: PreparedChatRequest
    ) -> AsyncIterator[Any]:
        attempts = self._max_retries()
        for attempt in range(attempts + 1):
            emitted = False
            owner._raise_if_aborted()
            try:
                async with self._get_async_client().stream(
                    request.method,
                    self._url(owner, request),
                    headers=await self._aheaders(owner, request),
                    json=request.json,
                ) as response:
                    if response.status_code in _RETRYABLE_STATUS_CODES:
                        if attempt < attempts:
                            await self._await(
                                owner,
                                self._retry_delay(attempt, response.headers),
                            )
                            continue
                    _raise_for_status(response, owner)
                    async for payload in _aiter_sse_json(response.aiter_lines()):
                        owner._raise_if_aborted()
                        emitted = True
                        yield owner.api_adapter.decode_stream_event(payload)
                    return
            except _HTTP_STATUS_ERRORS:
                raise
            except _REQUEST_ERRORS:
                if emitted or attempt == attempts:
                    raise
                await self._await(owner, self._retry_delay(attempt))

    def _headers(self, owner: Any, request: PreparedChatRequest) -> dict[str, str]:
        credentials = owner.credential_resolver.resolve(owner)
        return _merge_headers(
            {"accept": "application/json", "content-type": "application/json"},
            request.headers,
            credentials.headers,
        )

    async def _aheaders(
        self, owner: Any, request: PreparedChatRequest
    ) -> dict[str, str]:
        credentials = await owner.credential_resolver.aresolve(owner)
        return _merge_headers(
            {"accept": "application/json", "content-type": "application/json"},
            request.headers,
            credentials.headers,
        )

    def _url(self, owner: Any, request: PreparedChatRequest) -> str:
        base_url = owner.sampling_params.get("base_url")
        if not base_url:
            base_url = "https://api.openai.com/v1"
        return f"{str(base_url).rstrip('/')}/{request.endpoint.lstrip('/')}"

    def _get_client(self):
        self._require_httpx()
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout(),
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                verify=self._verify_ssl(),
            )
        return self._client

    def _get_async_client(self):
        self._require_httpx()
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._timeout(),
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                verify=self._verify_ssl(),
            )
        return self._async_client

    @staticmethod
    def _require_httpx() -> None:
        if httpx is None:
            raise ImportError(
                "`httpx` is required for direct chat transport. "
                "Install it with `pip install msgflux[httpx]`."
            )

    def _timeout(self) -> float | None:
        if self.timeout is not None:
            return self.timeout
        value = getenv("OPENAI_TIMEOUT")
        return float(value) if value else None

    def _max_retries(self) -> int:
        if self.max_retries is not None:
            return max(0, self.max_retries)
        value = getenv("OPENAI_MAX_RETRIES")
        return max(0, int(value)) if value else 2

    @staticmethod
    def _verify_ssl() -> bool:
        return getenv("OPENAI_SSL_VERIFY", "true").lower() not in {
            "0",
            "false",
            "no",
        }

    def _retry_delay(self, attempt: int, headers: Any = None) -> float:
        retry_after = _retry_after_seconds(headers)
        if retry_after is not None:
            return min(retry_after, self.max_retry_delay)
        base = min(0.25 * (2**attempt), self.max_retry_delay)
        jitter = random.uniform(0, base * 0.25)  # noqa: S311 - retry jitter
        return min(base + jitter, self.max_retry_delay)

    @staticmethod
    def _wait(owner: Any, delay: float) -> None:
        deadline = time.monotonic() + delay
        while True:
            owner._raise_if_aborted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.05))

    @staticmethod
    async def _await(owner: Any, delay: float) -> None:
        deadline = time.monotonic() + delay
        while True:
            owner._raise_if_aborted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 0.05))


def _merge_headers(*values: dict[str, str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    names: dict[str, str] = {}
    for value in values:
        for key, item in value.items():
            normalized = key.casefold()
            previous = names.get(normalized)
            if previous is not None:
                merged.pop(previous)
            names[normalized] = key
            merged[key] = item
    return merged


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def _raise_for_status(response: Any, owner: Any) -> None:
    try:
        response.raise_for_status()
    except _HTTP_STATUS_ERRORS as exc:
        error = _structured_error(response)
        raise ModelProviderHTTPError(
            status_code=response.status_code,
            description=error["description"],
            provider=getattr(owner, "provider", None),
            model_id=getattr(owner, "model_id", None),
            code=error.get("code"),
            error_type=error.get("type"),
            param=error.get("param"),
            request_id=_request_id(response.headers),
            response=response,
        ) from exc


def _structured_error(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        description = error.get("message") or error.get("detail")
        code = error.get("code")
        error_type = error.get("type") or error.get("status")
        param = error.get("param")
    elif isinstance(error, str):
        description = error
        code = None
        error_type = None
        param = None
    elif isinstance(payload, dict):
        description = payload.get("message") or payload.get("detail")
        code = payload.get("code")
        error_type = payload.get("type") or payload.get("status")
        param = payload.get("param")
    else:
        description = None
        code = None
        error_type = None
        param = None
    if not isinstance(description, str) or not description:
        description = getattr(response, "reason_phrase", None) or "Request failed"
    return {
        "description": description[:4000],
        "code": code,
        "type": error_type if isinstance(error_type, str) else None,
        "param": param if isinstance(param, str) else None,
    }


def _request_id(headers: Any) -> str | None:
    for name in ("x-request-id", "request-id", "x-groq-request-id", "cf-ray"):
        value = headers.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(
                value[1:] if separator and value.startswith(" ") else value
            )
    if data_lines:
        yield "\n".join(data_lines)


async def _aiter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field == "data":
            data_lines.append(
                value[1:] if separator and value.startswith(" ") else value
            )
    if data_lines:
        yield "\n".join(data_lines)


def _iter_sse_json(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for data in _iter_sse_data(lines):
        if data.strip() == "[DONE]":
            return
        yield _decode_sse_json(data)


async def _aiter_sse_json(
    lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    async for data in _aiter_sse_data(lines):
        if data.strip() == "[DONE]":
            return
        yield _decode_sse_json(data)


def _decode_sse_json(data: str) -> dict[str, Any]:
    try:
        payload = loads(data)
    except JSONDecodeError as exc:
        raise ValueError("Provider returned invalid SSE JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Provider SSE data must be a JSON object")
    return payload
