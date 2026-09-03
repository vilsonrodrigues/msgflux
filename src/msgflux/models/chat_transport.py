"""Direct HTTP transport for provider chat APIs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from json import JSONDecodeError, loads
from typing import Any

from msgflux.models.chat_api import ChatTransport, PreparedChatRequest
from msgflux.models.http_transport import HTTPTransport, merge_headers


class HTTPChatTransport(ChatTransport):
    """Adapt prepared chat requests to the shared model HTTP transport."""

    def __init__(
        self,
        *,
        client: Any = None,
        async_client: Any = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_retry_delay: float = 8.0,
    ) -> None:
        self.http = HTTPTransport(
            client=client,
            async_client=async_client,
            timeout=timeout,
            max_retries=max_retries,
            max_retry_delay=max_retry_delay,
        )

    def create(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._stream(owner, request)
        response = self.http.request(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
        )
        return owner.api_adapter.decode_response(response.json())

    async def acreate(self, owner: Any, request: PreparedChatRequest) -> Any:
        if request.params.get("stream"):
            return self._astream(owner, request)
        response = await self.http.arequest(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
        )
        return owner.api_adapter.decode_response(response.json())

    def close(self, _owner: Any) -> None:
        self.http.close()

    async def aclose(self, _owner: Any) -> None:
        await self.http.aclose()

    def _stream(self, owner: Any, request: PreparedChatRequest) -> Iterator[Any]:
        for payload in self.http.stream(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
            iterate=lambda response: _iter_sse_json(response.iter_lines()),
        ):
            yield owner.api_adapter.decode_stream_event(payload)

    async def _astream(
        self,
        owner: Any,
        request: PreparedChatRequest,
    ) -> AsyncIterator[Any]:
        async for payload in self.http.astream(
            owner,
            request.endpoint,
            method=request.method,
            headers=self._headers(request),
            json=request.json,
            iterate=lambda response: _aiter_sse_json(response.aiter_lines()),
        ):
            yield owner.api_adapter.decode_stream_event(payload)

    @staticmethod
    def _headers(request: PreparedChatRequest) -> dict[str, str]:
        return merge_headers(
            {"accept": "application/json", "content-type": "application/json"},
            request.headers,
        )


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
