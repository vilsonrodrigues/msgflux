import asyncio
from typing import Any, AsyncGenerator


class CoreResponse:
    def set_metadata(self, metadata: Any) -> None:
        self.metadata = metadata

    def set_response_type(self, response_type: str) -> None:
        if isinstance(response_type, str):
            self.response_type = response_type
        else:
            raise TypeError(
                f"`response_type` requires str, given `{type(response_type)}`"
            )


class BaseResponse(CoreResponse):
    """Base response for non-streaming model outputs."""

    def __init__(self) -> None:
        self.data: Any = None
        self.reasoning: str | None = None
        self.metadata: Any = None
        self.response_type: str | None = None

    def add(self, data: Any) -> None:
        """Add content data to the response."""
        self.data = data

    def add_reasoning(self, reasoning: str) -> None:
        """Add reasoning/thinking content to the response."""
        if self.reasoning is None:
            self.reasoning = reasoning
        else:
            self.reasoning += reasoning

    def consume(self) -> Any:
        """Return the response data."""
        return self.data


class BaseStreamResponse(CoreResponse):
    """Base response for streaming model outputs with separate content and reasoning."""

    def __init__(self) -> None:
        self.first_chunk_event: asyncio.Event = asyncio.Event()
        self.data: Any = None
        self.reasoning: str | None = None
        self.content_queue: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self.reasoning_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.metadata: Any = None
        self.response_type: str | None = None
        self._content_finished: bool = False
        self._reasoning_finished: bool = False

    @property
    def queue(self) -> asyncio.Queue[str | bytes | None]:
        """Alias for content_queue (used by some providers)."""
        return self.content_queue

    def add(self, data: str | bytes | None) -> None:
        """Add content chunk to the content queue."""
        self.add_content(data)

    def add_content(self, data: str | bytes | None) -> None:
        """Add content chunk to the content queue.

        Pass None to signal end of content stream.
        """
        self.content_queue.put_nowait(data)
        if data is None:
            self._content_finished = True

    def add_reasoning(self, data: str | None) -> None:
        """Add reasoning/thinking chunk to the reasoning queue.

        Pass None to signal end of reasoning stream.
        """
        self.reasoning_queue.put_nowait(data)
        if data is None:
            self._reasoning_finished = True

    async def consume(self) -> AsyncGenerator[str | bytes, None]:
        """Async generator that yields content chunks.

        Delegates to consume_content() for backward compatibility.
        """
        async for chunk in self.consume_content():
            yield chunk

    async def consume_content(self) -> AsyncGenerator[str | bytes, None]:
        """Async generator that yields content chunks until stream ends."""
        while (chunk := await self.content_queue.get()) is not None:
            yield chunk

    async def consume_reasoning(self) -> AsyncGenerator[str, None]:
        """Async generator that yields reasoning chunks until stream ends."""
        while (chunk := await self.reasoning_queue.get()) is not None:
            yield chunk

    @property
    def is_content_finished(self) -> bool:
        """Check if content stream has finished."""
        return self._content_finished

    @property
    def is_reasoning_finished(self) -> bool:
        """Check if reasoning stream has finished."""
        return self._reasoning_finished

    @property
    def is_finished(self) -> bool:
        """Check if both streams have finished."""
        return self._content_finished and self._reasoning_finished
