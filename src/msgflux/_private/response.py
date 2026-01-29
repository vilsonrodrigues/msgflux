import asyncio
from typing import Any, AsyncGenerator, Union


class CoreResponse:
    def set_metadata(self, metadata: Any):
        self.metadata = metadata

    def set_response_type(self, response_type: str):
        if isinstance(response_type, str):
            self.response_type = response_type
        else:
            raise TypeError(
                f"`response_type` requires strgiven `{type(response_type)}`"
            )


class BaseResponse(CoreResponse):
    def __init__(self):
        self.data = None
        self.metadata = None
        self.response_type = None

    def add(self, data: Any):
        self.data = data

    def consume(self) -> Any:
        return self.data


class BaseStreamResponse(CoreResponse):
    def __init__(self):
        self.first_chunk_event = asyncio.Event()
        self.data = None
        self.content_queue: asyncio.Queue = asyncio.Queue()
        self.reasoning_queue: asyncio.Queue = asyncio.Queue()
        self.metadata = None
        self.response_type = None

    @property
    def queue(self) -> asyncio.Queue:
        """Backward compatibility: alias for content_queue."""
        return self.content_queue

    def add(self, data: Any):
        """Add data to the content queue. Backward compatible."""
        self.add_content(data)

    def add_content(self, data: Any):
        """Add content chunk to the content queue."""
        self.content_queue.put_nowait(data)

    def add_reasoning(self, data: Any):
        """Add reasoning/thinking chunk to the reasoning queue."""
        self.reasoning_queue.put_nowait(data)

    async def consume(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields content chunks. Backward compatible."""
        async for chunk in self.consume_content():
            yield chunk

    async def consume_content(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields content chunks until None is received."""
        while True:
            try:
                chunk = await asyncio.wait_for(self.content_queue.get(), timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                continue

    async def consume_reasoning(self) -> AsyncGenerator[Union[bytes, str], None]:
        """Async generator that yields reasoning chunks until None is received."""
        while True:
            try:
                chunk = await asyncio.wait_for(self.reasoning_queue.get(), timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except asyncio.TimeoutError:
                continue
