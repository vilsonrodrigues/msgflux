"""Protocol boundary for chat-model wire APIs."""

from __future__ import annotations

from typing import Any


class ChatAPIAdapter:
    """Translate the shared chat runtime to one wire API.

    Adapters are stateless and may be shared by model instances. Providers
    select adapters by ``api_mode``; the model remains responsible for common
    lifecycle concerns such as caching, timing, aborts, and response metadata.
    """

    name: str
    endpoint: str
    canonical_history: bool = False

    def prepare_request(self, owner: Any, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def build_generation_params(self, owner: Any, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def process_output(self, owner: Any, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def stream(self, owner: Any, **kwargs: Any):
        raise NotImplementedError

    async def astream(self, owner: Any, **kwargs: Any):
        raise NotImplementedError


class ChatTransport:
    """Send prepared requests without owning protocol conversion."""

    def create(
        self,
        owner: Any,
        api: ChatAPIAdapter,
        params: dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    async def acreate(
        self,
        owner: Any,
        api: ChatAPIAdapter,
        params: dict[str, Any],
    ) -> Any:
        raise NotImplementedError
