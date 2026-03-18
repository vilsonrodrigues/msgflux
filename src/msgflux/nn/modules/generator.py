from typing import Any, Union

from msgflux.auto import AutoParams
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.modules.module import Module


class Generator(Module, metaclass=AutoParams):
    """Universal model wrapper — enables hooks for any model type."""

    def __init__(self, model: Any):
        super().__init__()
        self.register_buffer("model", model)

    def forward(self, **kwargs: Any) -> Union[ModelResponse, ModelStreamResponse]:
        return self.model(**kwargs)

    async def aforward(
        self, **kwargs: Any
    ) -> Union[ModelResponse, ModelStreamResponse]:
        return await self.model.acall(**kwargs)
