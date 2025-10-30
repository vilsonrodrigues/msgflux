from typing import Any

from msgflux.nn.modules.module import Module


class LM(Module):
    """Language Model wrapper - enables hooks and composition for LM calls."""

    def __init__(self, model):
        """Initialize LM wrapper.

        Args:
            model:
                A msgflux Model instance (e.g., from mf.Model.chat_completion()).

        Raises:
            TypeError:
                If model is not a chat completion model.
        """
        super().__init__()

        # Validate model type
        if not hasattr(model, "model_type") or model.model_type != "chat_completion":
            raise TypeError(
                f"`model` must be a `chat_completion` model, given `{type(model)}`"
            )

        self.model = model

    def forward(self, **kwargs) -> Any:
        """Forward call to underlying model.

        Args:
            **kwargs: Arguments passed to the model (messages, tools, etc.)

        Returns:
            Model response
        """
        return self.model(**kwargs)

    async def aforward(self, **kwargs) -> Any:
        """Async forward call to underlying model.

        Args:
            **kwargs: Arguments passed to the model (messages, tools, etc.)

        Returns:
            Model response
        """
        return await self.model.acall(**kwargs)
