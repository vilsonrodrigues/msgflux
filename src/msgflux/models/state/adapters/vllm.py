"""vLLM Adapter for message format conversion."""

from typing import Any

from msgflux.models.state.adapters.openai import OpenAIChatAdapter
from msgflux.models.state.types import AudioContent


class VLLMAdapter(OpenAIChatAdapter):
    """Adapter for vLLM servers.

    Inherits from OpenAI adapter with minor differences:
    - Prefers audio URLs over base64 content
    """

    def _convert_audio_content(self, block: AudioContent) -> dict[str, Any]:
        """Convert audio content block - vLLM prefers audio_url."""
        if block.url:
            return {"type": "audio_url", "audio_url": {"url": block.url}}
        if block.base64:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": block.base64,
                    "format": block.format or "wav",
                },
            }
        return {}
