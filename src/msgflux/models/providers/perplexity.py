from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BasePerplexity:
    """Configurations to use Perplexity Sonar models via OpenAI-compatible API."""

    provider: str = "perplexity"

    def _get_base_url(self):
        base_url = getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
        if base_url is None:
            raise ValueError("Please set `PERPLEXITY_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("PERPLEXITY_API_KEY")
        if not key:
            raise ValueError(
                "The Perplexity API key is not available. "
                "Please set `PERPLEXITY_API_KEY`"
            )
        return key


@register_model
class PerplexityChatCompletion(_BasePerplexity, OpenAIChatCompletion):
    """Perplexity Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters for Perplexity API.

        Perplexity supports most OpenAI parameters plus additional
        search-specific parameters via extra_body.
        """
        # Perplexity uses max_tokens, not max_completion_tokens
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")

        # Perplexity doesn't support tool_choice or tools
        params.pop("tool_choice", None)
        params.pop("tools", None)

        return params
