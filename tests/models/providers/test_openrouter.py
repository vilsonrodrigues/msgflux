"""Tests for msgflux.models.providers.openrouter module."""

from unittest.mock import patch
from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages


class TestOpenRouterChatCompletion:
    """Test suite for OpenRouterChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with (
            patch("msgflux.models.providers.openai.OpenAI") as mock_client,
            patch("msgflux.models.providers.openai.AsyncOpenAI") as mock_async_client,
        ):
            yield mock_client, mock_async_client

    def test_chat_completion_with_reasoning_max_tokens(self, mock_openai_client):
        """Test OpenRouter forwards reasoning_max_tokens as reasoning.max_tokens."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="openrouter/anthropic/claude-sonnet-4.5",
            reasoning_max_tokens=2000,
        )

        assert model.sampling_run_params["reasoning_max_tokens"] == 2000

        params = {
            "messages": [],
            "model": model.model_id,
            "tool_choice": None,
            "tools": None,
            "web_search_options": None,
            "extra_body": {},
            "extra_headers": {},
            **model.sampling_run_params,
        }

        adapted = model._adapt_params(params)

        assert adapted["extra_body"]["reasoning"]["max_tokens"] == 2000

    def test_adapt_params_accepts_requests_without_tool_keys(self, mock_openai_client):
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="nvidia/test-model")
        params = model._adapt_params(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": model.model_id,
            }
        )

        assert params["tool_choice"] == "none"
        assert params["extra_body"] == {}

    def test_responses_api_mode_is_not_inherited_implicitly(self, mock_openai_client):
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        with pytest.raises(ValueError, match="does not support"):
            OpenRouterChatCompletion(
                model_id="openai/gpt-oss-120b",
                api_mode="responses",
            )

    def test_chat_completion_with_reasoning_effort(self, mock_openai_client):
        """Test OpenRouter still forwards reasoning_effort."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(
            model_id="openrouter/anthropic/claude-sonnet-4.5",
            reasoning_effort="high",
        )

        params = {
            "messages": [],
            "model": model.model_id,
            "tool_choice": None,
            "tools": None,
            "web_search_options": None,
            "extra_body": {},
            "extra_headers": {},
            **model.sampling_run_params,
        }

        adapted = model._adapt_params(params)

        assert adapted["extra_body"]["reasoning"]["effort"] == "high"

    def test_model_converts_canonical_messages_for_selected_provider(
        self, mock_openai_client
    ):
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="openai/gpt-oss-120b")
        assert model.api_mode == "chat_completions"
        assert model.reasoning_codec.name == "openrouter_reasoning_details"
        details = [{"type": "reasoning.encrypted", "data": "opaque"}]
        messages = ChatMessages()
        messages.add_reasoning(
            "summary",
            provider="openrouter",
            provider_state=details,
        )
        messages.add_assistant("answer")

        params = model._build_generation_params(
            messages=messages,
            system_prompt=None,
            prefilling=None,
            tool_definitions=None,
        )

        assert params["messages"] == [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "summary",
                "reasoning_details": details,
            }
        ]

    def test_response_state_records_provider_api_and_codec(self, mock_openai_client):
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        model = OpenRouterChatCompletion(model_id="openai/gpt-oss-120b")
        details = [{"type": "reasoning.encrypted", "data": "opaque"}]
        output = SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    logprobs=None,
                    message=SimpleNamespace(
                        content="answer",
                        reasoning_content="summary",
                        reasoning_details=details,
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )

        response = model._process_completion_model_output(output)

        assert response.history_items[0]["provider_state"] == {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "codec": "openrouter_reasoning_details",
            "data": details,
        }

    def test_chat_completion_rejects_reasoning_effort_with_max_tokens(
        self, mock_openai_client
    ):
        """Test OpenRouter rejects reasoning_effort and reasoning_max_tokens together."""
        pytest.importorskip("openai")

        from msgflux.models.providers.openrouter import OpenRouterChatCompletion

        with pytest.raises(
            ValueError,
            match="`reasoning_max_tokens` cannot be used together with",
        ):
            OpenRouterChatCompletion(
                model_id="openrouter/anthropic/claude-sonnet-4.5",
                reasoning_effort="high",
                reasoning_max_tokens=2000,
            )
