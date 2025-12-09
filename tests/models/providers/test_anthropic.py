"""Tests for msgflux.models.providers.anthropic module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestAnthropicProviderImport:
    """Test Anthropic provider import and initialization."""

    def test_anthropic_import_available(self):
        """Test that Anthropic provider imports correctly when dependencies are available."""
        try:
            from msgflux.models.providers.anthropic import AnthropicChatCompletion
            # If we get here, imports worked
            assert True
        except ImportError:
            pytest.skip("Anthropic dependencies not available")

    def test_anthropic_models_registered(self):
        """Test that Anthropic models are registered with @register_model."""
        pytest.importorskip("anthropic", reason="anthropic not installed")

        from msgflux.models.registry import model_registry

        # Check if Anthropic models are registered
        if "chat_completion" in model_registry:
            assert "anthropic" in model_registry.get("chat_completion", {})


class TestAnthropicChatCompletion:
    """Test suite for AnthropicChatCompletion."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch("msgflux.models.providers.anthropic.Anthropic") as mock_client, \
             patch("msgflux.models.providers.anthropic.AsyncAnthropic") as mock_async_client:
            yield mock_client, mock_async_client

    def test_chat_completion_initialization(self, mock_anthropic_client):
        """Test AnthropicChatCompletion initialization."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(model_id="claude-3-5-sonnet-20241022")

        assert model.model_id == "claude-3-5-sonnet-20241022"
        assert model.provider == "anthropic"
        assert model.model_type == "chat_completion"

    def test_chat_completion_with_parameters(self, mock_anthropic_client):
        """Test AnthropicChatCompletion with custom parameters."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

        assert model.sampling_run_params["max_tokens"] == 2000
        assert model.sampling_run_params["temperature"] == 0.7
        assert model.sampling_run_params["top_p"] == 0.9
        assert model.sampling_run_params["top_k"] == 50

    def test_chat_completion_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        pytest.importorskip("anthropic")

        # Remove API key
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        with pytest.raises(ValueError, match="Anthropic API key is not available"):
            AnthropicChatCompletion(model_id="claude-3-5-sonnet-20241022")

    def test_chat_completion_with_thinking_params(self, mock_anthropic_client):
        """Test AnthropicChatCompletion with thinking parameters."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            thinking={"type": "enabled", "budget_tokens": 10000},
            return_thinking=True,
            thinking_in_tool_call=True,
        )

        assert model.sampling_run_params.get("thinking") == {"type": "enabled", "budget_tokens": 10000}
        assert model.return_thinking is True
        assert model.thinking_in_tool_call is True

    def test_chat_completion_with_stop_sequences(self, mock_anthropic_client):
        """Test AnthropicChatCompletion with stop sequences."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        stop_sequences = ["\n\nHuman:", "\n\nAssistant:"]
        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            stop_sequences=stop_sequences,
        )

        assert model.sampling_run_params.get("stop_sequences") == stop_sequences

    def test_chat_completion_convert_messages(self, mock_anthropic_client):
        """Test message conversion for Anthropic format."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(model_id="claude-3-5-sonnet-20241022")

        # Test with system prompt extraction
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        system_prompt, converted = model._convert_messages(messages)

        assert system_prompt == "You are a helpful assistant."
        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[1]["role"] == "assistant"

    def test_chat_completion_validate_typed_parser_output(self, mock_anthropic_client):
        """Test validate_typed_parser_output parameter."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            validate_typed_parser_output=True,
        )

        assert model.validate_typed_parser_output is True

    def test_chat_completion_verbose_mode(self, mock_anthropic_client):
        """Test verbose parameter."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            verbose=True,
        )

        assert model.verbose is True


class TestAnthropicBaseURL:
    """Test suite for custom base_url parameter."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        """Setup environment variables for tests."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch("msgflux.models.providers.anthropic.Anthropic") as mock_client, \
             patch("msgflux.models.providers.anthropic.AsyncAnthropic") as mock_async_client:
            yield mock_client, mock_async_client

    def test_chat_completion_custom_base_url(self, mock_anthropic_client):
        """Test AnthropicChatCompletion with custom base_url."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        custom_url = "https://custom-anthropic-api.example.com"
        model = AnthropicChatCompletion(
            model_id="claude-3-5-sonnet-20241022",
            base_url=custom_url,
        )

        assert model.sampling_params["base_url"] == custom_url

    def test_chat_completion_default_base_url(self, mock_anthropic_client):
        """Test AnthropicChatCompletion with default base_url (None)."""
        pytest.importorskip("anthropic")

        from msgflux.models.providers.anthropic import AnthropicChatCompletion

        model = AnthropicChatCompletion(model_id="claude-3-5-sonnet-20241022")

        assert model.sampling_params["base_url"] is None
