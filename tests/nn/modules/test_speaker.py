"""Tests for msgflux.nn.modules.speaker module."""

import pytest
from unittest.mock import Mock

from msgflux.nn.modules.speaker import Speaker
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockTTSModel(BaseModel):
    """Mock text-to-speech model for testing."""

    model_type = "text_to_speech"
    provider = "mock"

    def __init__(self):
        self.model_id = "test-tts"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add(b"audio_data")
        return response

    async def acall(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add(b"audio_data")
        return response


class TestSpeaker:
    """Test suite for Speaker module."""

    def test_speaker_initialization(self):
        """Test Speaker basic initialization."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model)

        assert speaker.model is mock_model
        assert speaker.response_format == "opus"

    def test_speaker_initialization_with_config(self):
        """Test Speaker initialization with configuration."""
        mock_model = MockTTSModel()
        config = {"stream": True}
        speaker = Speaker(model=mock_model, config=config)

        assert speaker._buffers["config"]["stream"] is True

    def test_speaker_invalid_model_type(self):
        """Test Speaker raises TypeError for invalid model type."""
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        with pytest.raises(TypeError, match="need be a `text_to_speech` model"):
            Speaker(model=mock_model)

    def test_speaker_with_response_format(self):
        """Test Speaker with custom response_format."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model, response_format="mp3")

        assert speaker.response_format == "mp3"

    def test_speaker_invalid_response_format(self):
        """Test Speaker raises ValueError for invalid response_format."""
        mock_model = MockTTSModel()

        with pytest.raises(ValueError, match="`response_format` can be"):
            Speaker(model=mock_model, response_format="invalid")

    def test_speaker_response_format_invalid_type(self):
        """Test Speaker raises TypeError for non-string response_format."""
        mock_model = MockTTSModel()

        with pytest.raises(TypeError, match="`response_format` need be a str"):
            Speaker(model=mock_model, response_format=123)

    def test_speaker_with_prompt(self):
        """Test Speaker initialization with prompt."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model, prompt="Speak clearly")

        assert speaker.prompt == "Speak clearly"

    def test_speaker_config_invalid_type(self):
        """Test Speaker raises TypeError for invalid config type."""
        mock_model = MockTTSModel()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            Speaker(model=mock_model, config="invalid")

    def test_speaker_forward_with_string(self):
        """Test Speaker forward with string input."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model)
        result = speaker("Hello world")

        assert result == b"audio_data"

    def test_speaker_forward_invalid_message_type(self):
        """Test Speaker raises ValueError for invalid message type."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model)

        with pytest.raises(ValueError, match="Unsupported message type"):
            speaker(123)

    def test_speaker_invalid_response_type(self):
        """Test Speaker raises ValueError for unsupported response type."""
        mock_model = Mock()
        mock_model.model_type = "text_to_speech"
        mock_response = Mock(spec=ModelResponse)
        mock_response.response_type = "text_generation"
        mock_model.return_value = mock_response

        speaker = Speaker(model=mock_model)
        with pytest.raises(ValueError, match="Unsupported model response type"):
            speaker("Hello")

    @pytest.mark.asyncio
    async def test_speaker_aforward_with_string(self):
        """Test Speaker async forward with string input."""
        mock_model = MockTTSModel()
        speaker = Speaker(model=mock_model)
        result = await speaker.aforward("Hello world")

        assert result == b"audio_data"

    def test_speaker_with_guardrails(self):
        """Test Speaker with input guardrails."""
        mock_model = MockTTSModel()
        mock_guardrail = Mock()

        speaker = Speaker(model=mock_model, guardrails={"input": mock_guardrail})
        assert speaker.guardrails["input"] is mock_guardrail

    def test_speaker_supported_formats(self):
        """Test all supported audio formats."""
        formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
        for fmt in formats:
            mock_model = MockTTSModel()
            speaker = Speaker(model=mock_model, response_format=fmt)
            assert speaker.response_format == fmt
