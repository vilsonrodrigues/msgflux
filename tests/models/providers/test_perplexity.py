import pytest
from unittest.mock import patch

from msgflux.models.providers.perplexity import PerplexityChatCompletion


def test_init_raises_without_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError):
            PerplexityChatCompletion(model_id="sonar-pro")


def test_config():
    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        model = PerplexityChatCompletion(model_id="sonar-pro")
        assert model.provider == "perplexity"
        assert model.sampling_params["base_url"] == "https://api.perplexity.ai"
        assert model.client is not None


def test_config_with_different_models():
    with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test_key"}):
        model = PerplexityChatCompletion(model_id="sonar-reasoning")
        assert model.provider == "perplexity"
        assert model.model_id == "sonar-reasoning"


def test_custom_base_url():
    with patch.dict(
        "os.environ",
        {
            "PERPLEXITY_API_KEY": "test_key",
            "PERPLEXITY_BASE_URL": "https://custom.perplexity.ai",
        },
    ):
        model = PerplexityChatCompletion(model_id="sonar")
        assert model.sampling_params["base_url"] == "https://custom.perplexity.ai"
