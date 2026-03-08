"""Tests for load_model_configs utility."""

import json
import tempfile
from pathlib import Path

import pytest

from msgflux.auto.models import load_model_configs


class TestLoadModelConfigs:
    def test_returns_models_from_config(self):
        """Loads 'models' section from config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {
                    "lm": {"model_id": "openai/gpt-5-mini", "temperature": 0.7},
                    "embedder": {"model_id": "openai/text-embedding-3-small"},
                },
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file)

            assert result["lm"]["model_id"] == "openai/gpt-5-mini"
            assert result["lm"]["temperature"] == 0.7
            assert result["embedder"]["model_id"] == "openai/text-embedding-3-small"

    def test_returns_empty_dict_when_no_models_key(self):
        """Returns empty dict when config.json has no 'models' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file)

            assert result == {}

    def test_returns_empty_dict_when_no_config_file(self):
        """Returns empty dict when config.json does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file)

            assert result == {}

    def test_overrides_replace_config_values(self):
        """Overrides replace matching keys from config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {
                    "lm": {"model_id": "openai/gpt-5-mini", "temperature": 0.7},
                    "embedder": {"model_id": "openai/text-embedding-3-small"},
                },
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(
                fake_file,
                overrides={"lm": {"model_id": "groq/llama-3.1-8b", "temperature": 0.3}},
            )

            assert result["lm"]["model_id"] == "groq/llama-3.1-8b"
            assert result["lm"]["temperature"] == 0.3
            assert result["embedder"]["model_id"] == "openai/text-embedding-3-small"

    def test_overrides_add_new_keys(self):
        """Overrides can add keys not present in config.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {"lm": {"model_id": "openai/gpt-5-mini"}},
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(
                fake_file,
                overrides={"speaker": {"model_id": "openai/tts-1"}},
            )

            assert result["lm"]["model_id"] == "openai/gpt-5-mini"
            assert result["speaker"]["model_id"] == "openai/tts-1"

    def test_none_overrides_uses_config_defaults(self):
        """Passing overrides=None returns config.json values unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {"lm": {"model_id": "openai/gpt-5-mini"}},
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file, overrides=None)

            assert result["lm"]["model_id"] == "openai/gpt-5-mini"

    def test_rich_model_config_dict(self):
        """Supports rich dict values, not just plain model IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {
                    "lm": {
                        "model_id": "openai/gpt-4o-mini",
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                },
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file)

            assert result["lm"]["model_id"] == "openai/gpt-4o-mini"
            assert result["lm"]["temperature"] == 0.7

    def test_invalid_json_returns_empty_dict(self):
        """Returns empty dict when config.json has invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "config.json").write_text("{ invalid }")
            fake_file = str(Path(tmpdir) / "modeling.py")

            result = load_model_configs(fake_file)

            assert result == {}

    def test_does_not_mutate_config_on_override(self):
        """Overrides do not mutate the base config between calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "msgflux_class": "MyAgent",
                "msgflux_entrypoint": "modeling.py",
                "msgflux_version": ">=0.1.0",
                "sharing_mode": "class",
                "models": {"lm": {"model_id": "openai/gpt-5-mini"}},
            }
            (Path(tmpdir) / "config.json").write_text(json.dumps(config))
            fake_file = str(Path(tmpdir) / "modeling.py")

            load_model_configs(fake_file, overrides={"lm": {"model_id": "groq/llama-3.1-8b"}})
            result = load_model_configs(fake_file)

            assert result["lm"]["model_id"] == "openai/gpt-5-mini"
