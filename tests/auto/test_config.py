"""Tests for AutoModule config loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest

from msgflux.auto.config import ModuleConfig, ModuleDependencies, ModuleMetadata
from msgflux.exceptions import ConfigurationError, IncompatibleVersionError


class TestModuleConfig:
    def test_from_dict_valid(self):
        """Test creating config from valid dictionary."""
        data = {
            "msgflux_class": "MyAgent",
            "msgflux_entrypoint": "modeling.py",
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "class",
        }

        config = ModuleConfig.from_dict(data, "test/repo")

        assert config.msgflux_class == "MyAgent"
        assert config.msgflux_entrypoint == "modeling.py"
        assert config.msgflux_version == ">=0.5.0"
        assert config.sharing_mode == "class"

    def test_from_dict_with_metadata(self):
        """Test creating config with metadata."""
        data = {
            "msgflux_class": "MyAgent",
            "msgflux_entrypoint": "modeling.py",
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "instance",
            "metadata": {
                "name": "Test Agent",
                "description": "A test agent",
                "author": "test_user",
                "tags": ["agent", "test"],
                "license": "MIT",
            },
        }

        config = ModuleConfig.from_dict(data, "test/repo")

        assert config.metadata.name == "Test Agent"
        assert config.metadata.description == "A test agent"
        assert config.metadata.author == "test_user"
        assert config.metadata.tags == ["agent", "test"]
        assert config.metadata.license == "MIT"

    def test_from_dict_with_dependencies(self):
        """Test creating config with dependencies."""
        data = {
            "msgflux_class": "MyAgent",
            "msgflux_entrypoint": "modeling.py",
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "class",
            "dependencies": {
                "python_packages": ["httpx>=0.28.0", "numpy"],
                "msgflux_extras": ["openai"],
            },
        }

        config = ModuleConfig.from_dict(data, "test/repo")

        assert config.dependencies.python_packages == ["httpx>=0.28.0", "numpy"]
        assert config.dependencies.msgflux_extras == ["openai"]

    def test_from_dict_missing_required_field(self):
        """Test error when required field is missing."""
        data = {
            "msgflux_class": "MyAgent",
            # Missing msgflux_entrypoint
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "class",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            ModuleConfig.from_dict(data, "test/repo")

        assert "msgflux_entrypoint" in str(exc_info.value)

    def test_from_dict_invalid_sharing_mode(self):
        """Test error when sharing_mode is invalid."""
        data = {
            "msgflux_class": "MyAgent",
            "msgflux_entrypoint": "modeling.py",
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "invalid",
        }

        with pytest.raises(ConfigurationError) as exc_info:
            ModuleConfig.from_dict(data, "test/repo")

        assert "sharing_mode" in str(exc_info.value)

    def test_from_file_valid(self):
        """Test loading config from file."""
        data = {
            "msgflux_class": "MyAgent",
            "msgflux_entrypoint": "modeling.py",
            "msgflux_version": ">=0.5.0",
            "sharing_mode": "class",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(data))

            config = ModuleConfig.from_file(config_path, "test/repo")

            assert config.msgflux_class == "MyAgent"

    def test_from_file_not_found(self):
        """Test error when config file not found."""
        with pytest.raises(ConfigurationError) as exc_info:
            ModuleConfig.from_file(Path("/nonexistent/config.json"), "test/repo")

        assert "not found" in str(exc_info.value)

    def test_from_file_invalid_json(self):
        """Test error when config file has invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{ invalid json }")

            with pytest.raises(ConfigurationError) as exc_info:
                ModuleConfig.from_file(config_path, "test/repo")

            assert "Invalid JSON" in str(exc_info.value)

    def test_check_dependencies_available(self):
        """Test checking dependencies that are available."""
        config = ModuleConfig(
            msgflux_class="MyAgent",
            msgflux_entrypoint="modeling.py",
            msgflux_version=">=0.5.0",
            sharing_mode="class",
            dependencies=ModuleDependencies(
                python_packages=["json"],  # json is always available
            ),
        )

        result = config.check_dependencies()

        assert "json" in result["available"]

    def test_check_dependencies_missing(self):
        """Test checking dependencies that are missing."""
        config = ModuleConfig(
            msgflux_class="MyAgent",
            msgflux_entrypoint="modeling.py",
            msgflux_version=">=0.5.0",
            sharing_mode="class",
            dependencies=ModuleDependencies(
                python_packages=["nonexistent_package_xyz123"],
            ),
        )

        result = config.check_dependencies()

        assert "nonexistent_package_xyz123" in result["missing"]


class TestModuleMetadata:
    def test_default_values(self):
        """Test default values for metadata."""
        metadata = ModuleMetadata()

        assert metadata.name is None
        assert metadata.description is None
        assert metadata.author is None
        assert metadata.tags == []
        assert metadata.license is None


class TestModuleDependencies:
    def test_default_values(self):
        """Test default values for dependencies."""
        deps = ModuleDependencies()

        assert deps.python_packages == []
        assert deps.msgflux_extras == []
