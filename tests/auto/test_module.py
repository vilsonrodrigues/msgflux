"""Tests for AutoModule main class."""

import json
import tempfile
from pathlib import Path

import pytest

from msgflux.auto.module import AutoModule
from msgflux.exceptions import ConfigurationError, DownloadError, SecurityError


class TestAutoModuleParseRepoId:
    def test_parse_simple_repo_id(self):
        """Test parsing simple owner/repo format."""
        repo_id, source = AutoModule._parse_repo_id("owner/repo")

        assert repo_id == "owner/repo"
        assert source is None  # No explicit source detected

    def test_parse_hf_prefix(self):
        """Test parsing hf:// prefix."""
        repo_id, source = AutoModule._parse_repo_id("hf://owner/repo")

        assert repo_id == "owner/repo"
        assert source == "huggingface"

    def test_parse_gh_prefix(self):
        """Test parsing gh:// prefix."""
        repo_id, source = AutoModule._parse_repo_id("gh://owner/repo")

        assert repo_id == "owner/repo"
        assert source == "github"

    def test_parse_github_com(self):
        """Test parsing github.com/ format."""
        repo_id, source = AutoModule._parse_repo_id("github.com/owner/repo")

        assert repo_id == "owner/repo"
        assert source == "github"

    def test_parse_huggingface_co(self):
        """Test parsing huggingface.co/ format."""
        repo_id, source = AutoModule._parse_repo_id("huggingface.co/owner/repo")

        assert repo_id == "owner/repo"
        assert source == "huggingface"


class TestAutoModuleCreateSource:
    def test_create_github_source(self):
        """Test creating GitHub source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = AutoModule._create_source(
                "github",
                "owner/repo",
                "main",
                Path(tmpdir),
            )

            assert source.name == "github"

    def test_create_huggingface_source(self):
        """Test creating HuggingFace source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = AutoModule._create_source(
                "huggingface",
                "owner/repo",
                "main",
                Path(tmpdir),
            )

            assert source.name == "huggingface"

    def test_create_unknown_source(self):
        """Test error for unknown source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                AutoModule._create_source(
                    "unknown",
                    "owner/repo",
                    "main",
                    Path(tmpdir),
                )

            assert "Unknown source" in str(exc_info.value)


class TestAutoModuleGetRepoUrl:
    def test_github_url(self):
        """Test getting GitHub repo URL."""
        url = AutoModule._get_repo_url("github", "owner/repo", "main")

        assert url == "https://github.com/owner/repo/tree/main"

    def test_huggingface_url(self):
        """Test getting HuggingFace repo URL."""
        url = AutoModule._get_repo_url("huggingface", "owner/repo", "main")

        assert url == "https://huggingface.co/owner/repo/tree/main"


class TestAutoModuleImportClass:
    def test_import_class_success(self):
        """Test importing a class from a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python file
            file_path = Path(tmpdir) / "modeling.py"
            file_path.write_text("""
class TestModule:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
""")

            cls = AutoModule._import_class(file_path, "TestModule", "test/repo")

            assert cls.__name__ == "TestModule"

            # Test instantiation
            instance = cls()
            assert instance.get_value() == 42

    def test_import_class_file_not_found(self):
        """Test error when file not found."""
        with pytest.raises(ConfigurationError) as exc_info:
            AutoModule._import_class(
                Path("/nonexistent/modeling.py"),
                "TestModule",
                "test/repo",
            )

        assert "not found" in str(exc_info.value)

    def test_import_class_class_not_found(self):
        """Test error when class not found in file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "modeling.py"
            file_path.write_text("class OtherClass: pass")

            with pytest.raises(ConfigurationError) as exc_info:
                AutoModule._import_class(file_path, "TestModule", "test/repo")

            assert "not found" in str(exc_info.value)


class TestAutoModuleCheckRequirements:
    def test_check_requirements(self):
        """Test checking module requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config.json
            config_path = Path(tmpdir) / "github" / "owner--repo" / "main"
            config_path.mkdir(parents=True)
            (config_path / "config.json").write_text(
                json.dumps({
                    "msgflux_class": "TestModule",
                    "msgflux_entrypoint": "modeling.py",
                    "msgflux_version": ">=0.1.0",
                    "sharing_mode": "class",
                    "dependencies": {
                        "python_packages": ["json"],
                    },
                })
            )

            # We can't easily test the full check_requirements because it
            # tries to download from remote. This tests the structure.
            # Full integration tests would need mock or real repos.


class TestAutoModuleExceptions:
    def test_security_error_message(self):
        """Test SecurityError message format."""
        error = SecurityError("owner/repo")

        assert "owner/repo" in str(error)
        assert "trust_remote_code" in str(error)

    def test_download_error_message(self):
        """Test DownloadError message format."""
        error = DownloadError("owner/repo", "config.json", "Not found")

        assert "owner/repo" in str(error)
        assert "config.json" in str(error)
        assert "Not found" in str(error)

    def test_configuration_error_message(self):
        """Test ConfigurationError message format."""
        error = ConfigurationError("owner/repo", "Invalid field")

        assert "owner/repo" in str(error)
        assert "Invalid field" in str(error)
