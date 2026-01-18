"""Tests for AutoModule sources."""

import tempfile
from pathlib import Path

import pytest

from msgflux.auto.sources.base import Source
from msgflux.auto.sources.github import GitHubSource
from msgflux.auto.sources.huggingface import HuggingFaceSource


class TestSourceBase:
    def test_parse_repo_id_valid(self):
        """Test parsing valid repo_id."""
        owner, repo = Source.parse_repo_id("owner/repo")

        assert owner == "owner"
        assert repo == "repo"

    def test_parse_repo_id_invalid(self):
        """Test parsing invalid repo_id."""
        with pytest.raises(ValueError) as exc_info:
            Source.parse_repo_id("invalid")

        assert "Invalid repo_id format" in str(exc_info.value)

    def test_parse_repo_id_too_many_parts(self):
        """Test parsing repo_id with too many parts."""
        with pytest.raises(ValueError):
            Source.parse_repo_id("a/b/c")


class TestGitHubSource:
    def test_init(self):
        """Test GitHubSource initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = GitHubSource("owner/repo", "main", Path(tmpdir))

            assert source.owner == "owner"
            assert source.repo == "repo"
            assert source.revision == "main"

    def test_default_revision(self):
        """Test default revision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = GitHubSource("owner/repo", cache_dir=Path(tmpdir))

            assert source.revision == "main"

    def test_get_raw_url(self):
        """Test getting raw URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = GitHubSource("owner/repo", "main", Path(tmpdir))

            url = source._get_raw_url("config.json")

            assert url == "https://raw.githubusercontent.com/owner/repo/main/config.json"

    def test_get_cache_path(self):
        """Test getting cache path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = GitHubSource("owner/repo", "main", Path(tmpdir))

            path = source.get_cache_path()

            assert path == Path(tmpdir) / "github" / "owner--repo" / "main"

    @pytest.mark.skipif(
        True,  # Skip by default, can be enabled for integration tests
        reason="Requires network access",
    )
    def test_download_file_real(self):
        """Integration test for downloading file from GitHub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Using a real public repo that should exist
            source = GitHubSource("octocat/Hello-World", "master", Path(tmpdir))

            path = source.download_file("README")

            assert path.exists()


class TestHuggingFaceSource:
    def test_init(self):
        """Test HuggingFaceSource initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = HuggingFaceSource("owner/repo", "main", Path(tmpdir))

            assert source.repo_id == "owner/repo"
            assert source.revision == "main"

    def test_default_revision(self):
        """Test default revision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = HuggingFaceSource("owner/repo", cache_dir=Path(tmpdir))

            assert source.revision == "main"

    def test_get_hf_hub_import_error(self):
        """Test error when huggingface_hub not installed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = HuggingFaceSource("owner/repo", "main", Path(tmpdir))

            # Mock the import to fail
            import sys

            original_modules = sys.modules.copy()
            sys.modules["huggingface_hub"] = None

            try:
                with pytest.raises(ImportError) as exc_info:
                    source._get_hf_hub()

                assert "huggingface_hub" in str(exc_info.value)
            finally:
                sys.modules.clear()
                sys.modules.update(original_modules)
