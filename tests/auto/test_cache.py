"""Tests for AutoModule cache management."""

import os
import tempfile
from pathlib import Path

import pytest

from msgflux.auto.cache import CacheManager, get_default_cache_dir
from msgflux.envs import envs


class TestGetDefaultCacheDir:
    def test_uses_envs_auto_cache_dir(self, monkeypatch):
        """Test that envs.auto_cache_dir is used when set."""
        custom_path = "/custom/cache/path"
        monkeypatch.setattr(envs, "auto_cache_dir", custom_path)

        result = get_default_cache_dir()

        assert result == Path(custom_path)

    def test_default_linux(self, monkeypatch):
        """Test default cache dir on Linux."""
        monkeypatch.setattr(envs, "auto_cache_dir", None)
        monkeypatch.setattr(os, "name", "posix")

        result = get_default_cache_dir()

        assert "msgflux" in str(result)
        assert "auto" in str(result)


class TestCacheManager:
    def test_init_creates_directory(self):
        """Test that CacheManager creates cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            manager = CacheManager(cache_dir)

            assert cache_dir.exists()

    def test_get_module_path(self):
        """Test getting module path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            manager = CacheManager(cache_dir)

            path = manager.get_module_path("github", "owner/repo", "main")

            assert path == cache_dir / "github" / "owner--repo" / "main"

    def test_is_cached_false_when_not_exists(self):
        """Test is_cached returns False when not cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            result = manager.is_cached("github", "owner/repo", "main")

            assert result is False

    def test_is_cached_true_when_exists(self):
        """Test is_cached returns True when cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create the cached directory
            module_path = manager.get_module_path("github", "owner/repo", "main")
            module_path.mkdir(parents=True)

            result = manager.is_cached("github", "owner/repo", "main")

            assert result is True

    def test_is_cached_checks_required_files(self):
        """Test is_cached checks for required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create the cached directory but not the files
            module_path = manager.get_module_path("github", "owner/repo", "main")
            module_path.mkdir(parents=True)

            result = manager.is_cached(
                "github",
                "owner/repo",
                "main",
                required_files=["config.json"],
            )

            assert result is False

            # Create the required file
            (module_path / "config.json").write_text("{}")

            result = manager.is_cached(
                "github",
                "owner/repo",
                "main",
                required_files=["config.json"],
            )

            assert result is True

    def test_clear_cache_all(self):
        """Test clearing entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create some cached content
            module_path = manager.get_module_path("github", "owner/repo", "main")
            module_path.mkdir(parents=True)
            (module_path / "config.json").write_text("{}")

            manager.clear_cache()

            assert not module_path.exists()

    def test_clear_cache_by_source(self):
        """Test clearing cache by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create cached content for two sources
            github_path = manager.get_module_path("github", "owner/repo", "main")
            github_path.mkdir(parents=True)
            (github_path / "config.json").write_text("{}")

            hf_path = manager.get_module_path("huggingface", "owner/repo", "main")
            hf_path.mkdir(parents=True)
            (hf_path / "config.json").write_text("{}")

            manager.clear_cache(source_name="github")

            assert not github_path.exists()
            assert hf_path.exists()

    def test_clear_cache_by_repo(self):
        """Test clearing cache by repo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create cached content for two repos
            repo1_path = manager.get_module_path("github", "owner/repo1", "main")
            repo1_path.mkdir(parents=True)

            repo2_path = manager.get_module_path("github", "owner/repo2", "main")
            repo2_path.mkdir(parents=True)

            manager.clear_cache(source_name="github", repo_id="owner/repo1")

            assert not repo1_path.exists()
            assert repo2_path.exists()

    def test_get_cache_size(self):
        """Test getting cache size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create some cached content
            module_path = manager.get_module_path("github", "owner/repo", "main")
            module_path.mkdir(parents=True)
            (module_path / "config.json").write_text("x" * 100)

            size = manager.get_cache_size()

            assert size >= 100

    def test_list_cached_modules(self):
        """Test listing cached modules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            # Create some cached content
            module_path = manager.get_module_path("github", "owner/repo", "main")
            module_path.mkdir(parents=True)
            (module_path / "config.json").write_text("{}")

            modules = manager.list_cached_modules()

            assert len(modules) == 1
            assert modules[0]["source"] == "github"
            assert modules[0]["repo_id"] == "owner/repo"
            assert modules[0]["revision"] == "main"

    def test_list_cached_modules_empty(self):
        """Test listing cached modules when empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CacheManager(Path(tmpdir))

            modules = manager.list_cached_modules()

            assert modules == []
