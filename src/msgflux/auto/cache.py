"""Cache management for AutoModule."""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from msgflux.envs import envs

logger = logging.getLogger(__name__)


def get_default_cache_dir() -> Path:
    """Get the default cache directory for AutoModule.

    The cache directory is determined in the following order:
    1. MSGFLUX_AUTO_CACHE_DIR environment variable
    2. ~/.cache/msgflux/auto (Linux/Mac)
    3. LOCALAPPDATA/msgflux/auto (Windows)

    Returns:
        Path to the cache directory.
    """
    # Check environment variable first
    if envs.auto_cache_dir:
        return Path(envs.auto_cache_dir)

    # Platform-specific default
    if os.name == "nt":
        # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        # Unix-like (Linux, macOS)
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base / "msgflux" / "auto"


class CacheManager:
    """Manages the local cache for downloaded modules."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache manager.

        Args:
            cache_dir: Custom cache directory. If None, uses default.
        """
        self.cache_dir = cache_dir or get_default_cache_dir()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_module_path(
        self,
        source_name: str,
        repo_id: str,
        revision: str,
    ) -> Path:
        """Get the cache path for a specific module.

        Args:
            source_name: Name of the source (e.g., "github", "huggingface").
            repo_id: Repository identifier.
            revision: Git revision.

        Returns:
            Path to the module's cache directory.
        """
        # Normalize repo_id for filesystem
        safe_repo_id = repo_id.replace("/", "--")
        return self.cache_dir / source_name / safe_repo_id / revision

    def is_cached(
        self,
        source_name: str,
        repo_id: str,
        revision: str,
        required_files: Optional[list[str]] = None,
    ) -> bool:
        """Check if a module is cached locally.

        Args:
            source_name: Name of the source.
            repo_id: Repository identifier.
            revision: Git revision.
            required_files: Optional list of files that must exist.

        Returns:
            True if all required files are cached.
        """
        module_path = self.get_module_path(source_name, repo_id, revision)

        if not module_path.exists():
            return False

        if required_files:
            for filename in required_files:
                if not (module_path / filename).exists():
                    return False

        return True

    def clear_cache(
        self,
        source_name: Optional[str] = None,
        repo_id: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> None:
        """Clear cached modules.

        Args:
            source_name: Clear only this source. If None, clear all.
            repo_id: Clear only this repo. Requires source_name.
            revision: Clear only this revision. Requires repo_id.
        """
        if source_name is None:
            # Clear everything
            logger.info("Clearing entire AutoModule cache: %s", self.cache_dir)
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self._ensure_cache_dir()
            return

        source_path = self.cache_dir / source_name
        if not source_path.exists():
            return

        if repo_id is None:
            # Clear entire source
            logger.info("Clearing cache for source: %s", source_name)
            shutil.rmtree(source_path)
            return

        safe_repo_id = repo_id.replace("/", "--")
        repo_path = source_path / safe_repo_id
        if not repo_path.exists():
            return

        if revision is None:
            # Clear entire repo
            logger.info("Clearing cache for repo: %s/%s", source_name, repo_id)
            shutil.rmtree(repo_path)
            return

        # Clear specific revision
        revision_path = repo_path / revision
        if revision_path.exists():
            logger.info(
                "Clearing cache for: %s/%s@%s",
                source_name,
                repo_id,
                revision,
            )
            shutil.rmtree(revision_path)

    def get_cache_size(self) -> int:
        """Get the total size of the cache in bytes.

        Returns:
            Total cache size in bytes.
        """
        total = 0
        for path in self.cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def list_cached_modules(self) -> list[dict]:
        """List all cached modules.

        Returns:
            List of dictionaries with module information.
        """
        modules = []

        if not self.cache_dir.exists():
            return modules

        for source_dir in self.cache_dir.iterdir():
            if not source_dir.is_dir():
                continue

            source_name = source_dir.name

            for repo_dir in source_dir.iterdir():
                if not repo_dir.is_dir():
                    continue

                # Convert back from safe name
                repo_id = repo_dir.name.replace("--", "/")

                for revision_dir in repo_dir.iterdir():
                    if not revision_dir.is_dir():
                        continue

                    revision = revision_dir.name

                    # Calculate size
                    size = sum(
                        f.stat().st_size
                        for f in revision_dir.rglob("*")
                        if f.is_file()
                    )

                    modules.append({
                        "source": source_name,
                        "repo_id": repo_id,
                        "revision": revision,
                        "path": str(revision_dir),
                        "size_bytes": size,
                    })

        return modules


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(cache_dir: Optional[Path] = None) -> CacheManager:
    """Get the global cache manager instance.

    Args:
        cache_dir: Custom cache directory. Only used on first call.

    Returns:
        CacheManager instance.
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_dir)
    return _cache_manager
