"""Base class for remote repository sources."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class Source(ABC):
    """Abstract base class for remote repository sources.

    Sources handle downloading files from remote repositories like GitHub
    or Hugging Face Hub.
    """

    name: str = "base"

    def __init__(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the source.

        Args:
            repo_id: Repository identifier (e.g., "owner/repo").
            revision: Git revision (branch, tag, or commit hash).
            cache_dir: Local cache directory.
        """
        self.repo_id = repo_id
        self.revision = revision or self.default_revision
        self.cache_dir = cache_dir

    @property
    def default_revision(self) -> str:
        """Default revision to use if none specified."""
        return "main"

    @abstractmethod
    def download_file(
        self,
        filename: str,
        force_download: bool = False,
    ) -> Path:
        """Download a single file from the repository.

        Args:
            filename: Name of the file to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the downloaded file in the local cache.

        Raises:
            DownloadError: If download fails.
        """
        ...

    @abstractmethod
    def download_files(
        self,
        filenames: List[str],
        force_download: bool = False,
    ) -> Path:
        """Download multiple files from the repository.

        Args:
            filenames: List of files to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the cache directory containing downloaded files.

        Raises:
            DownloadError: If download fails.
        """
        ...

    @abstractmethod
    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the remote repository.

        Args:
            filename: Name of the file to check.

        Returns:
            True if the file exists, False otherwise.
        """
        ...

    def get_cache_path(self) -> Path:
        """Get the cache path for this repository.

        Returns:
            Path to the local cache directory for this repo/revision.
        """
        if self.cache_dir is None:
            msg = "cache_dir must be set before calling get_cache_path"
            raise ValueError(msg)

        # Normalize repo_id for filesystem (replace / with --)
        safe_repo_id = self.repo_id.replace("/", "--")
        return self.cache_dir / self.name / safe_repo_id / self.revision

    @classmethod
    def parse_repo_id(cls, repo_id: str) -> tuple[str, str]:
        """Parse a repo_id into owner and repo name.

        Args:
            repo_id: Repository identifier (e.g., "owner/repo").

        Returns:
            Tuple of (owner, repo_name).

        Raises:
            ValueError: If repo_id format is invalid.
        """
        parts = repo_id.split("/")
        if len(parts) != 2:
            msg = f"Invalid repo_id format: {repo_id}. Expected 'owner/repo'"
            raise ValueError(msg)
        return parts[0], parts[1]
