"""GitHub source for AutoModule."""

import logging
from pathlib import Path
from typing import List, Optional

from msgflux.auto.sources.base import Source
from msgflux.exceptions import DownloadError

logger = logging.getLogger(__name__)


class GitHubSource(Source):
    """Source for downloading files from GitHub repositories.

    Uses httpx for HTTP requests to GitHub's raw content URLs.
    """

    name: str = "github"

    RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{path}"
    API_URL_TEMPLATE = "https://api.github.com/repos/{owner}/{repo}/contents/{path}"

    def __init__(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the GitHub source.

        Args:
            repo_id: Repository identifier (e.g., "owner/repo").
            revision: Git revision (branch, tag, or commit hash).
            cache_dir: Local cache directory.
        """
        super().__init__(repo_id, revision, cache_dir)
        self.owner, self.repo = self.parse_repo_id(repo_id)

    def _get_raw_url(self, filename: str) -> str:
        """Get the raw content URL for a file.

        Args:
            filename: Name of the file.

        Returns:
            URL to download the raw file content.
        """
        return self.RAW_URL_TEMPLATE.format(
            owner=self.owner,
            repo=self.repo,
            revision=self.revision,
            path=filename,
        )

    def _get_api_url(self, filename: str) -> str:
        """Get the GitHub API URL for a file.

        Args:
            filename: Name of the file.

        Returns:
            GitHub API URL for the file.
        """
        return self.API_URL_TEMPLATE.format(
            owner=self.owner,
            repo=self.repo,
            path=filename,
        )

    def download_file(
        self,
        filename: str,
        force_download: bool = False,
    ) -> Path:
        """Download a single file from GitHub.

        Args:
            filename: Name of the file to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the downloaded file.

        Raises:
            DownloadError: If download fails.
        """
        try:
            import httpx
        except ImportError as e:
            msg = "httpx is required for GitHub source. Install with: pip install httpx"
            raise ImportError(msg) from e

        cache_path = self.get_cache_path()
        file_path = cache_path / filename

        # Return cached file if exists and not forcing download
        if file_path.exists() and not force_download:
            logger.debug("Using cached file: %s", file_path)
            return file_path

        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        url = self._get_raw_url(filename)
        logger.debug("Downloading from GitHub: %s", url)

        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                file_path.write_bytes(response.content)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise DownloadError(
                    self.repo_id,
                    filename,
                    f"File not found at {url}",
                ) from e
            raise DownloadError(
                self.repo_id,
                filename,
                f"HTTP {e.response.status_code}: {e.response.text}",
            ) from e
        except httpx.RequestError as e:
            raise DownloadError(
                self.repo_id,
                filename,
                str(e),
            ) from e

        logger.info("Downloaded %s to %s", filename, file_path)
        return file_path

    def download_files(
        self,
        filenames: List[str],
        force_download: bool = False,
    ) -> Path:
        """Download multiple files from GitHub.

        Args:
            filenames: List of files to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the cache directory containing downloaded files.

        Raises:
            DownloadError: If any download fails.
        """
        for filename in filenames:
            self.download_file(filename, force_download)
        return self.get_cache_path()

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the GitHub repository.

        Args:
            filename: Name of the file to check.

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            import httpx
        except ImportError:
            return False

        url = self._get_raw_url(filename)

        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                response = client.head(url)
                return response.status_code == 200
        except httpx.RequestError:
            return False
