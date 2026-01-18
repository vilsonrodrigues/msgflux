"""Hugging Face Hub source for AutoModule."""

import logging
from pathlib import Path
from typing import List, Optional

from msgflux.auto.sources.base import Source
from msgflux.exceptions import DownloadError

logger = logging.getLogger(__name__)


class HuggingFaceSource(Source):
    """Source for downloading files from Hugging Face Hub.

    Uses the huggingface_hub library for downloads.
    """

    name: str = "huggingface"

    def __init__(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the Hugging Face Hub source.

        Args:
            repo_id: Repository identifier (e.g., "owner/repo").
            revision: Git revision (branch, tag, or commit hash).
            cache_dir: Local cache directory (huggingface_hub manages own cache).
        """
        super().__init__(repo_id, revision, cache_dir)
        self._hf_cache_dir = None

    def _get_hf_hub(self):
        """Import and return huggingface_hub module.

        Raises:
            ImportError: If huggingface_hub is not installed.
        """
        try:
            import huggingface_hub

            return huggingface_hub
        except ImportError as e:
            msg = (
                "huggingface_hub is required for Hugging Face Hub source. "
                "Install with: pip install huggingface_hub"
            )
            raise ImportError(msg) from e

    def download_file(
        self,
        filename: str,
        force_download: bool = False,
    ) -> Path:
        """Download a single file from Hugging Face Hub.

        Args:
            filename: Name of the file to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the downloaded file.

        Raises:
            DownloadError: If download fails.
        """
        hf_hub = self._get_hf_hub()

        logger.debug(
            "Downloading from Hugging Face Hub: %s/%s",
            self.repo_id,
            filename,
        )

        try:
            # huggingface_hub manages its own cache
            file_path = hf_hub.hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                force_download=force_download,
                cache_dir=self._hf_cache_dir,
            )
            return Path(file_path)
        except hf_hub.utils.EntryNotFoundError as e:
            raise DownloadError(
                self.repo_id,
                filename,
                "File not found in repository",
            ) from e
        except hf_hub.utils.RepositoryNotFoundError as e:
            raise DownloadError(
                self.repo_id,
                filename,
                f"Repository not found: {self.repo_id}",
            ) from e
        except hf_hub.utils.RevisionNotFoundError as e:
            raise DownloadError(
                self.repo_id,
                filename,
                f"Revision not found: {self.revision}",
            ) from e
        except Exception as e:
            raise DownloadError(
                self.repo_id,
                filename,
                str(e),
            ) from e

    def download_files(
        self,
        filenames: List[str],
        force_download: bool = False,
    ) -> Path:
        """Download multiple files from Hugging Face Hub.

        Args:
            filenames: List of files to download.
            force_download: Re-download even if cached.

        Returns:
            Path to the cache directory containing downloaded files.

        Raises:
            DownloadError: If any download fails.
        """
        hf_hub = self._get_hf_hub()

        logger.debug(
            "Downloading files from Hugging Face Hub: %s",
            self.repo_id,
        )

        try:
            # Download all specified files
            snapshot_path = hf_hub.snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                allow_patterns=filenames,
                force_download=force_download,
                cache_dir=self._hf_cache_dir,
            )
            return Path(snapshot_path)
        except Exception as e:
            raise DownloadError(
                self.repo_id,
                ", ".join(filenames),
                str(e),
            ) from e

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the Hugging Face Hub repository.

        Args:
            filename: Name of the file to check.

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            hf_hub = self._get_hf_hub()
        except ImportError:
            return False

        try:
            hf_hub.hf_hub_url(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
            )
            # If we can get the URL without error, file exists
            # Actually do a HEAD request to verify
            api = hf_hub.HfApi()
            info = api.repo_info(repo_id=self.repo_id, revision=self.revision)
            siblings = {s.rfilename for s in info.siblings}
            return filename in siblings
        except Exception:
            return False

    def get_cache_path(self) -> Path:
        """Get the cache path for this repository.

        For Hugging Face Hub, the cache is managed by the library itself.
        This returns a path within the HF cache directory.

        Returns:
            Path to the local cache directory.
        """
        hf_hub = self._get_hf_hub()

        # Get the HF cache directory
        cache_dir = hf_hub.constants.HF_HUB_CACHE
        return Path(cache_dir)
