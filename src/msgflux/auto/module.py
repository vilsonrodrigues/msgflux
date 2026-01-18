"""AutoModule - Load nn.Module from remote repositories."""

import importlib.util
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union

from msgflux.auto.cache import CacheManager, get_default_cache_dir
from msgflux.auto.config import ModuleConfig
from msgflux.auto.sources.base import Source
from msgflux.auto.sources.github import GitHubSource
from msgflux.auto.sources.huggingface import HuggingFaceSource
from msgflux.exceptions import (
    ConfigurationError,
    DownloadError,
    SecurityError,
)

logger = logging.getLogger(__name__)

SourceType = Literal["github", "huggingface"]


class AutoModule:
    """Load nn.Module from remote repositories.

    AutoModule enables sharing and loading of msgflux modules from GitHub
    and Hugging Face Hub repositories.

    Example:
        >>> import msgflux as mf
        >>>
        >>> # Load class (sharing_mode: class)
        >>> AgentClass = mf.AutoModule("owner/repo-name")
        >>> agent = AgentClass(model=my_model)
        >>>
        >>> # Load instance (sharing_mode: instance, requires trust_remote_code)
        >>> agent = mf.AutoModule("owner/repo", trust_remote_code=True)
        >>>
        >>> # Pin revision for reproducibility
        >>> AgentClass = mf.AutoModule("owner/repo", revision="abc123")
        >>>
        >>> # Explicit source (hf:// or gh://)
        >>> AgentClass = mf.AutoModule("hf://owner/repo")
    """

    # Source detection patterns
    _SOURCE_PATTERNS = [
        (r"^hf://(.+)$", "huggingface"),
        (r"^gh://(.+)$", "github"),
        (r"^github\.com/(.+)$", "github"),
        (r"^huggingface\.co/(.+)$", "huggingface"),
    ]

    def __new__(
        cls,
        repo_id: str,
        *,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        source: Optional[SourceType] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
    ) -> Union[Type[Any], Any]:
        """Load a module from a remote repository.

        Args:
            repo_id: Repository identifier. Formats supported:
                - "owner/repo" (defaults to GitHub)
                - "hf://owner/repo" (Hugging Face Hub)
                - "gh://owner/repo" (GitHub)
                - "github.com/owner/repo"
                - "huggingface.co/owner/repo"
            trust_remote_code: Allow code execution for instance loading.
                Required when sharing_mode is "instance".
            revision: Git revision (branch, tag, or commit hash).
                Defaults to "main".
            source: Explicit source ("github" or "huggingface").
                Auto-detected from repo_id if not specified.
            local_files_only: Use only cached files, don't download.
            cache_dir: Custom cache directory path.
            force_download: Re-download even if cached.

        Returns:
            Type[Module] if sharing_mode="class"
            Module instance if sharing_mode="instance"

        Raises:
            SecurityError: If trust_remote_code=False but instance loading required.
            ConfigurationError: If config.json is invalid.
            DownloadError: If download fails.
            IncompatibleVersionError: If msgflux version is incompatible.
        """
        # Parse repo_id and detect source
        clean_repo_id, detected_source = cls._parse_repo_id(repo_id)

        # Use explicit source if provided, otherwise use detected
        source = source or detected_source or "github"

        # Set up cache
        cache_path = Path(cache_dir) if cache_dir else get_default_cache_dir()
        cache_manager = CacheManager(cache_path)

        # Create source handler
        source_handler = cls._create_source(
            source,
            clean_repo_id,
            revision,
            cache_path,
        )

        # Download config.json first
        if local_files_only:
            config_path = cache_manager.get_module_path(
                source,
                clean_repo_id,
                revision or "main",
            ) / "config.json"
            if not config_path.exists():
                raise DownloadError(
                    clean_repo_id,
                    "config.json",
                    "File not in cache and local_files_only=True",
                )
        else:
            config_path = source_handler.download_file(
                "config.json",
                force_download,
            )

        # Load and validate config
        config = ModuleConfig.from_file(config_path, clean_repo_id)
        config.validate_version(clean_repo_id)

        # Check dependencies and warn if missing
        deps_status = config.check_dependencies()
        if deps_status["missing"]:
            warnings.warn(
                f"Module '{clean_repo_id}' has missing dependencies: "
                f"{', '.join(deps_status['missing'])}. "
                "These will not be installed automatically.",
                UserWarning,
                stacklevel=2,
            )

        # Check trust_remote_code for instance mode
        if config.sharing_mode == "instance" and not trust_remote_code:
            raise SecurityError(clean_repo_id)

        # Log download info
        actual_revision = revision or "main"
        logger.info(
            "Loading module from %s/%s@%s",
            source,
            clean_repo_id,
            actual_revision,
        )

        # Extract filename from entrypoint (handle "file.py:object" format)
        entrypoint_file = config.msgflux_entrypoint.split(":")[0]

        # Download entrypoint file
        if not local_files_only:
            source_handler.download_file(entrypoint_file, force_download)

        # Get cache path for module files
        module_cache_path = cache_manager.get_module_path(
            source,
            clean_repo_id,
            actual_revision,
        )

        # Return class or instance based on sharing_mode
        if config.sharing_mode == "class":
            # Import the module class
            module_cls = cls._import_class(
                module_cache_path / entrypoint_file,
                config.msgflux_class,
                clean_repo_id,
            )
            return module_cls
        else:
            # For instance mode, import the object directly from the module
            return cls._import_object(
                module_cache_path / entrypoint_file,
                config.msgflux_entrypoint,
                clean_repo_id,
            )

    @classmethod
    def load_class(
        cls,
        repo_id: str,
        **kwargs: Any,
    ) -> Type[Any]:
        """Always load the class, ignoring sharing_mode.

        Args:
            repo_id: Repository identifier.
            **kwargs: Additional arguments passed to AutoModule.

        Returns:
            The module class.
        """
        # Parse and download
        clean_repo_id, detected_source = cls._parse_repo_id(repo_id)
        source = kwargs.pop("source", None) or detected_source or "github"
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        force_download = kwargs.get("force_download", False)
        local_files_only = kwargs.get("local_files_only", False)

        cache_path = Path(cache_dir) if cache_dir else get_default_cache_dir()
        cache_manager = CacheManager(cache_path)

        source_handler = cls._create_source(
            source,
            clean_repo_id,
            revision,
            cache_path,
        )

        if local_files_only:
            config_path = cache_manager.get_module_path(
                source,
                clean_repo_id,
                revision or "main",
            ) / "config.json"
        else:
            config_path = source_handler.download_file("config.json", force_download)

        config = ModuleConfig.from_file(config_path, clean_repo_id)
        config.validate_version(clean_repo_id)

        if not local_files_only:
            source_handler.download_file(config.msgflux_entrypoint, force_download)

        module_cache_path = cache_manager.get_module_path(
            source,
            clean_repo_id,
            revision or "main",
        )

        return cls._import_class(
            module_cache_path / config.msgflux_entrypoint,
            config.msgflux_class,
            clean_repo_id,
        )

    @classmethod
    def load_instance(
        cls,
        repo_id: str,
        **kwargs: Any,
    ) -> Any:
        """Always load an instance, ignoring sharing_mode.

        Args:
            repo_id: Repository identifier.
            **kwargs: Additional arguments passed to AutoModule.

        Returns:
            Module instance.

        Note:
            This implicitly sets trust_remote_code=True.
        """
        # Parse and download
        clean_repo_id, detected_source = cls._parse_repo_id(repo_id)
        source = kwargs.pop("source", None) or detected_source or "github"
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        force_download = kwargs.get("force_download", False)
        local_files_only = kwargs.get("local_files_only", False)

        cache_path = Path(cache_dir) if cache_dir else get_default_cache_dir()
        cache_manager = CacheManager(cache_path)

        source_handler = cls._create_source(
            source,
            clean_repo_id,
            revision,
            cache_path,
        )

        if local_files_only:
            config_path = cache_manager.get_module_path(
                source,
                clean_repo_id,
                revision or "main",
            ) / "config.json"
        else:
            config_path = source_handler.download_file("config.json", force_download)

        config = ModuleConfig.from_file(config_path, clean_repo_id)
        config.validate_version(clean_repo_id)

        # Extract filename from entrypoint (handle "file.py:object" format)
        entrypoint_file = config.msgflux_entrypoint.split(":")[0]

        if not local_files_only:
            source_handler.download_file(entrypoint_file, force_download)

        module_cache_path = cache_manager.get_module_path(
            source,
            clean_repo_id,
            revision or "main",
        )

        logger.info(
            "Loading instance from %s/%s@%s",
            source,
            clean_repo_id,
            revision or "main",
        )

        return cls._import_object(
            module_cache_path / entrypoint_file,
            config.msgflux_entrypoint,
            clean_repo_id,
        )

    @classmethod
    def check_requirements(
        cls,
        repo_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Check module requirements without executing code.

        Args:
            repo_id: Repository identifier.
            **kwargs: Additional arguments for source configuration.

        Returns:
            Dictionary with:
                - config: ModuleConfig object
                - version_ok: bool
                - dependencies: dict with missing/available lists
        """
        clean_repo_id, detected_source = cls._parse_repo_id(repo_id)
        source = kwargs.pop("source", None) or detected_source or "github"
        revision = kwargs.get("revision")
        cache_dir = kwargs.get("cache_dir")
        force_download = kwargs.get("force_download", False)

        cache_path = Path(cache_dir) if cache_dir else get_default_cache_dir()

        source_handler = cls._create_source(
            source,
            clean_repo_id,
            revision,
            cache_path,
        )

        config_path = source_handler.download_file("config.json", force_download)
        config = ModuleConfig.from_file(config_path, clean_repo_id)

        # Check version
        version_ok = True
        try:
            config.validate_version(clean_repo_id)
        except Exception:
            version_ok = False

        return {
            "config": config,
            "version_ok": version_ok,
            "dependencies": config.check_dependencies(),
        }

    @classmethod
    def _parse_repo_id(cls, repo_id: str) -> tuple[str, Optional[str]]:
        """Parse repo_id and detect source.

        Args:
            repo_id: Raw repository identifier.

        Returns:
            Tuple of (clean_repo_id, detected_source).
        """
        for pattern, source_name in cls._SOURCE_PATTERNS:
            match = re.match(pattern, repo_id)
            if match:
                return match.group(1), source_name

        # Default: assume it's a simple "owner/repo" format
        return repo_id, None

    @classmethod
    def _create_source(
        cls,
        source: str,
        repo_id: str,
        revision: Optional[str],
        cache_dir: Path,
    ) -> Source:
        """Create a source handler.

        Args:
            source: Source type name.
            repo_id: Repository identifier.
            revision: Git revision.
            cache_dir: Cache directory.

        Returns:
            Source instance.

        Raises:
            ValueError: If source is not recognized.
        """
        if source == "github":
            return GitHubSource(repo_id, revision, cache_dir)
        elif source == "huggingface":
            return HuggingFaceSource(repo_id, revision, cache_dir)
        else:
            msg = f"Unknown source: {source}. Must be 'github' or 'huggingface'"
            raise ValueError(msg)

    @classmethod
    def _get_repo_url(
        cls,
        source: str,
        repo_id: str,
        revision: str,
    ) -> str:
        """Get the web URL for a repository.

        Args:
            source: Source type name.
            repo_id: Repository identifier.
            revision: Git revision.

        Returns:
            URL to view the repository.
        """
        if source == "github":
            return f"https://github.com/{repo_id}/tree/{revision}"
        elif source == "huggingface":
            return f"https://huggingface.co/{repo_id}/tree/{revision}"
        else:
            return f"{source}://{repo_id}@{revision}"

    @classmethod
    def _import_class(
        cls,
        file_path: Path,
        class_name: str,
        repo_id: str,
    ) -> Type[Any]:
        """Import a class from a Python file.

        Args:
            file_path: Path to the Python file.
            class_name: Name of the class to import.
            repo_id: Repository identifier for error messages.

        Returns:
            The imported class.

        Raises:
            ConfigurationError: If import fails.
        """
        if not file_path.exists():
            raise ConfigurationError(
                repo_id,
                f"Entrypoint file not found: {file_path}",
            )

        try:
            # Create a unique module name
            module_name = f"msgflux_auto_{repo_id.replace('/', '_')}_{file_path.stem}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ConfigurationError(
                    repo_id,
                    f"Could not load module spec from {file_path}",
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get the class
            if not hasattr(module, class_name):
                raise ConfigurationError(
                    repo_id,
                    f"Class '{class_name}' not found in {file_path}",
                )

            return getattr(module, class_name)

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                repo_id,
                f"Failed to import class: {e}",
            ) from e

    @classmethod
    def _import_object(
        cls,
        file_path: Path,
        entrypoint: str,
        repo_id: str,
    ) -> Any:
        """Import an object from a Python file.

        The entrypoint format is "file.py:object_name" where object_name
        is the variable name of the pre-instantiated object.

        Args:
            file_path: Path to the Python file.
            entrypoint: Entrypoint in format "file.py:object_name".
            repo_id: Repository identifier for error messages.

        Returns:
            The imported object.

        Raises:
            ConfigurationError: If import fails or object not found.
        """
        # Parse entrypoint to get object name
        parts = entrypoint.split(":")
        if len(parts) != 2:
            raise ConfigurationError(
                repo_id,
                f"Invalid entrypoint format for instance mode: '{entrypoint}'. "
                "Expected 'file.py:object_name'",
            )

        object_name = parts[1]

        if not file_path.exists():
            raise ConfigurationError(
                repo_id,
                f"Entrypoint file not found: {file_path}",
            )

        try:
            # Create a unique module name
            module_name = f"msgflux_auto_{repo_id.replace('/', '_')}_{file_path.stem}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ConfigurationError(
                    repo_id,
                    f"Could not load module spec from {file_path}",
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Get the object
            if not hasattr(module, object_name):
                raise ConfigurationError(
                    repo_id,
                    f"Object '{object_name}' not found in {file_path}",
                )

            return getattr(module, object_name)

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                repo_id,
                f"Failed to import object: {e}",
            ) from e


__all__ = ["AutoModule"]
