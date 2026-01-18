"""Configuration loading and validation for AutoModule."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from msgflux.exceptions import ConfigurationError, IncompatibleVersionError
from msgflux.utils.msgspec import read_json
from msgflux.version import __version__

logger = logging.getLogger(__name__)

SharingMode = Literal["class", "instance"]


@dataclass
class ModuleMetadata:
    """Optional metadata for a shared module."""

    name: Optional[str] = None
    description: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    license: Optional[str] = None


@dataclass
class ModuleDependencies:
    """Dependencies for a shared module."""

    python_packages: List[str] = field(default_factory=list)
    msgflux_extras: List[str] = field(default_factory=list)


@dataclass
class ModuleConfig:
    """Configuration for a shared module.

    Represents the parsed and validated config.json file from a repository.
    """

    msgflux_class: str
    msgflux_entrypoint: str
    msgflux_version: str
    sharing_mode: SharingMode
    metadata: ModuleMetadata = field(default_factory=ModuleMetadata)
    dependencies: ModuleDependencies = field(default_factory=ModuleDependencies)

    # Original raw config dict
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], repo_id: str) -> "ModuleConfig":
        """Create ModuleConfig from a dictionary.

        Args:
            data: Dictionary from config.json.
            repo_id: Repository identifier for error messages.

        Returns:
            Validated ModuleConfig instance.

        Raises:
            ConfigurationError: If required fields are missing or invalid.
        """
        # Validate required fields
        required_fields = [
            "msgflux_class",
            "msgflux_entrypoint",
            "msgflux_version",
            "sharing_mode",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ConfigurationError(
                    repo_id,
                    f"Missing required field: {field_name}",
                )

        # Validate sharing_mode
        sharing_mode = data["sharing_mode"]
        if sharing_mode not in ("class", "instance"):
            raise ConfigurationError(
                repo_id,
                f"Invalid sharing_mode: {sharing_mode}. Must be 'class' or 'instance'",
            )

        # Parse metadata
        metadata_dict = data.get("metadata", {})
        metadata = ModuleMetadata(
            name=metadata_dict.get("name"),
            description=metadata_dict.get("description"),
            author=metadata_dict.get("author"),
            tags=metadata_dict.get("tags", []),
            license=metadata_dict.get("license"),
        )

        # Parse dependencies
        deps_dict = data.get("dependencies", {})
        dependencies = ModuleDependencies(
            python_packages=deps_dict.get("python_packages", []),
            msgflux_extras=deps_dict.get("msgflux_extras", []),
        )

        return cls(
            msgflux_class=data["msgflux_class"],
            msgflux_entrypoint=data["msgflux_entrypoint"],
            msgflux_version=data["msgflux_version"],
            sharing_mode=sharing_mode,
            metadata=metadata,
            dependencies=dependencies,
            _raw=data,
        )

    @classmethod
    def from_file(cls, path: Path, repo_id: str) -> "ModuleConfig":
        """Load ModuleConfig from a config.json file.

        Args:
            path: Path to config.json.
            repo_id: Repository identifier for error messages.

        Returns:
            Validated ModuleConfig instance.

        Raises:
            ConfigurationError: If file is missing, invalid JSON, or invalid config.
        """
        if not path.exists():
            raise ConfigurationError(repo_id, "config.json not found")

        try:
            data = read_json(path)
        except Exception as e:
            raise ConfigurationError(repo_id, f"Invalid JSON: {e}") from e

        return cls.from_dict(data, repo_id)

    def validate_version(self, repo_id: str) -> None:
        """Validate that the current msgflux version meets requirements.

        Args:
            repo_id: Repository identifier for error messages.

        Raises:
            IncompatibleVersionError: If version requirement is not met.
        """
        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version
        except ImportError:
            logger.warning(
                "packaging library not installed, skipping version validation"
            )
            return

        try:
            specifier = SpecifierSet(self.msgflux_version)
            current = Version(__version__)

            if current not in specifier:
                raise IncompatibleVersionError(
                    repo_id,
                    self.msgflux_version,
                    __version__,
                )
        except Exception as e:
            if isinstance(e, IncompatibleVersionError):
                raise
            logger.warning("Could not validate version: %s", e)

    def check_dependencies(self) -> Dict[str, List[str]]:
        """Check if all dependencies are available.

        Returns:
            Dictionary with 'missing' and 'available' keys, each containing
            a list of package names.
        """
        result = {"missing": [], "available": []}

        # Check Python packages
        for package in self.dependencies.python_packages:
            # Extract package name (handle version specifiers)
            name = package.split(">=")[0].split("==")[0].split("<")[0].strip()
            try:
                __import__(name.replace("-", "_"))
                result["available"].append(package)
            except ImportError:
                result["missing"].append(package)

        # Check msgflux extras
        # These are typically optional dependencies in msgflux
        # We just note them for now
        for extra in self.dependencies.msgflux_extras:
            # Try to check if the extra is installed
            # This is a simplified check
            try:
                if extra == "openai":
                    __import__("openai")
                elif extra == "httpx":
                    __import__("httpx")
                elif extra == "google":
                    __import__("google.genai")
                elif extra == "xml":
                    __import__("defusedxml")
                elif extra == "fal":
                    __import__("fal_client")
                elif extra == "hub":
                    __import__("huggingface_hub")
                else:
                    # Unknown extra, assume available
                    pass
                result["available"].append(f"msgflux[{extra}]")
            except ImportError:
                result["missing"].append(f"msgflux[{extra}]")

        return result
