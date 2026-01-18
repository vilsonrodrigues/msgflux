from typing import List, Optional, Tuple


class ToolCallTimeOutError(Exception):
    pass


class ModelRouterError(Exception):
    def __init__(
        self,
        exceptions: List[Exception],
        model_info: List[Tuple[str, str, Exception]],
        message: Optional[str] = "Model routing failed",
    ):
        self.exceptions = exceptions
        self.model_info = model_info
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = "\n".join(
            [
                f"  - Model: {m_id} ({prov}): {type(exc).__name__}: {exc}"
                for m_id, prov, exc in self.model_info
            ]
        )
        return f"{self.message}\nCaptured Exceptions:\n{details}"


class UnsafeUserInputError(Exception):
    pass


class UnsafeModelResponseError(Exception):
    pass


class TypedParserNotFoundError(ValueError):
    """Raised when a requested typed parser is not registered."""


class AutoModuleError(Exception):
    """Base error for AutoModule operations."""


class SecurityError(AutoModuleError):
    """trust_remote_code=True required but not provided."""

    def __init__(self, repo_id: str, message: Optional[str] = None):
        self.repo_id = repo_id
        if message is None:
            message = (
                f"Loading '{repo_id}' requires trust_remote_code=True.\n"
                "This module has sharing_mode='instance' which executes remote code.\n"
                "Only enable this if you trust the repository author."
            )
        super().__init__(message)


class IncompatibleVersionError(AutoModuleError):
    """msgflux version is incompatible with module requirements."""

    def __init__(
        self,
        repo_id: str,
        required_version: str,
        current_version: str,
    ):
        self.repo_id = repo_id
        self.required_version = required_version
        self.current_version = current_version
        message = (
            f"Module '{repo_id}' requires msgflux{required_version}, "
            f"but current version is {current_version}"
        )
        super().__init__(message)


class ConfigurationError(AutoModuleError):
    """config.json is invalid or missing."""

    def __init__(self, repo_id: str, reason: str):
        self.repo_id = repo_id
        self.reason = reason
        message = f"Invalid configuration for '{repo_id}': {reason}"
        super().__init__(message)


class DownloadError(AutoModuleError):
    """Failed to download files from remote repository."""

    def __init__(self, repo_id: str, filename: str, reason: str):
        self.repo_id = repo_id
        self.filename = filename
        self.reason = reason
        message = f"Failed to download '{filename}' from '{repo_id}': {reason}"
        super().__init__(message)
