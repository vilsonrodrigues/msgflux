"""Exceptions for the environments module."""

from typing import Optional


class SandboxError(Exception):
    """Base exception for sandbox errors."""

    pass


class SandboxTimeoutError(SandboxError):
    """Raised when code execution times out."""

    def __init__(self, timeout: float, message: Optional[str] = None):
        """Args:
        timeout:
            The timeout value in seconds that was exceeded.
        message:
            Optional custom error message.
        """
        self.timeout = timeout
        super().__init__(message or f"Execution timed out after {timeout}s")


class SandboxMemoryError(SandboxError):
    """Raised when memory limit is exceeded."""

    def __init__(self, limit_mb: float, message: Optional[str] = None):
        """Args:
        limit_mb:
            The memory limit in megabytes that was exceeded.
        message:
            Optional custom error message.
        """
        self.limit_mb = limit_mb
        super().__init__(message or f"Memory limit exceeded: {limit_mb}MB")


class SandboxConnectionError(SandboxError):
    """Raised when sandbox connection fails."""

    pass


class SandboxSecurityError(SandboxError):
    """Raised when a security violation is detected."""

    pass


class SandboxNotReadyError(SandboxError):
    """Raised when sandbox is not initialized."""

    pass


class VariableSizeLimitError(SandboxError):
    """Raised when variable exceeds size limit."""

    def __init__(
        self,
        name: str,
        size_mb: float,
        limit_mb: float = 100,
        message: Optional[str] = None,
    ):
        """Args:
        name:
            The name of the variable that exceeded the limit.
        size_mb:
            The size of the variable in megabytes.
        limit_mb:
            The maximum allowed size in megabytes.
        message:
            Optional custom error message.
        """
        self.name = name
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        super().__init__(
            message
            or f"Variable '{name}' ({size_mb:.1f}MB) exceeds limit ({limit_mb}MB)"
        )
