"""Environments module for secure code execution."""

from msgflux.environments.exceptions import (
    SandboxConnectionError,
    SandboxError,
    SandboxMemoryError,
    SandboxNotReadyError,
    SandboxSecurityError,
    SandboxTimeoutError,
    VariableSizeLimitError,
)
from msgflux.environments.sandboxes import (
    BasePythonSandbox,
    BaseSandbox,
    ExecutionResult,
    Sandbox,
)

__all__ = [
    "BasePythonSandbox",
    "BaseSandbox",
    "ExecutionResult",
    "Sandbox",
    "SandboxConnectionError",
    "SandboxError",
    "SandboxMemoryError",
    "SandboxNotReadyError",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "VariableSizeLimitError",
]

if __all__ != sorted(__all__):
    raise RuntimeError("__all__ must be sorted")
