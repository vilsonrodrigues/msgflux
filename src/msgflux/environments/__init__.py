"""Environments module for secure code execution."""

from msgflux.environments.base import BaseEnvironment
from msgflux.environments.code import (
    BaseCodeEnvironment,
    BasePythonEnvironment,
    DenoPyodideSandbox,
    Environments,
    ExecutionResult,
    environment_registry,
    register_environment,
)
from msgflux.environments.exceptions import (
    SandboxConnectionError,
    SandboxError,
    SandboxMemoryError,
    SandboxNotReadyError,
    SandboxSecurityError,
    SandboxTimeoutError,
    VariableSizeLimitError,
)
from msgflux.environments.pool import EnvironmentPool, PooledEnvironmentContext

__all__ = [
    "BaseCodeEnvironment",
    "BaseEnvironment",
    "BasePythonEnvironment",
    "DenoPyodideSandbox",
    "EnvironmentPool",
    "Environments",
    "ExecutionResult",
    "PooledEnvironmentContext",
    "SandboxConnectionError",
    "SandboxError",
    "SandboxMemoryError",
    "SandboxNotReadyError",
    "SandboxSecurityError",
    "SandboxTimeoutError",
    "VariableSizeLimitError",
    "environment_registry",
    "register_environment",
]

if __all__ != sorted(__all__):
    raise RuntimeError("__all__ must be sorted")
