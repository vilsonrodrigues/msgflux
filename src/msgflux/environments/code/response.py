"""Response types for code environment execution."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from msgflux.utils.msgspec import msgspec_dumps


@dataclass
class ExecutionResult:
    """Result of code execution in an environment.

    This dataclass provides a consistent interface for all environment
    execution results, whether from Python sandboxes, code interpreters,
    or other execution environments.

    Attributes:
        success: Whether the execution completed without errors.
        output: Captured stdout from the execution.
        error: Error message if execution failed.
        return_value: Return value of the last expression (if any).
        variables: Variables defined during execution.
        execution_time_ms: Execution time in milliseconds.
        memory_used_bytes: Memory used during execution (if available).
        metadata: Additional execution metadata (extensible).
    """

    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    return_value: Optional[Any] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    memory_used_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_value": self.return_value,
            "variables": self.variables,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_bytes": self.memory_used_bytes,
            "metadata": self.metadata,
        }

    def to_json(self) -> bytes:
        """Convert to JSON bytes.

        Returns:
            JSON-encoded bytes representation.
        """
        return msgspec_dumps(self.to_dict())

    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return self.output or str(self.return_value) or "Success"
        return f"Error: {self.error}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ExecutionResult(success={self.success}, "
            f"output={self.output!r}, error={self.error!r})"
        )
