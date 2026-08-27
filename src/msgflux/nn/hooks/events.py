"""Typed payloads for stable lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from msgflux.runtime.context import ExecutionScope

if TYPE_CHECKING:
    from msgflux.tools.definitions import ToolCatalog

__all__ = [
    "AgentContext",
    "BeforeRun",
    "BeforeTool",
    "AfterTool",
    "BeforeResume",
    "OutputContext",
    "ModelContext",
]


@dataclass(frozen=True)
class BeforeRun:
    """Input accepted by a fresh Agent run."""

    message: Any
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeforeResume:
    """Durable state that may be transformed before execution resumes.

    Hooks may replace messages, model preference, or non-identity scope fields.
    The restored thread, namespace, and run identity cannot be changed.
    """

    scope: ExecutionScope
    messages: Any
    model_preference: str | None = None


@dataclass(frozen=True, kw_only=True)
class AgentContext:
    """Runtime context shared by Agent lifecycle payloads."""

    scope: ExecutionScope
    vars: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ModelContext(AgentContext):
    """Model-facing prompt and read-only catalog exposed to prompt extensions.

    ``transform_system_prompt`` consumes only ``prompt``. The catalog provides
    request context to prompt extensions and is not a catalog transformation
    boundary.
    """

    prompt: str
    tool_catalog: ToolCatalog | None = None

    @property
    def tool_names(self) -> frozenset[str]:
        if self.tool_catalog is None:
            return frozenset()
        return frozenset(tool.name for tool in self.tool_catalog.portable_tools())


@dataclass(frozen=True, kw_only=True)
class OutputContext(AgentContext):
    """Settled Agent output before presentation to the caller."""

    output: Any


@dataclass(frozen=True)
class BeforeTool:
    """A resolved and validated tool call awaiting execution."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    block: str | None = None


@dataclass(frozen=True)
class AfterTool:
    """A tool outcome before it becomes an Agent tool result."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    error: BaseException | str | None = None
