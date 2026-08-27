"""Typed payloads for stable lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from msgflux.runtime.context import ExecutionScope

__all__ = [
    "BeforeRun",
    "BeforeTool",
    "AfterTool",
    "BeforeResume",
    "SystemPromptContext",
]


@dataclass(frozen=True)
class BeforeRun:
    """Input accepted by a fresh Agent run."""

    message: Any
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeforeResume:
    """Durable state restored before execution resumes."""

    scope: ExecutionScope
    messages: Any
    model_preference: str | None = None


@dataclass(frozen=True)
class SystemPromptContext:
    """Rendered prompt and runtime inputs exposed to prompt extensions."""

    prompt: str
    scope: ExecutionScope
    vars: Mapping[str, Any] = field(default_factory=dict)
    tool_names: frozenset[str] = field(default_factory=frozenset)


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
