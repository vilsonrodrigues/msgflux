from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PTCContext:
    """Per-call programmatic tool call permissions for code interpreters."""

    allowed_tool_names: frozenset[str]


_CURRENT_PTC_CONTEXT: ContextVar[PTCContext | None] = ContextVar(
    "msgflux_ptc_context",
    default=None,
)


@contextmanager
def ptc_context(allowed_tool_names: Iterable[str]):
    context = PTCContext(allowed_tool_names=frozenset(allowed_tool_names))
    token = _CURRENT_PTC_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_PTC_CONTEXT.reset(token)


def get_ptc_allowed_tool_names() -> frozenset[str]:
    context = _CURRENT_PTC_CONTEXT.get()
    if context is None:
        return frozenset()
    return context.allowed_tool_names

