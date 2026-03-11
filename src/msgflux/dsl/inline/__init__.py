"""Inline DSL — declarative workflow language for module orchestration."""

from msgflux.dsl.inline.core import AsyncInlineDSL, InlineDSL, ainline, inline
from msgflux.dsl.inline.runtime import AsyncDurableInlineDSL, DurableInlineDSL

__all__ = [
    "AsyncDurableInlineDSL",
    "AsyncInlineDSL",
    "DurableInlineDSL",
    "InlineDSL",
    "ainline",
    "inline",
]
