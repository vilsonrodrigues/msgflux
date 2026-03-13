"""Inline DSL — declarative workflow language for module orchestration."""

from msgflux.dsl.inline.core import AsyncInlineDSL, InlineDSL
from msgflux.dsl.inline.module import Inline
from msgflux.dsl.inline.runtime import AsyncDurableInlineDSL, DurableInlineDSL

__all__ = [
    "AsyncDurableInlineDSL",
    "AsyncInlineDSL",
    "DurableInlineDSL",
    "Inline",
    "InlineDSL",
]
