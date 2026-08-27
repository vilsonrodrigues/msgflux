"""Composable Agent extensions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.extensions.base import AgentExtension, AgentExtensionHandle
    from msgflux.nn.extensions.prompt import (
        CurrentDateExtension,
        ToolUsageGuidanceExtension,
    )
    from msgflux.nn.extensions.skills import SkillsExtension

__all__ = [
    "AgentExtension",
    "AgentExtensionHandle",
    "CurrentDateExtension",
    "SkillsExtension",
    "ToolUsageGuidanceExtension",
]

_LAZY_IMPORTS = {
    "AgentExtension": ("msgflux.nn.extensions.base", "AgentExtension"),
    "AgentExtensionHandle": (
        "msgflux.nn.extensions.base",
        "AgentExtensionHandle",
    ),
    "CurrentDateExtension": (
        "msgflux.nn.extensions.prompt",
        "CurrentDateExtension",
    ),
    "SkillsExtension": ("msgflux.nn.extensions.skills", "SkillsExtension"),
    "ToolUsageGuidanceExtension": (
        "msgflux.nn.extensions.prompt",
        "ToolUsageGuidanceExtension",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
