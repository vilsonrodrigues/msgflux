"""Composable Agent extensions."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.extensions.base import AgentExtension, AgentExtensionHandle
    from msgflux.nn.extensions.skills import SkillsExtension

__all__ = ["AgentExtension", "AgentExtensionHandle", "SkillsExtension"]

_LAZY_IMPORTS = {
    "AgentExtension": ("msgflux.nn.extensions.base", "AgentExtension"),
    "AgentExtensionHandle": (
        "msgflux.nn.extensions.base",
        "AgentExtensionHandle",
    ),
    "SkillsExtension": ("msgflux.nn.extensions.skills", "SkillsExtension"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
