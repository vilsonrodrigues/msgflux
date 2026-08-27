from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.hooks.events import (
        AfterTool,
        AgentContext,
        BeforeResume,
        BeforeRun,
        BeforeTool,
        ModelContext,
        OutputContext,
    )
    from msgflux.nn.hooks.guard import Guard
    from msgflux.nn.hooks.hook import Hook, RemovableHandle

__all__ = [
    "AgentContext",
    "AfterTool",
    "BeforeResume",
    "BeforeRun",
    "BeforeTool",
    "Guard",
    "Hook",
    "OutputContext",
    "RemovableHandle",
    "ModelContext",
]

_LAZY_IMPORTS = {
    "AgentContext": ("msgflux.nn.hooks.events", "AgentContext"),
    "AfterTool": ("msgflux.nn.hooks.events", "AfterTool"),
    "BeforeResume": ("msgflux.nn.hooks.events", "BeforeResume"),
    "BeforeRun": ("msgflux.nn.hooks.events", "BeforeRun"),
    "BeforeTool": ("msgflux.nn.hooks.events", "BeforeTool"),
    "Guard": ("msgflux.nn.hooks.guard", "Guard"),
    "Hook": ("msgflux.nn.hooks.hook", "Hook"),
    "OutputContext": ("msgflux.nn.hooks.events", "OutputContext"),
    "RemovableHandle": ("msgflux.nn.hooks.hook", "RemovableHandle"),
    "ModelContext": ("msgflux.nn.hooks.events", "ModelContext"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
