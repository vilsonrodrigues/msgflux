from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.nn.hooks.guard import Guard
    from msgflux.nn.hooks.hook import Hook, RemovableHandle

__all__ = ["Guard", "Hook", "RemovableHandle"]

_LAZY_IMPORTS = {
    "Guard": ("msgflux.nn.hooks.guard", "Guard"),
    "Hook": ("msgflux.nn.hooks.hook", "Hook"),
    "RemovableHandle": ("msgflux.nn.hooks.hook", "RemovableHandle"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
