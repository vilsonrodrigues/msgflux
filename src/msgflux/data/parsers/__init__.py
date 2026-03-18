from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.data.parsers.parser import Parser

__all__ = ["Parser"]


def __getattr__(name: str):
    if name != "Parser":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.data.parsers.parser"), name)
    globals()[name] = value
    return value
