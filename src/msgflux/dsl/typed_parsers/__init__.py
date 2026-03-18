from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.dsl.typed_parsers.registry import (
        register_typed_parser,
        typed_parser_registry,
    )

__all__ = ["register_typed_parser", "typed_parser_registry"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.dsl.typed_parsers.registry"), name)
    globals()[name] = value
    return value
