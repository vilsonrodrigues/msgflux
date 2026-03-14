from typing import TYPE_CHECKING

from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.data.parsers.base import BaseParser

parser_registry = AutoloadRegistry("msgflux.data.parsers.providers")


def register_parser(cls: type["BaseParser"]):
    parser_type = getattr(cls, "parser_type", None)
    provider = getattr(cls, "provider", None)

    if not parser_type or not provider:
        raise ValueError(f"{cls.__name__} must define `parser_type` and `provider`.")

    parser_registry.setdefault(parser_type, {})[provider] = cls
    return cls
