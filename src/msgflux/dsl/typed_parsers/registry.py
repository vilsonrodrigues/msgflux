from typing import TYPE_CHECKING

from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.dsl.typed_parsers.base import BaseTypedParser

typed_parser_registry = AutoloadRegistry("msgflux.dsl.typed_parsers.providers")


def register_typed_parser(cls: type["BaseTypedParser"]):
    key = cls.typed_parser_type
    typed_parser_registry[key] = cls
    return cls
