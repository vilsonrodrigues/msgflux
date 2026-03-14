from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.data.dbs.db import DB

__all__ = ["DB"]


def __getattr__(name: str):
    if name != "DB":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.data.dbs.db"), name)
    globals()[name] = value
    return value
