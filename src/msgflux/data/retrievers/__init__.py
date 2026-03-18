from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.data.retrievers.retriever import Retriever

__all__ = ["Retriever"]


def __getattr__(name: str):
    if name != "Retriever":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.data.retrievers.retriever"), name)
    globals()[name] = value
    return value
