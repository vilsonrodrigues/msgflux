from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.models.model import Model

__all__ = ["Model"]


def __getattr__(name: str):
    if name != "Model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.models.model"), name)
    globals()[name] = value
    return value
