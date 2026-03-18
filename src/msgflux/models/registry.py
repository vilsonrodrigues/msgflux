from typing import TYPE_CHECKING

from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.models.base import BaseModel

model_registry = AutoloadRegistry("msgflux.models.providers")


def register_model(cls: type["BaseModel"]):
    model_type = getattr(cls, "model_type", None)
    provider = getattr(cls, "provider", None)

    if not model_type or not provider:
        raise ValueError(f"{cls.__name__} must define `model_type` and `provider`.")

    model_registry.setdefault(model_type, {})[provider] = cls
    return cls
