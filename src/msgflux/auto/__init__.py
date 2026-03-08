"""Auto module for automatic parameter and configuration management."""

from msgflux.auto.models import load_model_configs
from msgflux.auto.module import AutoModule
from msgflux.auto.params import AutoParams

__all__ = ["AutoModule", "AutoParams", "load_model_configs"]
