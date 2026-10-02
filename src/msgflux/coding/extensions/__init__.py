"""Public coding extension declarations and registry."""

from msgflux.coding.extensions.records import (
    CommandSpec,
    PanelSide,
    PanelSpec,
    RegistrationHandle,
)
from msgflux.coding.extensions.registry import CodingExtensions

__all__ = [
    "CodingExtensions",
    "CommandSpec",
    "PanelSpec",
    "PanelSide",
    "RegistrationHandle",
]
