"""Toolkit-independent coding extension declarations and registration handles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import msgspec

PanelSide = Literal["left", "right"]
PanelFactory = Callable[[], Any]
# Callback signature: ``handler(argument_text: str) -> Any``.
CommandHandler = Callable[[str], Any]


class PanelSpec(msgspec.Struct, frozen=True):
    id: str
    title: str
    factory: PanelFactory
    side: PanelSide = "left"


class CommandSpec(msgspec.Struct, frozen=True):
    id: str
    handler: CommandHandler
    description: str = ""
    accepts_arguments: bool = True
    preserve_status: bool = False


class ToolSpec(msgspec.Struct, frozen=True):
    name: str
    tool: Any


class RegistrationHandle:
    """A handle that removes one registration, at most once."""

    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove
        self._active = True

    def unregister(self) -> None:
        if self._active:
            self._active = False
            self._remove()

    close = unregister

    @property
    def active(self) -> bool:
        return self._active
