"""Instance-local registration for coding panels and commands."""

from collections.abc import Iterable
from typing import Any

from msgflux.coding.extensions.records import (
    CommandHandler,
    CommandSpec,
    PanelFactory,
    PanelSide,
    PanelSpec,
    RegistrationHandle,
)


class CodingExtensions:
    """Instance-local registry for coding UI panels and commands."""

    def __init__(self) -> None:
        self._panels: dict[str, tuple[PanelSpec, object]] = {}
        self._commands: dict[str, tuple[CommandSpec, object]] = {}

    @staticmethod
    def _validate(spec: PanelSpec | CommandSpec) -> None:
        if not isinstance(spec.id, str) or not spec.id.strip():
            raise ValueError("Extension id must be a non-empty string.")
        if isinstance(spec, PanelSpec):
            if spec.side not in ("left", "right"):
                raise ValueError("Panel side must be 'left' or 'right'.")
            if not isinstance(spec.title, str) or not spec.title.strip():
                raise ValueError("Panel title must be a non-empty string.")
            if not callable(spec.factory):
                raise TypeError("Panel factory must be callable.")
        elif not callable(spec.handler):
            raise TypeError("Command handler must be callable.")

    def register_panel(
        self,
        id: str,  # noqa: A002 - `id` is the public extension identifier.
        title: str,
        factory: PanelFactory,
        *,
        side: PanelSide = "left",
    ) -> RegistrationHandle:
        return self.register_many(panels=(PanelSpec(id, title, factory, side),))[0]

    def register_command(
        self,
        id: str,  # noqa: A002 - `id` is the public extension identifier.
        handler: CommandHandler,
        *,
        description: str = "",
    ) -> RegistrationHandle:
        """Register ``handler(argument_text: str)`` for a slash command."""
        return self.register_many(commands=(CommandSpec(id, handler, description),))[0]

    def register_many(
        self,
        *,
        panels: Iterable[PanelSpec] = (),
        commands: Iterable[CommandSpec] = (),
    ) -> tuple[RegistrationHandle, ...]:
        """Register a batch atomically, rejecting invalid or duplicate ids."""
        panel_specs, command_specs = tuple(panels), tuple(commands)
        for spec in (*panel_specs, *command_specs):
            self._validate(spec)
        self._check_ids(panel_specs, self._panels, "panel")
        self._check_ids(command_specs, self._commands, "command")

        tokens: list[tuple[dict[str, tuple[Any, object]], str, object]] = []
        for spec, target in (
            *((spec, self._panels) for spec in panel_specs),
            *((spec, self._commands) for spec in command_specs),
        ):
            token = object()
            target[spec.id] = (spec, token)
            tokens.append((target, spec.id, token))

        return tuple(
            RegistrationHandle(
                lambda target=target, key=key, token=token: self._remove_if_owner(
                    target, key, token
                )
            )
            for target, key, token in tokens
        )

    @staticmethod
    def _check_ids(specs: tuple[Any, ...], target: dict[str, Any], kind: str) -> None:
        seen: set[str] = set()
        for spec in specs:
            if spec.id in seen or spec.id in target:
                raise ValueError(f"Duplicate {kind} id: {spec.id!r}.")
            seen.add(spec.id)

    @staticmethod
    def _remove_if_owner(
        target: dict[str, tuple[Any, object]], key: str, token: object
    ) -> None:
        current = target.get(key)
        if current is not None and current[1] is token:
            del target[key]

    def panels(self, side: PanelSide | None = None) -> tuple[PanelSpec, ...]:
        """Return panel specifications in registration order."""
        specs = (entry[0] for entry in self._panels.values())
        return tuple(spec for spec in specs if side is None or spec.side == side)

    def commands(self) -> tuple[CommandSpec, ...]:
        """Return command specifications in registration order."""
        return tuple(entry[0] for entry in self._commands.values())
