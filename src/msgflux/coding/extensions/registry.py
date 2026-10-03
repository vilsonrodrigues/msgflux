"""Instance-local registration for coding panels, commands, and tool factories."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from typing import Any

from msgflux.coding.extensions.records import (
    CommandHandler,
    CommandSpec,
    PanelFactory,
    PanelSide,
    PanelSpec,
    RegistrationHandle,
    ToolFactory,
    ToolFactorySpec,
)

_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class CodingExtensions:
    """Instance-local registry of coding extension declarations."""

    def __init__(self) -> None:
        self._panels: dict[str, tuple[PanelSpec, object]] = {}
        self._commands: dict[str, tuple[CommandSpec, object]] = {}
        self._tools: dict[str, tuple[ToolFactorySpec, object]] = {}

    @staticmethod
    def _validate(spec: PanelSpec | CommandSpec | ToolFactorySpec) -> None:
        if isinstance(spec, ToolFactorySpec):
            CodingExtensions._validate_tool(spec)
            return

        CodingExtensions._validate_ui(spec)

    @staticmethod
    def _validate_tool(spec: ToolFactorySpec) -> None:
        if not isinstance(spec.name, str) or not _TOOL_NAME.fullmatch(spec.name):
            raise ValueError(f"Invalid tool name: {spec.name!r}.")
        if not callable(spec.factory):
            raise TypeError("Tool factory must be callable.")
        if inspect.iscoroutinefunction(spec.factory):
            raise TypeError("Tool factory must be synchronous.")
        if not isinstance(spec.factory, type) and not inspect.isfunction(spec.factory):
            raise TypeError(
                "Tool factory must be a class or function; "
                "callable instances are ambiguous."
            )
        try:
            inspect.signature(spec.factory).bind()
        except (TypeError, ValueError) as exc:
            raise TypeError("Tool factory must accept zero arguments.") from exc
        if not isinstance(spec.description, str):
            raise TypeError("Tool description must be a string.")

    @staticmethod
    def _validate_ui(spec: PanelSpec | CommandSpec) -> None:
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

    def load(self, register: Callable[[CodingExtensions], Any]) -> Any:
        """Run one synchronous extension callback transactionally.

        Declarations made before the callback are preserved. If the callback
        raises or returns an awaitable, registrations made during that callback
        are rolled back in place so existing handles remain attached to this
        registry.
        """
        snapshots = (
            self._panels.copy(),
            self._commands.copy(),
            self._tools.copy(),
        )
        try:
            result = register(self)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Coding extension registration must be synchronous.")
        except BaseException:
            for target, snapshot in zip(
                (self._panels, self._commands, self._tools), snapshots
            ):
                target.clear()
                target.update(snapshot)
            raise
        return result

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

    def register_tool(
        self,
        factory: ToolFactory,
        *,
        name: str | None = None,
        description: str = "",
    ) -> RegistrationHandle:
        """Register a zero-argument tool factory without creating its tool."""
        if name is None:
            if not isinstance(factory, type):
                raise ValueError(
                    "Tool factories that are not classes require an explicit name."
                )
            try:
                name = inspect.getattr_static(factory, "name")
            except AttributeError as exc:
                raise ValueError(
                    "Tool class must define a static 'name' or name must be provided."
                ) from exc
        return self.register_many(tools=(ToolFactorySpec(name, factory, description),))[
            0
        ]

    def register_many(
        self,
        *,
        panels: Iterable[PanelSpec] = (),
        commands: Iterable[CommandSpec] = (),
        tools: Iterable[ToolFactorySpec] = (),
    ) -> tuple[RegistrationHandle, ...]:
        """Register a batch atomically, rejecting invalid or duplicate names."""
        panel_specs, command_specs, tool_specs = (
            tuple(panels),
            tuple(commands),
            tuple(tools),
        )
        specs = (*panel_specs, *command_specs, *tool_specs)
        for spec in specs:
            self._validate(spec)
        self._check_ids(panel_specs, self._panels, "panel")
        self._check_ids(command_specs, self._commands, "command")
        self._check_ids(tool_specs, self._tools, "tool")

        registrations: list[tuple[Any, str, object]] = []
        for spec, target, key in (
            *((spec, self._panels, spec.id) for spec in panel_specs),
            *((spec, self._commands, spec.id) for spec in command_specs),
            *((spec, self._tools, spec.name) for spec in tool_specs),
        ):
            token = object()
            target[key] = (spec, token)
            registrations.append((target, key, token))

        return tuple(
            RegistrationHandle(
                lambda target=target, key=key, token=token: self._remove_if_owner(
                    target, key, token
                )
            )
            for target, key, token in registrations
        )

    @staticmethod
    def _check_ids(specs: tuple[Any, ...], target: dict[str, Any], kind: str) -> None:
        seen: set[str] = set()
        for spec in specs:
            key = spec.name if isinstance(spec, ToolFactorySpec) else spec.id
            if key in seen or key in target:
                raise ValueError(f"Duplicate {kind} id: {key!r}.")
            seen.add(key)

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

    def tools(self) -> tuple[ToolFactorySpec, ...]:
        """Return tool factory specifications in registration order."""
        return tuple(entry[0] for entry in self._tools.values())
