"""Instance-local registration for coding panels, commands, and tools."""

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
    ToolSpec,
)

_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


class CodingExtensions:
    """Instance-local registry of coding extension declarations."""

    def __init__(self) -> None:
        self._panels: dict[str, tuple[PanelSpec, object]] = {}
        self._commands: dict[str, tuple[CommandSpec, object]] = {}
        self._tools: dict[str, tuple[ToolSpec, object]] = {}

    @staticmethod
    def _validate(spec: PanelSpec | CommandSpec | ToolSpec) -> None:
        if isinstance(spec, ToolSpec):
            CodingExtensions._validate_tool(spec)
            return

        CodingExtensions._validate_ui(spec)

    @staticmethod
    def _validate_tool(spec: ToolSpec) -> None:
        if not isinstance(spec.name, str) or not _TOOL_NAME.fullmatch(spec.name):
            raise ValueError(f"Invalid tool name: {spec.name!r}.")
        if not callable(spec.tool):
            raise TypeError("Tool must be callable.")
        if isinstance(spec.tool, type):
            try:
                inspect.signature(spec.tool).bind()
            except (TypeError, ValueError) as exc:
                raise TypeError("Tool class must accept zero arguments.") from exc

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
        tool: Any,
    ) -> RegistrationHandle:
        """Register a tool definition without creating or invoking it."""
        name = self._tool_name(tool)
        return self.register_many(tools=(ToolSpec(name, tool),))[0]

    @staticmethod
    def _tool_name(tool: Any) -> str:
        config = getattr(tool, "tool_config", None)
        if config is not None:
            if isinstance(config, dict):
                overridden = config.get("name_overridden")
            else:
                overridden = getattr(config, "name_overridden", None)
            if overridden:
                return overridden

        name = getattr(tool, "name", None)
        if name:
            return name

        name = getattr(tool, "__name__", None)
        if name:
            return name

        get_module_name = getattr(tool, "get_module_name", None)
        if callable(get_module_name):
            name = get_module_name()
            if name:
                return name

        return type(tool).__name__

    def register_many(
        self,
        *,
        panels: Iterable[PanelSpec] = (),
        commands: Iterable[CommandSpec] = (),
        tools: Iterable[ToolSpec] = (),
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
            key = spec.name if isinstance(spec, ToolSpec) else spec.id
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

    def tools(self) -> tuple[ToolSpec, ...]:
        """Return registered tool definitions in registration order."""
        return tuple(entry[0] for entry in self._tools.values())
