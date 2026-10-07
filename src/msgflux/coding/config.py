"""User-editable configuration for the coding host."""

from __future__ import annotations

import tomllib
from pathlib import Path

import msgspec


class ToolSelection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Logical tool names or capability groups selected by a profile."""

    active: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()


class CodingProfile(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    tools: ToolSelection = ToolSelection()
    approvals: tuple[str, ...] = ()
    model: str | None = None
    reasoning_effort: str | None = None


class SubagentConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    models: tuple[str, ...] = ()
    description: str = ""


class CodingConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    default_profile: str = "lite"
    default_model: str | None = None
    reasoning_effort: str | None = None
    active_accounts: dict[str, str] = msgspec.field(default_factory=dict)
    profiles: dict[str, CodingProfile] = msgspec.field(default_factory=dict)
    agents: dict[str, SubagentConfig] = msgspec.field(default_factory=dict)

    def profile(self, name: str | None = None) -> CodingProfile:
        selected = name or self.default_profile
        if selected in self.profiles:
            return self.profiles[selected]
        if selected == "lite":
            return CodingProfile(tools=ToolSelection(active=("workspace",)))
        raise ValueError(f"Unknown coding profile: {selected!r}")


def load_config(path: Path, overrides: tuple[str, ...] = ()) -> CodingConfig:
    """Load one TOML file and apply ``-c dotted.key=value`` overrides."""
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for override in overrides:
        key, separator, raw = override.partition("=")
        parts = key.split(".")
        if not separator or any(not part for part in parts):
            raise ValueError("Configuration override must be dotted.key=value")
        value = tomllib.loads(f"value = {raw}")["value"]
        target = data
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override nested key under {part!r}")
            target = child
        target[parts[-1]] = value
    config = msgspec.convert(data, type=CodingConfig, strict=True)
    _validate_config(config)
    return config


def select_tools(
    profile_selection: ToolSelection,
    *,
    active: str | None = None,
    deferred: str | None = None,
) -> ToolSelection:
    """Return the profile selection or replace it with CLI tool lists.

    Supplying either list replaces both lists. CSV values are trimmed around
    each tool name; an empty whole argument means an empty list.
    """
    if active is None and deferred is None:
        return profile_selection

    selection = ToolSelection(
        active=_parse_tool_csv(active),
        deferred=_parse_tool_csv(deferred),
    )
    _validate_tool_selection(selection, "CLI tool selection")
    return selection


def _parse_tool_csv(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()
    items = tuple(item.strip() for item in value.split(","))
    if any(not item for item in items):
        raise ValueError("Tool lists cannot contain empty entries")
    return items


def _validate_tool_selection(selection: ToolSelection, context: str) -> None:
    active, deferred = selection.active, selection.deferred
    if any(not item for item in (*active, *deferred)):
        raise ValueError(f"{context} has an empty tool name")
    if len(set(active)) != len(active) or len(set(deferred)) != len(deferred):
        raise ValueError(f"{context} has duplicate tools")
    if set(active) & set(deferred):
        raise ValueError(f"{context} has active and deferred tools in common")


def _validate_config(config: CodingConfig) -> None:  # noqa: C901
    if not config.default_profile:
        raise ValueError("default_profile must be non-empty")
    config.profile()
    if config.default_model is not None and "/" not in config.default_model:
        raise ValueError("default_model must be provider/model-id")
    for name, profile in config.profiles.items():
        if not name:
            raise ValueError("Profile names must be non-empty")
        _validate_tool_selection(profile.tools, f"Profile {name!r}")
        if any(not item for item in profile.approvals):
            raise ValueError(f"Profile {name!r} has an empty approval tool name")
        if len(set(profile.approvals)) != len(profile.approvals):
            raise ValueError(f"Profile {name!r} has duplicate approval tools")
        if profile.model is not None and "/" not in profile.model:
            raise ValueError(f"Profile {name!r} model must be provider/model-id")
    for provider, account in config.active_accounts.items():
        if not provider or not account:
            raise ValueError("Active account provider and name must be non-empty")
    for name, agent in config.agents.items():
        if not name or not agent.models:
            raise ValueError("Configured agents need a name and at least one model")
        if len(set(agent.models)) != len(agent.models) or any(
            "/" not in model for model in agent.models
        ):
            raise ValueError(f"Agent {name!r} needs unique provider/model-id entries")


__all__ = [
    "CodingConfig",
    "CodingProfile",
    "SubagentConfig",
    "ToolSelection",
    "load_config",
    "select_tools",
]
