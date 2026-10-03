"""Resolve coding profile selections into per-session tool instances."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from inspect import isawaitable
from typing import Any

from msgflux.coding.config import ToolSelection
from msgflux.coding.extensions.records import ToolFactorySpec
from msgflux.tools.builtin import (
    AgentTool,
    ApplyPatchTool,
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    LsTool,
    ReadFileTool,
    SendUserMessageTool,
    TaskTool,
    WebFetchTool,
    WebSearchTool,
    WriteTool,
)

_GROUPS = {"workspace", "process", "agents"}
_MANAGED_NAMES = {"agent", "task", "tool_search", "send_user_message"}


def _builtin_factories(*, vision: bool) -> dict[str, Callable[[], Any]]:
    """Constructors for the builtins selectable by coding profiles."""
    return {
        "read": lambda: ReadFileTool(supports_vision=vision),
        "ls": LsTool,
        "glob": GlobTool,
        "grep": GrepTool,
        "bash": BashTool,
        "apply_patch": ApplyPatchTool,
        "edit": EditTool,
        "write": WriteTool,
        "web_search": WebSearchTool,
        "web_fetch": WebFetchTool,
    }


def _tool_name(tool: Any) -> str | None:
    name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
    if name is None:
        get_module_name = getattr(tool, "get_module_name", None)
        if callable(get_module_name):
            name = get_module_name()
    return name if isinstance(name, str) and name else None


def validate_tool_selection(
    selection: ToolSelection, *, tool_factories: tuple[ToolFactorySpec, ...] = ()
) -> None:
    """Validate names and registry collisions without constructing tools."""
    custom_names = [spec.name for spec in tool_factories]
    reserved = _GROUPS | _MANAGED_NAMES | set(_builtin_factories(vision=False))
    collisions = sorted(set(custom_names) & reserved)
    if collisions:
        raise ValueError(f"Custom tool names collide with reserved names: {collisions}")
    if len(custom_names) != len(set(custom_names)):
        raise ValueError("Duplicate custom tool factory names")
    requested = (*selection.active, *selection.deferred)
    available = sorted(
        set(_builtin_factories(vision=False)) | set(custom_names) | _GROUPS
    )
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown coding tool or capability: {unknown[0]!r}. "
            f"Available names: {', '.join(available)}"
        )


def resolve_tools(  # noqa: C901
    selection: ToolSelection,
    *,
    model,
    allow_edits: bool,
    has_executor: bool,
    agents: tuple = (),
    interactive: bool = False,
    tool_factories: tuple[ToolFactorySpec, ...] = (),
) -> list:
    """Resolve selected tools, constructing only selected tools for this session.

    ``workspace`` is an abstract capability: listing/search tools are useful
    without a shell; edit tools appear only with explicit write permission.
    Custom factories must be synchronous and return an instance whose declared
    ``name`` matches the registered name.
    """
    validate_tool_selection(selection, tool_factories=tool_factories)
    custom = {spec.name: spec for spec in tool_factories}
    requested = (*selection.active, *selection.deferred)
    if "process" in requested and not has_executor:
        raise ValueError("The process capability requires a workspace executor")
    if "agents" in requested and not agents:
        raise ValueError("The agents capability needs configured subagents")

    openai_patch = getattr(model, "provider", None) in {"openai", "openai-codex"}
    openai_native_patch = (
        getattr(model, "provider", None) == "openai"
        and getattr(model, "api_mode", None) == "responses"
        and getattr(model, "native_tools", False)
    )
    vision = "image" in getattr(
        getattr(getattr(model, "profile", None), "modalities", None), "input", ()
    )
    builtins = _builtin_factories(vision=vision)
    agent_names = [_tool_name(agent) for agent in agents]
    if any(name is None for name in agent_names):
        raise ValueError("Configured agents must expose a public tool name")
    agent_collisions = sorted(set(agent_names) & set(custom))
    if agent_collisions:
        raise ValueError(
            f"Custom tool names collide with configured agents: {agent_collisions}"
        )
    if len(agent_names) != len(set(agent_names)):
        raise ValueError("Configured agents have duplicate public tool names")
    agent_reserved = _GROUPS | _MANAGED_NAMES | set(builtins)
    reserved_agents = sorted(set(agent_names) & agent_reserved)
    if reserved_agents:
        raise ValueError(
            f"Configured agent names collide with reserved tools: {reserved_agents}"
        )
    background_names = {"agent", *agent_names}
    use_shell = has_executor and any(
        item in selection.active for item in ("workspace", "process", "bash")
    )

    def construct(name: str) -> Any:
        if name in custom:
            return custom[name].factory()
        return builtins[name]()

    def group_factories(name: str) -> list[tuple[str, Callable[[], Any]]]:
        if name == "workspace":
            factories: list[tuple[str, Callable[[], Any]]] = [
                ("read", lambda: construct("read"))
            ]
            if use_shell:
                factories.append(("bash", lambda: construct("bash")))
            else:
                factories.extend(
                    (n, lambda n=n: construct(n)) for n in ("ls", "glob", "grep")
                )
            if allow_edits:
                factories.append(
                    (
                        "apply_patch" if openai_patch else "edit",
                        lambda n="apply_patch" if openai_patch else "edit": construct(
                            n
                        ),
                    )
                )
                if not openai_patch:
                    factories.append(("write", lambda: construct("write")))
            return factories
        if name == "process":
            return [("bash", lambda: construct("bash"))]
        return [
            ("agent", AgentTool),
            *(
                (tool_name, lambda agent=agent: agent)
                for tool_name, agent in zip(agent_names, agents)
            ),
        ]

    plan: list[tuple[str, bool, Callable[[], Any]]] = []
    modes: dict[str, bool] = {}
    for name, deferred in (
        *((item, False) for item in selection.active),
        *((item, True) for item in selection.deferred),
    ):
        entries = (
            group_factories(name)
            if name in _GROUPS
            else [(name, lambda name=name: construct(name))]
        )
        for logical_name, factory in entries:
            previous_mode = modes.get(logical_name)
            if previous_mode is not None and previous_mode != deferred:
                raise ValueError(
                    f"Tool {logical_name!r} is selected as both active and deferred"
                )
            if logical_name in modes:
                continue
            modes[logical_name] = deferred
            if logical_name == "bash" and not has_executor:
                raise ValueError("Bash requires a workspace executor")
            if logical_name in {"write", "edit", "apply_patch"} and not allow_edits:
                raise ValueError(f"{logical_name} requires edit permissions")
            if (
                deferred
                and logical_name in {"bash", "apply_patch"}
                and openai_native_patch
            ):
                raise ValueError(f"Native tool {logical_name} cannot be deferred")
            plan.append((logical_name, deferred, factory))

    resolved: list[Any] = []
    has_background = False
    for logical_name, deferred, factory in plan:
        tool = factory()
        if isawaitable(tool):
            close = getattr(tool, "close", None)
            if callable(close):
                close()
            raise TypeError("Tool factories must be synchronous")
        tool_name = _tool_name(tool)
        if tool_name != logical_name:
            raise ValueError(
                f"Tool factory for {logical_name!r} returned tool named {tool_name!r}"
            )
        config = deepcopy(getattr(tool, "tool_config", {}))
        spec = custom.get(logical_name)
        if spec is not None and spec.description:
            config["description"] = spec.description
        if logical_name == "bash" or logical_name in background_names:
            config["allow_background"] = True
            has_background = True
        config["defer_loading"] = bool(deferred)
        tool.tool_config = config
        resolved.append(tool)
    if has_background:
        resolved.append(TaskTool())
    capabilities = getattr(model, "api_mode_capabilities", None)
    if interactive and not getattr(capabilities, "assistant_commentary", False):
        resolved.append(SendUserMessageTool())
    return resolved


__all__ = ["resolve_tools", "validate_tool_selection"]
