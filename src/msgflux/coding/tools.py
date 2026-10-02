"""Resolve coding profile capabilities into the existing builtin tools."""

from __future__ import annotations

from copy import deepcopy

from msgflux.coding.config import ToolSelection
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


def resolve_tools(  # noqa: C901
    selection: ToolSelection,
    *,
    model,
    allow_edits: bool,
    has_executor: bool,
    agents: tuple = (),
    interactive: bool = False,
) -> list:
    """Select tools for live host capabilities and the chosen model transport.

    ``workspace`` is an abstract capability: listing/search tools are useful
    without a shell; edit tools appear only with explicit write permission.
    """
    requested = (*selection.active, *selection.deferred)
    if "process" in requested and not has_executor:
        raise ValueError("The process capability requires a workspace executor")
    use_shell = has_executor and any(
        item in selection.active for item in ("workspace", "process", "bash")
    )
    openai_patch = getattr(model, "provider", None) in {"openai", "openai-codex"}
    openai_native_patch = (
        getattr(model, "provider", None) == "openai"
        and getattr(model, "api_mode", None) == "responses"
        and getattr(model, "native_tools", False)
    )
    vision = "image" in getattr(
        getattr(getattr(model, "profile", None), "modalities", None), "input", ()
    )
    workspace = [ReadFileTool(supports_vision=vision)]
    if use_shell:
        workspace.append(BashTool())
    if not use_shell:
        workspace.extend((LsTool(), GlobTool(), GrepTool()))
    if allow_edits:
        workspace.extend(
            (ApplyPatchTool(),) if openai_patch else (EditTool(), WriteTool())
        )
    groups = {
        "workspace": workspace,
        "process": [BashTool()],
        "agents": [AgentTool(), *agents] if agents else [],
    }
    concrete = {
        "read": ReadFileTool,
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
    resolved = []
    seen = set()
    for name, deferred in (
        *((item, False) for item in selection.active),
        *((item, True) for item in selection.deferred),
    ):
        if name == "agents" and not agents:
            raise ValueError("The agents capability needs configured subagents")
        if name in groups:
            items = groups[name]
        elif name in concrete:
            items = [concrete[name]()]
        else:
            raise ValueError(f"Unknown coding tool or capability: {name!r}")
        for tool in items:
            tool_name = getattr(tool, "name", None)
            if tool_name in seen:
                continue
            if tool_name in {"bash"} and not has_executor:
                raise ValueError("Bash requires a workspace executor")
            if tool_name in {"write", "edit", "apply_patch"} and not allow_edits:
                raise ValueError(f"{tool_name} requires edit permissions")
            if tool_name in {"bash", "agent"}:
                tool.tool_config = deepcopy(tool.tool_config)
                tool.tool_config["allow_background"] = True
            if deferred:
                if tool_name in {"bash", "apply_patch"} and openai_native_patch:
                    raise ValueError(f"Native tool {tool_name} cannot be deferred")
                tool.tool_config = deepcopy(getattr(tool, "tool_config", {}))
                tool.tool_config["defer_loading"] = True
            resolved.append(tool)
            seen.add(tool_name)
    if any(getattr(tool, "name", None) in {"bash", "agent"} for tool in resolved):
        resolved.append(TaskTool())
    capabilities = getattr(model, "api_mode_capabilities", None)
    if interactive and not getattr(capabilities, "assistant_commentary", False):
        resolved.append(SendUserMessageTool())
    return resolved


__all__ = ["resolve_tools"]
