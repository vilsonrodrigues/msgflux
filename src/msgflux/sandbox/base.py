from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import msgflux.nn.functional as F
from msgflux.sandbox.context import get_ptc_allowed_tool_names


@dataclass(frozen=True)
class SandboxCapabilities:
    language: str
    programmatic_tool_calls: bool = False
    network: bool = False
    filesystem: bool = False
    persistence: bool = False
    snapshots: bool = False
    limitations: tuple[str, ...] = field(default_factory=tuple)


class ToolNamespace:
    """Namespaced view of tools exposed to code running inside a sandbox."""

    def __init__(self, tools: Mapping[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)

    def __getattr__(self, name: str):
        tool = self._tools.get(name)
        if tool is None:
            raise AttributeError(f"Tool `{name}` is not available.")
        return tool

    def available(self) -> list[str]:
        return sorted(self._tools)


class BaseSandbox:
    """Base callable sandbox that can be registered as a msgFlux tool."""

    name = "sandbox"
    display_name = "Sandbox"
    description = "Execute code in a sandbox."
    annotations = {"code": str, "return": str}
    capabilities = SandboxCapabilities(language="unknown")

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._vars: dict[str, Any] = {}

    def set_tools(self, tools: Mapping[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)

    def set_vars(self, values: Mapping[str, Any]) -> None:
        self._vars = dict(values)

    def _get_allowed_tools(self) -> dict[str, Callable[..., Any]]:
        allowed_tool_names = get_ptc_allowed_tool_names()
        return {
            name: tool
            for name, tool in self._tools.items()
            if name in allowed_tool_names
        }

    def render_description(
        self,
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> str:
        sections = [self.description.strip()]
        limitations = self.capabilities.limitations
        if limitations:
            sections.append(
                "Limitations:\n"
                + "\n".join(f"- {limitation}" for limitation in limitations)
            )
        if tool_schemas:
            tool_lines = []
            for schema in tool_schemas:
                function = schema.get("function", {})
                name = function.get("name")
                description = function.get("description") or "No description."
                if name:
                    tool_lines.append(f"- tools.{name}(...): {description}")
            if tool_lines:
                sections.append(
                    "Programmatic tools available inside this sandbox:\n"
                    + "\n".join(tool_lines)
                )
        else:
            sections.append("No programmatic tools are available inside this sandbox.")
        return "\n\n".join(sections)

    def __call__(self, code: str) -> str:
        raise NotImplementedError

    async def acall(self, code: str) -> str:
        raise NotImplementedError


class LocalPythonSandbox(BaseSandbox):
    name = "python_interpreter"
    display_name = "Python Interpreter"
    description = (
        "Execute Python code in an isolated interpreter. Use `tools.<name>(...)` "
        "to call programmatic tools that are explicitly available."
    )
    capabilities = SandboxCapabilities(
        language="python",
        programmatic_tool_calls=True,
        network=False,
        filesystem=False,
        persistence=True,
        snapshots=False,
        limitations=(
            "No direct network access is provided by the sandbox.",
            "No direct host filesystem access is provided by the sandbox.",
            "External interactions must use injected programmatic tools.",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._globals: dict[str, Any] = {}

    def __call__(self, code: str) -> str:
        allowed_tools = self._get_allowed_tools()
        namespace = ToolNamespace(
            {
                name: _make_sync_tool_proxy(tool)
                for name, tool in allowed_tools.items()
            }
        )
        previous_tools = self._globals.get("tools")
        previous_vars = self._globals.get("vars")
        previous_result = self._globals.pop("result", None)
        self._globals["tools"] = namespace
        self._globals["vars"] = dict(self._vars)
        try:
            exec(code, self._globals, self._globals)  # noqa: S102
            result = self._globals.get("result")
        finally:
            if previous_tools is None:
                self._globals.pop("tools", None)
            else:
                self._globals["tools"] = previous_tools
            if previous_vars is None:
                self._globals.pop("vars", None)
            else:
                self._globals["vars"] = previous_vars
            if "result" not in self._globals and previous_result is not None:
                self._globals["result"] = previous_result
        return "" if result is None else str(result)

    async def acall(self, code: str) -> str:
        allowed_tools = self._get_allowed_tools()
        namespace = ToolNamespace(
            {
                name: _make_async_tool_proxy(tool)
                for name, tool in allowed_tools.items()
            }
        )
        previous_tools = self._globals.get("tools")
        previous_vars = self._globals.get("vars")
        previous_result = self._globals.pop("result", None)
        self._globals["tools"] = namespace
        self._globals["vars"] = dict(self._vars)
        try:
            exec(code, self._globals, self._globals)  # noqa: S102
            result = self._globals.get("result")
        finally:
            if previous_tools is None:
                self._globals.pop("tools", None)
            else:
                self._globals["tools"] = previous_tools
            if previous_vars is None:
                self._globals.pop("vars", None)
            else:
                self._globals["vars"] = previous_vars
            if "result" not in self._globals and previous_result is not None:
                self._globals["result"] = previous_result
        if inspect.isawaitable(result):
            result = await result
        return "" if result is None else str(result)


def _make_sync_tool_proxy(tool: Callable[..., Any]):
    def _call(**kwargs: Any) -> Any:
        if hasattr(tool, "acall"):
            return F.wait_for(tool.acall, **kwargs)
        if inspect.iscoroutinefunction(tool):
            return F.wait_for(tool, **kwargs)
        return tool(**kwargs)

    return _call


def _make_async_tool_proxy(tool: Callable[..., Any]):
    async def _call(**kwargs: Any) -> Any:
        if hasattr(tool, "acall"):
            return await tool.acall(**kwargs)
        if inspect.iscoroutinefunction(tool):
            return await tool(**kwargs)
        return tool(**kwargs)

    return _call


def sandbox_metadata(sandbox: BaseSandbox) -> SimpleNamespace:
    return SimpleNamespace(
        name=sandbox.name,
        display_name=getattr(sandbox, "display_name", sandbox.name),
        capabilities=sandbox.capabilities,
        limitations=sandbox.capabilities.limitations,
    )
