from __future__ import annotations

import inspect
from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import msgflux.nn.functional as F
from msgflux.sandbox.artifacts import ArtifactNamespace, ArtifactRef
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
    usage_guidance = ""
    annotations = {"code": str, "return": str}
    capabilities = SandboxCapabilities(language="unknown")

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._vars: dict[str, Any] = {}
        self._artifacts: dict[str, ArtifactRef] = {}

    def set_tools(self, tools: Mapping[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)

    def set_vars(self, values: Mapping[str, Any]) -> None:
        self._vars = dict(values)

    def set_artifacts(self, artifacts: Mapping[str, ArtifactRef]) -> None:
        self._artifacts = dict(artifacts)

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
        artifacts_enabled: bool = False,
    ) -> str:
        sections = [self.description.strip()]
        if artifacts_enabled:
            sections.append(
                "Mounted artifacts are available through the `artifacts` namespace. "
                "Use `artifacts.info(name)` for metadata and "
                "`artifacts.read(name, offset=0, limit=...)` for bounded text reads. "
                "`artifacts.read` returns a str and requires keyword arguments for "
                "offset and limit."
            )
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
        "Execute Python code in an isolated interpreter. Assign final output to "
        "`result`; use `print(...)` for debug output. Use `await "
        "tools.<name>(...)` to call programmatic tools that are explicitly "
        "available."
    )
    usage_guidance = (
        "Use `python_interpreter` for bounded runtime computation and controlled "
        "programmatic tool calls. The interpreter returns captured stdout plus "
        "the value assigned to `result`; bare expressions are not returned. Use "
        "`print(...)` for debug output. Prefer passing large context slices "
        "directly to another tool instead of retaining them in interpreter globals."
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
        previous_artifacts = self._globals.get("artifacts")
        previous_result = self._globals.pop("result", None)
        self._globals["tools"] = namespace
        self._globals["vars"] = dict(self._vars)
        self._globals["artifacts"] = ArtifactNamespace(self._artifacts)
        stdout = StringIO()
        try:
            with redirect_stdout(stdout):
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
            if previous_artifacts is None:
                self._globals.pop("artifacts", None)
            else:
                self._globals["artifacts"] = previous_artifacts
            if "result" not in self._globals and previous_result is not None:
                self._globals["result"] = previous_result
        return _format_execution_output(stdout.getvalue(), result)

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
        previous_artifacts = self._globals.get("artifacts")
        previous_result = self._globals.pop("result", None)
        self._globals["tools"] = namespace
        self._globals["vars"] = dict(self._vars)
        self._globals["artifacts"] = ArtifactNamespace(self._artifacts)
        stdout = StringIO()
        try:
            compiled = compile(
                code,
                "<msgflux-python-interpreter>",
                "exec",
                flags=PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            with redirect_stdout(stdout):
                execution_result = eval(compiled, self._globals, self._globals)  # noqa: S307
                if inspect.isawaitable(execution_result):
                    await execution_result
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
            if previous_artifacts is None:
                self._globals.pop("artifacts", None)
            else:
                self._globals["artifacts"] = previous_artifacts
            if "result" not in self._globals and previous_result is not None:
                self._globals["result"] = previous_result
        if inspect.isawaitable(result):
            result = await result
        return _format_execution_output(stdout.getvalue(), result)


def _format_execution_output(stdout: str, result: Any) -> str:
    stdout = stdout.rstrip()
    if result is None:
        return stdout
    result_text = str(result)
    if not stdout:
        return result_text
    return f"{stdout}\n{result_text}"


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
