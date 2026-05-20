from __future__ import annotations

from typing import Any, Callable

import msgflux.nn.functional as F
from msgflux.sandbox.artifacts import ArtifactNamespace
from msgflux.sandbox.base import (
    BaseSandbox,
    SandboxCapabilities,
    _format_execution_output,
)
from msgflux.sandbox.context import get_ptc_allowed_tool_names
from msgflux.sandbox.registry import register_sandbox

try:
    from pydantic_monty import CollectString, MontyRepl
except ImportError:  # pragma: no cover - exercised by factory import path.
    CollectString = None  # type: ignore[assignment]
    MontyRepl = None  # type: ignore[assignment]


@register_sandbox
class MontySandbox(BaseSandbox):
    sandbox_type = "python"
    provider = "monty"
    name = "python_interpreter"
    display_name = "Python Interpreter"
    description = (
        "Execute Python code in a Monty sandboxed interpreter. Assign final output "
        "to `result`; use `print(...)` for debug output. Programmatic tools are "
        "async callables; always call them with `await tools['name'](...)`."
    )
    usage_guidance = (
        "Use `python_interpreter` for bounded runtime computation in a Monty "
        "sandbox. The interpreter returns captured stdout plus the value assigned "
        "to `result`; bare expressions are not returned by msgFlux. Use "
        "`print(...)` for debug output. Monty receives injected async tools and "
        "artifacts as dictionaries, so call `await tools['name'](...)` and "
        "`await artifacts['read'](...)`."
    )
    capabilities = SandboxCapabilities(
        language="python",
        programmatic_tool_calls=True,
        network=False,
        filesystem=False,
        persistence=True,
        snapshots=False,
        limitations=(
            "Runs through Monty's Python subset, so unsupported Python syntax or "
            "stdlib modules may fail.",
            "No direct network access is provided by the sandbox.",
            "No direct host filesystem access is provided by the sandbox.",
            "Injected programmatic tools are available as `await tools['name'](...)`, "
            "not as attribute access.",
            "Mounted artifacts are available as `await artifacts['read'](...)` "
            "and `await artifacts['info'](...)`, not as attribute access.",
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        if MontyRepl is None:
            raise ImportError(
                "`Sandbox.python('monty')` requires the optional dependency "
                "`pydantic-monty`. Install it with `msgflux[monty]`."
            )
        self._repl = MontyRepl()

    def render_description(
        self,
        *,
        tool_schemas: list[dict[str, Any]] | None = None,
        artifacts_enabled: bool = False,
    ) -> str:
        description = super().render_description(
            tool_schemas=tool_schemas,
            artifacts_enabled=False,
        )
        if not artifacts_enabled:
            return description
        return (
            description
            + "\n\nMounted artifacts are available through the `artifacts` "
            "dictionary. Use `await artifacts['info'](name)` for metadata and "
            "`await artifacts['read'](name, offset=0, limit=...)` for bounded "
            "text reads. `artifacts['read']` returns a str and requires keyword "
            "arguments for offset and limit."
        )

    def __call__(self, code: str) -> str:
        output = CollectString()
        external_functions = self._build_sync_external_functions()
        self._repl.feed_run(
            "result = None",
            inputs=self._build_inputs(),
            external_functions=external_functions,
            print_callback=output,
        )
        self._repl.feed_run(
            self._build_prelude() + "\n" + code,
            inputs=self._build_inputs(),
            external_functions=external_functions,
            print_callback=output,
        )
        result = self._read_result(external_functions=external_functions)
        return _format_execution_output(output.output, result)

    async def acall(self, code: str) -> str:
        output = CollectString()
        external_functions = self._build_async_external_functions()
        await self._repl.feed_run_async(
            "result = None",
            inputs=self._build_inputs(),
            external_functions=external_functions,
            print_callback=output,
        )
        await self._repl.feed_run_async(
            self._build_prelude() + "\n" + code,
            inputs=self._build_inputs(),
            external_functions=external_functions,
            print_callback=output,
        )
        result = await self._aread_result(external_functions=external_functions)
        return _format_execution_output(output.output, result)

    def _build_inputs(self) -> dict[str, Any]:
        return {"vars": dict(self._vars)}

    def _build_prelude(self) -> str:
        allowed_tool_names = sorted(get_ptc_allowed_tool_names())
        tool_entries = ", ".join(f"{name!r}: {name}" for name in allowed_tool_names)
        return (
            f"tools = {{{tool_entries}}}\n"
            "artifacts = {\n"
            "    'list': artifacts_list,\n"
            "    'info': artifacts_info,\n"
            "    'read': artifacts_read,\n"
            "    'search': artifacts_search,\n"
            "}"
        )

    def _build_sync_external_functions(self) -> dict[str, Callable[..., Any]]:
        functions = {
            name: _make_monty_sync_tool_proxy(tool)
            for name, tool in self._get_allowed_tools().items()
        }
        functions.update(self._build_artifact_functions(async_mode=False))
        return functions

    def _build_async_external_functions(self) -> dict[str, Callable[..., Any]]:
        functions = {
            name: _make_monty_async_tool_proxy(tool)
            for name, tool in self._get_allowed_tools().items()
        }
        functions.update(self._build_artifact_functions(async_mode=True))
        return functions

    def _build_artifact_functions(
        self,
        *,
        async_mode: bool,
    ) -> dict[str, Callable[..., Any]]:
        namespace = ArtifactNamespace(self._artifacts)
        if async_mode:
            return {
                "artifacts_list": namespace.alist,
                "artifacts_info": namespace.ainfo,
                "artifacts_read": namespace.aread,
                "artifacts_search": namespace.asearch,
            }
        return {
            "artifacts_list": namespace.list,
            "artifacts_info": namespace.info,
            "artifacts_read": namespace.read,
            "artifacts_search": namespace.search,
        }

    def _read_result(self, *, external_functions: dict[str, Callable[..., Any]]) -> Any:
        return self._repl.feed_run(
            "result",
            inputs=self._build_inputs(),
            external_functions=external_functions,
        )

    async def _aread_result(
        self,
        *,
        external_functions: dict[str, Callable[..., Any]],
    ) -> Any:
        return await self._repl.feed_run_async(
            "result",
            inputs=self._build_inputs(),
            external_functions=external_functions,
        )


def _make_monty_sync_tool_proxy(tool: Callable[..., Any]):
    def _call(**kwargs: Any) -> Any:
        if hasattr(tool, "acall"):
            return F.wait_for(tool.acall, **kwargs)
        return tool(**kwargs)

    return _call


def _make_monty_async_tool_proxy(tool: Callable[..., Any]):
    async def _call(**kwargs: Any) -> Any:
        if hasattr(tool, "acall"):
            return await tool.acall(**kwargs)
        return tool(**kwargs)

    return _call
