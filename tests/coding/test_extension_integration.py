"""Exercise extension composition through the CLI with real tools and SQLite."""

import sys
from types import ModuleType

import pytest

from msgflux.coding.cli import _load_extensions, _parser, _run
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.generator import Generator
from msgflux.runtime import AgentWorkspace
from msgflux.tools.types import Hidden


def _install(monkeypatch, register):
    module = ModuleType("coding_integration_extension")
    module.register = register
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return f"{module.__name__}:register"


def _response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("deferred", [False, True])
async def test_print_extension_executes_with_injected_workspace_and_closes(
    tmp_path, monkeypatch, capsys, deferred
):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    project = tmp_path / "project"
    project.mkdir()
    (project / "note.txt").write_text("cobalt")
    constructed, closed, calls = [], [], []

    class ReadNote:
        name = "read_note"
        tool_config = {"runtime_inputs": ["workspace"]}

        def __init__(self):
            constructed.append(self)

        async def acall(self, workspace: Hidden[AgentWorkspace] = None) -> str:
            """Read the workspace's note."""
            calls.append(workspace)
            return await workspace.aread_text("note.txt")

        def __call__(self, workspace: Hidden[AgentWorkspace] = None) -> str:
            """Read the workspace's note."""
            return workspace.read_text("note.txt")

        async def aclose(self):
            closed.append(self)

    def register(c):
        c.register_tool(ReadNote)

    entry = _install(monkeypatch, register)
    generations = []

    async def generate(self, **kwargs):
        generations.append(kwargs)
        if deferred and len(generations) == 1:
            search_calls = ToolCallAggregator()
            search_calls.process(
                0, "search-call", "tool_search", '{"select":["read_note"]}'
            )
            response = ModelResponse()
            response.set_response_type("tool_call")
            response.add(search_calls)
            return response
        if len(generations) == (2 if deferred else 1):
            tool_calls = ToolCallAggregator()
            tool_calls.process(0, "note-call", "read_note", "{}")
            response = ModelResponse()
            response.set_response_type("tool_call")
            response.add(tool_calls)
            return response
        history = kwargs["messages"].to_chatml()
        assert any(
            item.get("role") == "tool" and "cobalt" in str(item) for item in history
        )
        return _response("verified cobalt")

    monkeypatch.setattr(Generator, "aforward", generate)
    state = tmp_path / "state"
    args = _parser().parse_args(
        [
            "--workspace",
            str(project),
            "--state-dir",
            str(state),
            "--model",
            "openai/gpt-4.1-mini",
            "--extension",
            entry,
            "--tools",
            "read_note",
            "--read-only",
            "-p",
            "read note",
        ]
    )
    await _run(args)
    assert len(constructed) == 1
    assert closed == constructed
    assert len(calls) == 1
    assert not calls[0].permissions.missing(("filesystem.read",))
    assert calls[0].permissions.missing(("filesystem.write",))
    assert "verified cobalt" in capsys.readouterr().out
    assert len(list((state / "threads").glob("*/checkpoints.sqlite3"))) == 1


@pytest.mark.asyncio
async def test_session_switch_creates_distinct_tools_and_preserves_lazy_drafts(
    tmp_path, monkeypatch
):
    from msgflux.coding.tui import CodingApp

    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    built, closed = [], []
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"

    class Echo:
        name = "echo_tool"
        tool_config = {"defer_loading": True}

        def __init__(self):
            built.append(self)

        def __call__(self, text: str) -> str:
            """Echo supplied text."""
            return text

        def close(self):
            closed.append(self)

    entry = _install(monkeypatch, lambda c: c.register_tool(Echo))

    async def generate(self, **kwargs):
        return _response("ok")

    monkeypatch.setattr(Generator, "aforward", generate)

    async def run_ui(self):
        assert not built
        assert not (state / "threads").exists()
        [event async for event in self.coding_session.stream("first")]
        assert built[0].tool_config["defer_loading"] is True
        assert not self.host.session.agent.tool_library.get_tool_definition(
            "echo_tool"
        ).loading.deferred
        await self.host.select()
        assert closed == built
        assert len(built) == 1
        [event async for event in self.host.session.stream("second")]
        assert len(built) == 2
        assert built[0] is not built[1]

    monkeypatch.setattr(CodingApp, "run_async", run_ui)
    args = _parser().parse_args(
        [
            "--workspace",
            str(project),
            "--state-dir",
            str(state),
            "--model",
            "openai/gpt-4.1-mini",
            "--extension",
            entry,
            "--tools",
            "echo_tool",
        ]
    )
    await _run(args)
    assert closed == built
    assert Echo.tool_config == {"defer_loading": True}


@pytest.mark.asyncio
async def test_failed_factory_closes_previously_created_async_tool(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    project = tmp_path / "project"
    project.mkdir()
    closed = []

    class Resource:
        name = "resource"

        def __call__(self) -> str:
            """Return a resource marker."""
            return "resource"

        async def aclose(self):
            closed.append(self)

    class BrokenTool:
        name = "broken"

        def __init__(self):
            raise RuntimeError("factory unavailable")

        def __call__(self) -> str:
            """Return a marker."""
            return "unreachable"

    def register(c):
        c.register_tool(Resource)
        c.register_tool(BrokenTool)

    entry = _install(monkeypatch, register)
    args = _parser().parse_args(
        [
            "--workspace",
            str(project),
            "--state-dir",
            str(tmp_path / "state"),
            "--model",
            "openai/gpt-4.1-mini",
            "--extension",
            entry,
            "--tools",
            "resource,broken",
            "-p",
            "start",
        ]
    )
    with pytest.raises(RuntimeError, match="factory unavailable"):
        await _run(args)
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_unknown_cli_selection_fails_before_session_resources(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"

    def forbidden_model(*args):
        pytest.fail("unknown selection must fail before model creation")

    monkeypatch.setattr("msgflux.coding.cli._make_model", forbidden_model)
    args = _parser().parse_args(
        [
            "--workspace",
            str(project),
            "--state-dir",
            str(state),
            "--model",
            "openai/gpt-4.1-mini",
            "--tools",
            "missing",
            "-p",
            "start",
        ]
    )
    with pytest.raises(ValueError, match="missing"):
        await _run(args)
    assert not (state / "threads").exists()


def test_entry_point_failure_rolls_back_partial_registrations(monkeypatch):
    from msgflux.coding.extensions import CodingExtensions

    registries = []

    def create_registry():
        registry = CodingExtensions()
        registries.append(registry)
        return registry

    monkeypatch.setattr("msgflux.coding.cli.CodingExtensions", create_registry)

    def register(c):
        c.register_command("partial", lambda args: args)
        raise RuntimeError("registration failed")

    entry = _install(monkeypatch, register)
    good = ModuleType("good_coding_extension")
    good.register = lambda c: c.register_command("good", lambda args: args)
    monkeypatch.setitem(sys.modules, good.__name__, good)
    with pytest.raises(RuntimeError, match="registration failed"):
        _load_extensions([f"{good.__name__}:register", entry])
    assert [command.id for command in registries[0].commands()] == ["good"]
    assert len(registries) == 1


def test_entry_point_handles_and_registry_remain_connected_to_host(monkeypatch):
    captured = []

    class EchoTool:
        name = "echo"

        def __call__(self, text: str) -> str:
            return text

    def register(c):
        captured.append((c, c.register_tool(EchoTool)))

    registry = _load_extensions([_install(monkeypatch, register)])
    extension, handle = captured[0]
    assert extension is registry
    handle.unregister()
    assert registry.tools() == ()
    extension.register_tool(EchoTool)
    assert [spec.name for spec in registry.tools()] == ["echo"]
