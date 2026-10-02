import pytest

from msgflux.coding.config import ToolSelection
from msgflux.coding.tools import resolve_tools
from msgflux.nn import ToolLibrary


class _Model:
    provider = "openai"
    api_mode = "responses"
    native_tools = True


def test_workspace_resolves_by_permissions_and_model():
    selected = ToolSelection(active=("workspace",))
    read_only = resolve_tools(
        selected, model=_Model(), allow_edits=False, has_executor=False
    )
    writable = resolve_tools(
        selected, model=_Model(), allow_edits=True, has_executor=False
    )
    assert [tool.name for tool in read_only] == ["read", "ls", "glob", "grep"]
    assert [tool.name for tool in writable] == [
        "read",
        "ls",
        "glob",
        "grep",
        "apply_patch",
    ]


def test_process_requires_executor_and_replaces_query_helpers():
    selected = ToolSelection(active=("workspace", "process"))
    with pytest.raises(ValueError, match="executor"):
        resolve_tools(selected, model=_Model(), allow_edits=False, has_executor=False)
    tools = resolve_tools(
        selected, model=_Model(), allow_edits=False, has_executor=True
    )
    assert [tool.name for tool in tools] == ["read", "bash", "task"]


def test_cannot_defer_native_patch():
    with pytest.raises(ValueError, match="cannot be deferred"):
        resolve_tools(
            ToolSelection(deferred=("apply_patch",)),
            model=_Model(),
            allow_edits=True,
            has_executor=False,
        )


def test_deferred_tool_registers_search_in_real_library():
    tools = resolve_tools(
        ToolSelection(active=("read",), deferred=("grep",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
    )

    library = ToolLibrary("coding", tools)

    assert "tool_search" in library.get_tool_names()
    assert library.get_tool_definition("grep").loading.deferred


def test_codex_workspace_uses_patch_even_without_native_transport():
    from types import SimpleNamespace

    model = SimpleNamespace(provider="openai-codex", native_tools=False)
    tools = resolve_tools(
        ToolSelection(active=("workspace",)),
        model=model,
        allow_edits=True,
        has_executor=True,
    )
    assert [getattr(tool, "name", None) for tool in tools] == [
        "read",
        "bash",
        "apply_patch",
        "task",
    ]


def test_progress_tool_is_interactive_and_capability_driven():
    from types import SimpleNamespace

    selection = ToolSelection(active=("read",))
    model = SimpleNamespace(provider="other")
    tools = resolve_tools(
        selection, model=model, allow_edits=False, has_executor=False, interactive=True
    )
    assert tools[-1].name == "send_user_message"
    model.api_mode_capabilities = SimpleNamespace(assistant_commentary=True)
    assert (
        len(
            resolve_tools(
                selection,
                model=model,
                allow_edits=False,
                has_executor=False,
                interactive=True,
            )
        )
        == 1
    )
    del model.api_mode_capabilities
    assert (
        len(
            resolve_tools(
                selection,
                model=model,
                allow_edits=False,
                has_executor=False,
                interactive=False,
            )
        )
        == 1
    )


@pytest.mark.parametrize("deferred", [False, True])
def test_web_fetch_is_available_in_profile_and_tool_catalog(deferred):
    from msgflux.tools.builtin import WebFetchTool

    selection = (
        ToolSelection(deferred=("web_fetch",))
        if deferred
        else ToolSelection(active=("web_fetch",))
    )
    tools = resolve_tools(
        selection, model=_Model(), allow_edits=False, has_executor=False
    )
    assert len(tools) == 1
    assert isinstance(tools[0], WebFetchTool)
    library = ToolLibrary("coding-web-fetch", tools)
    assert library.get_tool_definition("web_fetch").loading.deferred is deferred
    assert ("tool_search" in library.get_tool_names()) is deferred
