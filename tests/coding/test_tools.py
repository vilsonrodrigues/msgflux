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


def test_custom_factories_are_lazy_and_config_is_copied_per_selection():
    from msgflux.coding.extensions.records import ToolFactorySpec

    calls = []

    class CustomTool:
        name = "custom"
        tool_config = {"defer_loading": True, "setting": {"value": 1}}

        def __init__(self):
            calls.append("created")

    specs = (ToolFactorySpec(name="custom", factory=CustomTool),)
    active = resolve_tools(
        ToolSelection(active=("read",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_factories=specs,
    )
    assert calls == []
    custom = resolve_tools(
        ToolSelection(active=("custom",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_factories=specs,
    )[0]
    assert calls == ["created"]
    assert custom.tool_config == {"defer_loading": False, "setting": {"value": 1}}
    assert CustomTool.tool_config["defer_loading"] is True
    assert active[0].name == "read"


def test_factory_collisions_and_unknown_names_fail_before_instantiation():
    from msgflux.coding.extensions.records import ToolFactorySpec

    calls = []

    def factory():
        calls.append(True)
        return object()

    collision = (ToolFactorySpec(name="bash", factory=factory),)
    with pytest.raises(ValueError, match="collide"):
        resolve_tools(
            ToolSelection(),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
            tool_factories=collision,
        )
    spec = (ToolFactorySpec(name="custom", factory=factory),)
    with pytest.raises(ValueError, match="Available names"):
        resolve_tools(
            ToolSelection(active=("missing",)),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
            tool_factories=spec,
        )
    assert calls == []


def test_group_expansion_rejects_cross_mode_duplicate_before_instantiation():
    with pytest.raises(ValueError, match="both active and deferred"):
        resolve_tools(
            ToolSelection(active=("workspace",), deferred=("read",)),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
        )


def test_factory_description_is_applied_to_selected_tool_config():
    from msgflux.coding.extensions.records import ToolFactorySpec

    class CustomTool:
        name = "custom"
        description = "Class description"
        tool_config = {"setting": True}

    tools = resolve_tools(
        ToolSelection(active=("custom",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_factories=(
            ToolFactorySpec("custom", CustomTool, "Registered description"),
        ),
    )
    assert tools[0].tool_config["description"] == "Registered description"


def test_agents_group_includes_subagent_instance_without_calling_it():
    calls = []

    class Subagent:
        tool_config = {}

        def get_module_name(self):
            return "researcher"

        def __call__(self, *args, **kwargs):
            calls.append(True)

    subagent = Subagent()
    tools = resolve_tools(
        ToolSelection(active=("agents",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        agents=(subagent,),
    )
    assert tools[1] is subagent
    assert tools[1].tool_config["allow_background"] is True
    assert calls == []


def test_function_tool_factory_result_uses_function_name():
    from msgflux.coding.extensions.records import ToolFactorySpec

    def fn_tool() -> str:
        """Run function tool."""
        return "ok"

    tools = resolve_tools(
        ToolSelection(active=("fn_tool",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_factories=(ToolFactorySpec("fn_tool", lambda: fn_tool),),
    )
    assert tools[0] is fn_tool
