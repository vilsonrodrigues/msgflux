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


@pytest.mark.parametrize("deferred", [False, True])
def test_custom_function_compiles_schema_and_context_without_mutating_source(deferred):
    from msgflux.coding.extensions.records import ToolSpec

    calls = []

    def lookup(query: str) -> str:
        """Look up a value using the current workspace."""
        calls.append(query)
        return query

    lookup.tool_config = {
        "runtime_inputs": ("workspace",),
        "nested": {"value": 1},
    }
    original_config = {
        "runtime_inputs": ("workspace",),
        "nested": {"value": 1},
    }
    selection = (
        ToolSelection(deferred=("lookup",))
        if deferred
        else ToolSelection(active=("lookup",))
    )

    definition = resolve_tools(
        selection,
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(ToolSpec("lookup", lookup),),
    )[0]

    assert definition.name == "lookup"
    assert definition.input_schema["properties"]["query"]["type"] == "string"
    assert definition.context.bindings[0].source == "workspace"
    assert definition.loading.deferred is deferred
    assert definition.declaration["defer_loading"] is deferred
    assert definition.executor.tool_config["defer_loading"] is deferred
    assert lookup.tool_config == original_config
    assert calls == []
    library = ToolLibrary("coding-custom", [definition])
    if deferred:
        assert library.get_tool_json_schemas()[0]["function"]["name"] == "tool_search"
        from msgflux.chat_messages import ChatMessages

        messages = ChatMessages(thread_id="custom_tool_thread")
        assert library.get_handle().load_tools(messages, ["lookup"]) == ["lookup"]
        assert [
            schema["function"]["name"]
            for schema in library.get_tool_catalog_view(messages).portable_schemas()
        ] == ["lookup"]
    else:
        assert (
            library.get_tool_definition("lookup").input_schema
            == definition.input_schema
        )
        assert [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ] == ["lookup"]


@pytest.mark.parametrize("deferred", [False, True])
def test_custom_object_compiles_schema_and_preserves_its_config(deferred):
    from msgflux.coding.extensions.records import ToolSpec

    class ObjectTool:
        name = "object_tool"
        tool_config = {
            "runtime_inputs": ("workspace",),
            "nested": {"value": 2},
        }

        def __call__(self, path: str) -> str:
            """Read a path using the current workspace."""
            return path

    tool = ObjectTool()
    original_config = {
        "runtime_inputs": ("workspace",),
        "nested": {"value": 2},
    }
    selection = (
        ToolSelection(deferred=("object_tool",))
        if deferred
        else ToolSelection(active=("object_tool",))
    )

    definition = resolve_tools(
        selection,
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(ToolSpec("object_tool", tool),),
    )[0]

    assert definition.name == "object_tool"
    assert definition.input_schema["properties"]["path"]["type"] == "string"
    assert definition.context.bindings[0].source == "workspace"
    assert definition.loading.deferred is deferred
    assert definition.declaration["defer_loading"] is deferred
    assert definition.executor.tool_config["defer_loading"] is deferred
    assert tool.tool_config == original_config


@pytest.mark.parametrize("deferred", [False, True])
def test_nn_tool_instance_is_compiled_without_mutating_its_buffers(deferred):
    from msgflux.coding.extensions.records import ToolSpec
    from msgflux.nn.modules.tool.implementations import LocalTool

    def lookup(item: str) -> str:
        """Look up one item."""
        return item

    original_config = {
        "runtime_inputs": ("workspace",),
        "nested": {"value": 3},
    }
    tool = LocalTool(
        name="local_lookup",
        description="Look up one item.",
        annotations={"item": str},
        tool_config=dict(original_config),
        impl=lookup,
    )
    selection = (
        ToolSelection(deferred=("local_lookup",))
        if deferred
        else ToolSelection(active=("local_lookup",))
    )

    definition = resolve_tools(
        selection,
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(ToolSpec("local_lookup", tool),),
    )[0]

    assert definition.name == "local_lookup"
    assert definition.executor is not tool
    assert definition.input_schema["properties"]["item"]["type"] == "string"
    assert definition.context.bindings[0].source == "workspace"
    assert definition.loading.deferred is deferred
    assert tool.tool_config == original_config


@pytest.mark.parametrize("deferred", [False, True])
def test_class_tools_are_lazy_and_created_instances_are_reported_before_compile(
    deferred,
):
    from msgflux.coding.extensions.records import ToolSpec

    class ToolClass:
        name = "class_tool"
        tool_config = {"defer_loading": True}
        created = []

        def __init__(self):
            self.created.append(self)
            self.tool_config = {
                **self.tool_config,
                "runtime_inputs": ("workspace",),
            }

        def __call__(self, value: int) -> str:
            """Use a value with workspace context."""
            return str(value)

    spec = ToolSpec("class_tool", ToolClass)
    unused = resolve_tools(
        ToolSelection(active=("read",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(spec,),
    )
    assert ToolClass.created == []
    assert unused[0].name == "read"

    created = []
    definition = resolve_tools(
        (
            ToolSelection(deferred=("class_tool",))
            if deferred
            else ToolSelection(active=("class_tool",))
        ),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(spec,),
        on_tool_created=created.append,
    )[0]
    assert definition.name == "class_tool"
    assert definition.input_schema["properties"]["value"]["type"] == "integer"
    assert definition.context.bindings[0].source == "workspace"
    assert definition.loading.deferred is deferred
    assert definition.declaration["defer_loading"] is deferred
    assert definition.executor.tool_config["defer_loading"] is deferred
    assert created == ToolClass.created
    library = ToolLibrary(f"class-tool-{deferred}", [definition])
    assert ("tool_search" in library.get_tool_names()) is deferred


def test_class_created_callback_runs_before_definition_compile_failure(monkeypatch):
    from msgflux.coding.extensions.records import ToolSpec
    from msgflux.nn.modules.tool.library import ToolLibrary

    class InvalidTool:
        name = "invalid_tool"

        def __call__(self, value):
            """This missing parameter annotation makes compilation fail."""

    created = []

    def fail_compilation(_tool):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        ToolLibrary,
        "inspect_tool_definition",
        staticmethod(fail_compilation),
    )
    with pytest.raises(RuntimeError, match="compile failed"):
        resolve_tools(
            ToolSelection(active=("invalid_tool",)),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
            tool_specs=(ToolSpec("invalid_tool", InvalidTool),),
            on_tool_created=created.append,
        )
    assert len(created) == 1
    assert isinstance(created[0], InvalidTool)


def test_tool_specs_validate_collisions_and_unknown_names_without_compiling():
    from msgflux.coding.extensions.records import ToolSpec

    class BashToolSpec:
        name = "bash"

        def __call__(self):
            """Conflicting definition."""

    with pytest.raises(ValueError, match="collide"):
        resolve_tools(
            ToolSelection(),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
            tool_specs=(ToolSpec("bash", BashToolSpec),),
        )
    with pytest.raises(ValueError, match="Available names"):
        resolve_tools(
            ToolSelection(active=("missing",)),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
            tool_specs=(ToolSpec("custom", BashToolSpec),),
        )


def test_async_function_tool_is_registered_without_invoking_its_body():
    from msgflux.coding.extensions.records import ToolSpec

    calls = []

    async def async_lookup(query: str) -> str:
        """Look up a value asynchronously."""
        calls.append(query)
        return query

    definition = resolve_tools(
        ToolSelection(active=("async_lookup",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(ToolSpec("async_lookup", async_lookup),),
    )[0]

    assert definition.name == "async_lookup"
    assert definition.input_schema["properties"]["query"]["type"] == "string"
    assert calls == []


def test_group_expansion_rejects_cross_mode_duplicate_before_instantiation():
    with pytest.raises(ValueError, match="both active and deferred"):
        resolve_tools(
            ToolSelection(active=("workspace",), deferred=("read",)),
            model=_Model(),
            allow_edits=False,
            has_executor=False,
        )


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


def test_function_tool_spec_uses_its_compiled_name():
    from msgflux.coding.extensions.records import ToolSpec

    def fn_tool() -> str:
        """Run function tool."""
        return "ok"

    definition = resolve_tools(
        ToolSelection(active=("fn_tool",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=(ToolSpec("fn_tool", fn_tool),),
    )[0]
    assert definition.name == "fn_tool"
    assert definition.executor.impl is fn_tool


@pytest.mark.parametrize("instance", [False, True])
def test_tool_without_name_uses_class_name_consistently(instance):
    from msgflux.coding.extensions import CodingExtensions

    class Echo:
        def __call__(self, text: str) -> str:
            """Echo the supplied text."""
            return text

    extensions = CodingExtensions()
    extensions.register_tool(Echo() if instance else Echo)
    definitions = resolve_tools(
        ToolSelection(active=("Echo",)),
        model=_Model(),
        allow_edits=False,
        has_executor=False,
        tool_specs=extensions.tools(),
    )
    library = ToolLibrary("unnamed", definitions)
    response = library([("echo-call", "Echo", {"text": "hello"})])
    assert response.tool_calls[0].result == "hello"
