from copy import deepcopy
from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.nn import ContextBinding
from msgflux.nn.modules.tool import LocalTool, ToolLibrary
from msgflux.nn.modules.tool_v2 import ToolDefinitionCompiler
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import ToolIntent


def test_library_compiles_legacy_config_once_into_canonical_definition():
    @tool_config(
        allow_background=True,
        background_capabilities=("activity",),
        runtime_inputs=[
            "message",
            "messages",
            "handle",
            ContextBinding(
                source="vars",
                parameter="tenant",
                options={"key": "tenant"},
            ),
        ],
        defer_loading=True,
    )
    def inspect_inventory(sku: str) -> str:
        """Inspect one inventory item."""
        return sku

    library = ToolLibrary(name="warehouse", tools=[inspect_inventory])
    definition = library.get_tool_definition("inspect_inventory")

    assert definition.dispatch.name == "optional_background"
    assert definition.dispatch.options["capabilities"] == ("activity",)
    assert definition.feedback.name == "model"
    assert [binding.source for binding in definition.context.bindings] == [
        "message",
        "messages",
        "handle",
        "vars",
    ]
    assert definition.context.bindings[-1].parameter == "tenant"
    assert definition.context.bindings[-1].options == {"key": "tenant"}
    assert definition.loading.deferred
    assert definition.kind == "tool"
    assert "run_in_background" in definition.input_schema["properties"]
    assert definition.metadata["declaration"]["allow_background"] is True

    inspect_inventory.tool_config.background = True
    captured = library.library["tool_search"].impl.tools["inspect_inventory"]
    captured.source_tool.tool_config.allow_background = False

    assert definition.dispatch.name == "optional_background"
    assert definition.metadata["declaration"]["allow_background"] is True


def test_deferred_loading_uses_compiled_definition_after_config_mutation():
    @tool_config(defer_loading=True)
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    search_bucket = library.library["tool_search"].impl
    search_bucket.tools["lookup"].tool_config["defer_loading"] = False
    messages = ChatMessages(thread_id="thread_1")

    outcome = library.execute_intents(
        [ToolIntent(id="call_1", name="lookup", arguments={"query": "SKU-1"})],
        messages=messages,
    )[0]

    assert outcome.result == "SKU-1"
    assert messages.get_loaded_tools(library.name) == {"lookup"}


def test_library_registry_indexes_public_and_captured_definitions():
    def calculate(value: int) -> int:
        """Calculate one value."""
        return value * 2

    @tool_config(defer_loading=True)
    def remote_lookup(query: str) -> str:
        """Look up one remote value."""
        return query

    library = ToolLibrary(name="runtime", tools=[calculate, remote_lookup])
    search_bucket = library.library["tool_search"].impl

    assert {definition.name for definition in library.registry.definitions()} == {
        "calculate",
        "remote_lookup",
        "tool_search",
    }
    assert library.registry.get("calculate").executor is library.library["calculate"]
    assert (
        library.registry.get("remote_lookup").executor
        is search_bucket.tools["remote_lookup"].source_tool
    )

    copied = deepcopy(library)
    copied_search = copied.library["tool_search"].impl

    assert copied.registry.get("calculate").executor is copied.library["calculate"]
    assert (
        copied.registry.get("remote_lookup").executor
        is copied_search.tools["remote_lookup"].source_tool
    )

    library.remove("remote_lookup")

    assert not library.registry.has("remote_lookup")
    assert not library.registry.has("tool_search")


def test_library_catalog_view_is_scoped_to_chat_messages_thread():
    @tool_config(defer_loading=True)
    def lookup(query: str) -> str:
        """Look up one value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    first = ChatMessages(thread_id="thread_a")
    second = ChatMessages(thread_id="thread_b")
    first.load_tools(library.name, ["lookup", "removed_tool"])

    first_view = library.get_tool_catalog_view(first)
    second_view = library.get_tool_catalog_view(second)

    assert first_view.thread_id == "thread_a"
    assert [entry.name for entry in first_view.visible_entries()] == ["lookup"]
    assert [entry.name for entry in second_view.visible_entries()] == ["tool_search"]
    assert [entry.name for entry in second_view.entries] == ["tool_search", "lookup"]


def test_library_catalog_view_requires_configured_thread():
    library = ToolLibrary(name="empty", tools=[])

    with pytest.raises(ValueError, match="configured thread id"):
        library.get_tool_catalog_view(ChatMessages())


def test_feedback_flags_compile_to_one_feedback_axis():
    def implementation() -> str:
        """Return a value."""
        return "ok"

    executor = LocalTool(
        name="implementation",
        description="Return a value.",
        annotations={"return": str},
        tool_config={},
        impl=implementation,
    )
    metadata = ToolLibrary.inspect_tool_metadata(implementation)

    expected = {
        "return_direct": "direct",
        "handoff": "handoff",
        "call_as_response": "call_as_response",
    }
    for flag, feedback in expected.items():
        metadata.tool_config = {flag: True, "tool_kind": "tool"}
        definition = ToolDefinitionCompiler.compile(metadata, executor=executor)
        assert definition.feedback.name == feedback


def test_mcp_tools_receive_canonical_definitions_without_losing_executor_type():
    tool_info = SimpleNamespace(
        name="lookup",
        description="Remote lookup.",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )

    class Client:
        async def call_tool(self, _name, _arguments):
            raise AssertionError("This compilation test must not execute the tool")

    remote = ToolLibrary.create_mcp_tool(
        name="lookup",
        mcp_client=Client(),
        mcp_tool_info=tool_info,
        namespace="search",
        config={"defer_loading": True},
    )
    library = ToolLibrary(name="remote", tools=[remote])

    definition = library.get_tool_definition("search__lookup")

    assert definition.executor is remote
    assert definition.input_schema == tool_info.inputSchema
    assert definition.loading.deferred


def test_model_catalog_projects_compiled_definition_without_rebuilding_schema():
    def lookup(query: str) -> str:
        """Look up a value."""
        return query

    library = ToolLibrary(name="search", tools=[lookup])
    tool = library.library["lookup"]

    def unexpected_schema_rebuild():
        raise AssertionError("Catalog generation must use the compiled definition")

    tool.get_json_schema = unexpected_schema_rebuild
    catalog = library.get_tool_catalog()

    assert len(catalog.tools) == 1
    assert catalog.tools[0].name == "lookup"
    assert catalog.tools[0].parameters["properties"]["query"] == {"type": "string"}
