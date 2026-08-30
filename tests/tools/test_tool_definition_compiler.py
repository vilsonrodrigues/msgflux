from types import SimpleNamespace

from msgflux.nn.modules.tool import LocalTool, ToolLibrary
from msgflux.nn.modules.tool_v2 import ToolDefinitionCompiler
from msgflux.tools.config import tool_config


def test_library_compiles_legacy_config_once_into_canonical_definition():
    @tool_config(
        allow_background=True,
        background_capabilities=("activity",),
        inject_message=True,
        inject_messages=True,
        inject_handle=True,
        inject_vars=["tenant"],
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

    inspect_inventory.tool_config.background = True

    assert definition.dispatch.name == "optional_background"


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
