import asyncio
from copy import deepcopy
from types import SimpleNamespace

import msgspec
import pytest

from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import LocalTool, MCPTool
from msgflux.nn.modules.tool_v2 import (
    AfterToolPolicy,
    BeforeDispatchPolicy,
    BeforeToolPolicy,
    ContextBinding,
    ContextSpec,
    DispatchSpec,
    FeedbackSpec,
    LoadingSpec,
    ToolDefinition,
    ToolDispatch,
    ToolIntent,
    ToolLibraryV2,
    ToolOutcome,
    ToolPolicy,
    ToolRef,
    ToolRegistry,
    ToolRuntimeContext,
)
from msgflux.protocols.mcp.types import MCPContent, MCPToolResult
from msgflux.runtime.abort import AbortSignal
from msgflux.tools.definitions import ToolCatalog


class RecordingExecutor(Module):
    def __init__(self, *, result=None, error=None):
        super().__init__()
        self.result = result
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.release = None

    def forward(self, **arguments):
        self.calls.append(arguments)
        if self.error is not None:
            raise self.error
        return self.result

    async def aforward(self, **arguments):
        self.calls.append(arguments)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


def make_definition(name="lookup", **overrides):
    values = {
        "name": name,
        "executor": RecordingExecutor(result={"found": True}),
        "description": "Look up an inventory item.",
        "input_schema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
        },
    }
    values.update(overrides)
    return ToolDefinition(**values)


def make_intent(name="lookup"):
    return ToolIntent(id="call_1", name=name, arguments={"sku": "SKU-1842"})


def test_registry_owns_stable_definitions_without_executor_modules():
    definition = make_definition()
    registry = ToolRegistry("warehouse_tools", [definition])

    ref = registry.ref("lookup")

    assert ref == ToolRef(library_id="warehouse_tools", tool_id="lookup")
    assert registry.get(ref) is definition
    assert not hasattr(registry, "executors")
    with pytest.raises(ValueError, match="already registered"):
        registry.add(definition)

    removed = registry.remove(ref)

    assert removed is definition
    assert not registry.has("lookup")


def test_registry_rejects_foreign_refs():
    registry = ToolRegistry("warehouse_tools", [make_definition()])
    foreign = ToolRef(library_id="other_tools", tool_id="lookup")

    with pytest.raises(ValueError, match="belongs to `other_tools`"):
        registry.get(foreign)


def test_registry_replace_preserves_executor_ownership():
    definition = make_definition()
    registry = ToolRegistry("warehouse_tools", [definition])
    updated = msgspec.structs.replace(definition, description="Updated lookup.")

    previous = registry.replace(updated)

    assert previous is definition
    assert registry.get("lookup") is updated
    with pytest.raises(ValueError, match="cannot change its executor"):
        registry.replace(
            msgspec.structs.replace(updated, executor=RecordingExecutor(result="new"))
        )


def test_registry_deepcopy_copies_definition_executor_reference():
    registry = ToolRegistry("warehouse_tools", [make_definition()])

    copied = deepcopy(registry)
    definition = copied.get("lookup")

    assert definition.executor is not registry.get("lookup").executor


def test_library_facade_owns_executors_and_deepcopy_preserves_identity():
    library = ToolLibraryV2([make_definition()], name="warehouse_tools")

    copied = deepcopy(library)
    definition = copied.registry.get("lookup")

    assert definition.executor is copied.executors["lookup"]
    assert definition.executor is not library.registry.get("lookup").executor


def test_catalog_views_isolate_deferred_loading_by_thread():
    registry = ToolRegistry(
        "warehouse_tools",
        [
            make_definition(),
            make_definition(
                "reconcile_inventory",
                loading=LoadingSpec(deferred=True),
            ),
        ],
    )

    first = registry.catalog_view(
        "thread_a",
        loaded_tools=("reconcile_inventory",),
    )
    second = registry.catalog_view("thread_b")

    assert [entry.name for entry in first.visible_entries()] == [
        "lookup",
        "reconcile_inventory",
    ]
    assert [entry.name for entry in second.visible_entries()] == ["lookup"]
    assert first.entries[1].loaded
    assert not second.entries[1].loaded
    assert registry.get("reconcile_inventory").loading.deferred


def test_catalog_selection_can_expose_one_unloaded_deferred_tool():
    registry = ToolRegistry(
        "warehouse_tools",
        [make_definition("reconcile", loading=LoadingSpec(deferred=True))],
    )

    view = registry.catalog_view("thread_a", choice="reconcile")

    assert [entry.name for entry in view.visible_entries()] == ["reconcile"]
    assert view.choice.mode == "tool"


@pytest.mark.parametrize(
    ("choice", "expected_name"),
    [
        ({"type": "function", "function": {"name": "reconcile"}}, "reconcile"),
        ({"type": "function", "name": "reconcile"}, "reconcile"),
    ],
)
def test_catalog_choice_normalizes_provider_function_shapes(choice, expected_name):
    registry = ToolRegistry("warehouse_tools", [make_definition("reconcile")])

    view = registry.catalog_view("thread_a", choice=choice)

    assert view.choice.mode == "tool"
    assert view.choice.name == expected_name


def test_catalog_choice_rejects_provider_specific_non_function_shape():
    registry = ToolRegistry("warehouse_tools", [make_definition("reconcile")])

    with pytest.raises(ValueError, match="must select a function tool"):
        registry.catalog_view(
            "thread_a",
            choice={"type": "computer", "name": "reconcile"},
        )


def test_catalog_view_identifies_search_by_role_and_preserves_entry_metadata():
    search = make_definition(
        "discover",
        metadata={"catalog_role": "search"},
    )
    deferred = make_definition(
        "remote_lookup",
        loading=LoadingSpec(deferred=True),
        metadata={
            "strict": True,
            "execution_namespace": "remote",
        },
    )
    registry = ToolRegistry("warehouse_tools", [search, deferred])

    unloaded = registry.catalog_view("thread_a")
    loaded = registry.catalog_view(
        "thread_a",
        loaded_tools=("remote_lookup",),
    )
    selected = registry.catalog_view("thread_b", choice="remote_lookup")

    assert unloaded.search_entry.name == "discover"
    assert [entry.name for entry in unloaded.visible_entries()] == ["discover"]
    assert [entry.name for entry in loaded.visible_entries()] == ["remote_lookup"]
    assert [entry.name for entry in selected.visible_entries()] == ["remote_lookup"]
    assert unloaded.tool_entries()[0].strict is True
    assert unloaded.tool_entries()[0].namespace == "remote"


def test_legacy_catalog_adapter_preserves_canonical_view_semantics():
    registry = ToolRegistry(
        "warehouse_tools",
        [
            make_definition("discover", metadata={"catalog_role": "search"}),
            make_definition(
                "remote_lookup",
                loading=LoadingSpec(deferred=True),
                metadata={"strict": True, "execution_namespace": "remote"},
            ),
        ],
    )
    view = registry.catalog_view("thread_a", choice="remote_lookup")

    catalog = ToolCatalog.from_view(view)

    assert catalog.catalog_id == "warehouse_tools"
    assert catalog.choice == "remote_lookup"
    assert catalog.search_tool.name == "discover"
    assert catalog.tools[0].strict is True
    assert catalog.tools[0].namespace == "remote"
    assert catalog.tools[0].ref == registry.ref("remote_lookup")
    assert [tool.name for tool in catalog.portable_tools()] == ["remote_lookup"]


def test_catalog_view_can_project_public_subset_without_losing_registry_entries():
    registry = ToolRegistry(
        "warehouse_tools",
        [make_definition("bucket"), make_definition("captured")],
    )

    view = registry.catalog_view("thread_a", include_tools=("bucket",))

    assert [entry.name for entry in view.entries] == ["bucket"]
    assert registry.has("captured")


@pytest.mark.asyncio
async def test_library_resolves_opt_in_context_and_executes_foreground():
    executor = RecordingExecutor(result="available")
    definition = make_definition(
        executor=executor,
        feedback=FeedbackSpec(name="direct"),
        context=ContextSpec(
            bindings=(
                ContextBinding(
                    source="vars",
                    parameter="context",
                    options={"select": ("tenant",)},
                ),
            )
        ),
    )
    library = ToolLibraryV2([definition], name="warehouse_tools")

    outcome = await library.acall(
        make_intent(),
        ToolRuntimeContext(values={"vars": {"tenant": "north", "secret": 1}}),
    )

    assert outcome.status == "completed"
    assert outcome.result == "available"
    assert outcome.feedback.name == "direct"
    assert executor.calls == [{"sku": "SKU-1842", "context": {"tenant": "north"}}]


@pytest.mark.asyncio
async def test_library_executes_real_local_and_mcp_adapters():
    mcp_calls = []
    mcp_info = SimpleNamespace(
        description="Look up remote inventory.",
        inputSchema={
            "type": "object",
            "properties": {"sku": {"type": "string"}},
        },
    )

    class MCPClient:
        async def call_tool(self, name, arguments):
            mcp_calls.append((name, arguments))
            return MCPToolResult(content=[MCPContent(type="text", text="remote")])

    local = LocalTool(
        name="local_lookup",
        description="Look up local inventory.",
        annotations={"sku": str, "return": str},
        tool_config={},
        impl=lambda sku: f"local:{sku}",
    )
    remote = MCPTool(
        name="lookup",
        mcp_client=MCPClient(),
        mcp_tool_info=mcp_info,
        namespace="remote",
    )
    library = ToolLibraryV2(
        [
            make_definition("local_lookup", executor=local),
            make_definition(
                "remote__lookup",
                executor=remote,
                description=mcp_info.description,
                input_schema=mcp_info.inputSchema,
            ),
        ],
        name="warehouse_tools",
    )

    local_outcome = await library.acall(make_intent("local_lookup"))
    remote_outcome = await library.acall(make_intent("remote__lookup"))

    assert local_outcome.result == "local:SKU-1842"
    assert remote_outcome.result == "remote"
    assert mcp_calls == [("lookup", {"sku": "SKU-1842"})]


@pytest.mark.asyncio
async def test_library_returns_structured_not_found_and_execution_failures():
    executor = RecordingExecutor(error=RuntimeError("scanner offline"))
    library = ToolLibraryV2(
        [make_definition(executor=executor)],
        name="warehouse_tools",
    )

    missing = await library.acall(make_intent("missing"))
    failed = await library.acall(make_intent())

    assert missing.status == "not_found"
    assert missing.error.code == "tool_not_found"
    assert failed.status == "execution_failed"
    assert failed.error.code == "tool_execution_failed"
    assert failed.error.message == "scanner offline"


@pytest.mark.asyncio
async def test_library_dispatches_through_custom_extension():
    class QueueDispatch(ToolDispatch):
        def __init__(self):
            super().__init__("dispatch_queue", dispatch_name="queue")

        async def dispatch(self, request):
            return ToolOutcome.dispatched(
                request.plan.intent,
                result={"queue": request.plan.dispatch.options["queue"]},
            )

    executor = RecordingExecutor(result="must not execute")
    definition = make_definition(
        executor=executor,
        dispatch=DispatchSpec(name="queue", options={"queue": "durable"}),
        feedback=FeedbackSpec(name="direct"),
    )
    library = ToolLibraryV2(
        [definition],
        name="warehouse_tools",
        extensions=[QueueDispatch()],
    )

    outcome = await library.acall(make_intent())

    assert outcome.status == "dispatched"
    assert outcome.result == {"queue": "durable"}
    assert outcome.feedback.name == "direct"
    assert executor.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("run_in_background", [False, None])
async def test_optional_background_dispatch_removes_reserved_foreground_argument(
    run_in_background,
):
    executor = RecordingExecutor(result="foreground")
    definition = make_definition(
        executor=executor,
        dispatch=DispatchSpec(
            name="optional_background",
            options={"argument": "run_in_background"},
        ),
    )
    library = ToolLibraryV2([definition], name="warehouse_tools")
    intent = msgspec.structs.replace(
        make_intent(),
        arguments={
            "sku": "SKU-1842",
            "run_in_background": run_in_background,
        },
    )

    outcome = await library.acall(intent)

    assert outcome.status == "completed"
    assert outcome.result == "foreground"
    assert executor.calls == [{"sku": "SKU-1842"}]


@pytest.mark.asyncio
async def test_optional_background_dispatch_delegates_to_background_extension():
    requests = []

    class Scheduler:
        async def adispatch(self, request):
            requests.append(request)
            return ToolOutcome.dispatched(
                request.plan.intent,
                result={"task_id": "task_1"},
            )

    executor = RecordingExecutor(result="must not execute immediately")
    definition = make_definition(
        executor=executor,
        dispatch=DispatchSpec(name="optional_background"),
    )
    library = ToolLibraryV2([definition], name="warehouse_tools")
    intent = msgspec.structs.replace(
        make_intent(),
        arguments={"sku": "SKU-1842", "run_in_background": True},
    )

    outcome = await library.acall(
        intent,
        ToolRuntimeContext(values={"background_dispatcher": Scheduler()}),
    )

    assert outcome.status == "dispatched"
    assert outcome.result == {"task_id": "task_1"}
    assert requests[0].plan.visible_arguments == {"sku": "SKU-1842"}
    assert executor.calls == []


@pytest.mark.asyncio
async def test_library_runs_policy_extensions_sequentially_and_monotonically():
    observed = []

    class AddAuditArgument(ToolPolicy):
        def __init__(self):
            super().__init__("add_audit_argument")

        async def before_tool(self, payload):
            observed.append("add")
            intent = msgspec.structs.replace(
                payload.intent,
                arguments={**payload.intent.arguments, "audited": True},
            )
            return BeforeToolPolicy(
                intent=intent,
                definition=payload.definition,
                context=payload.context,
            )

    class BlockAuditedCall(ToolPolicy):
        def __init__(self):
            super().__init__("block_audited_call")

        async def before_tool(self, payload):
            observed.append(("block", payload.intent.arguments["audited"]))
            return ToolOutcome.failed(
                payload.intent,
                status="blocked",
                code="policy_denied",
                message="Audited calls are disabled.",
            )

    class MustNotRun(ToolPolicy):
        def __init__(self):
            super().__init__("must_not_run")

        async def before_tool(self, payload):
            observed.append("unexpected")
            return payload

    executor = RecordingExecutor(result="must not execute")
    library = ToolLibraryV2(
        [make_definition(executor=executor)],
        name="warehouse_tools",
        extensions=[AddAuditArgument(), BlockAuditedCall(), MustNotRun()],
    )

    outcome = await library.acall(make_intent())

    assert outcome.status == "blocked"
    assert outcome.error.code == "policy_denied"
    assert observed == ["add", ("block", True)]
    assert executor.calls == []


@pytest.mark.asyncio
async def test_before_tool_policy_errors_fail_closed_and_stop_the_chain():
    observed = []

    class BrokenPolicy(ToolPolicy):
        def __init__(self):
            super().__init__("broken_policy")

        async def before_tool(self, _payload):
            observed.append("broken")
            raise RuntimeError("policy backend unavailable")

    class LaterPolicy(ToolPolicy):
        def __init__(self):
            super().__init__("later_policy")

        async def before_tool(self, payload):
            observed.append("later")
            return payload

    executor = RecordingExecutor(result="must not execute")
    library = ToolLibraryV2(
        [make_definition(executor=executor)],
        name="warehouse_tools",
        extensions=[BrokenPolicy(), LaterPolicy()],
    )

    outcome = await library.acall(make_intent())

    assert outcome.status == "blocked"
    assert outcome.error.code == "tool_policy_failed"
    assert "failed closed" in outcome.error.message
    assert observed == ["broken"]
    assert executor.calls == []


@pytest.mark.asyncio
async def test_policy_can_change_dispatch_and_transform_outcome():
    class LocalFallbackPolicy(ToolPolicy):
        def __init__(self):
            super().__init__("local_fallback")

        async def before_dispatch(self, payload):
            plan = msgspec.structs.replace(
                payload.plan,
                dispatch=DispatchSpec(name="foreground"),
            )
            return BeforeDispatchPolicy(plan=plan, context=payload.context)

        async def after_tool(self, payload):
            outcome = msgspec.structs.replace(
                payload.outcome,
                result={"wrapped": payload.outcome.result},
            )
            return AfterToolPolicy(
                plan=payload.plan,
                outcome=outcome,
                context=payload.context,
            )

    definition = make_definition(dispatch="unavailable_remote_queue")
    library = ToolLibraryV2(
        [definition],
        name="warehouse_tools",
        extensions=[LocalFallbackPolicy()],
    )

    outcome = await library.acall(make_intent())

    assert outcome.status == "completed"
    assert outcome.result == {"wrapped": {"found": True}}


@pytest.mark.asyncio
async def test_library_converts_abort_into_interrupted_outcome():
    executor = RecordingExecutor(result="late")
    executor.release = asyncio.Event()
    library = ToolLibraryV2(
        [make_definition(executor=executor)],
        name="warehouse_tools",
    )
    signal = AbortSignal()

    pending = asyncio.create_task(
        library.acall(
            make_intent(),
            ToolRuntimeContext(values={"abort_signal": signal}),
        )
    )
    await executor.started.wait()
    signal.abort("operator stopped the run")
    outcome = await pending

    assert outcome.status == "interrupted"
    assert outcome.error.code == "tool_interrupted"
    assert outcome.error.message == "operator stopped the run"
