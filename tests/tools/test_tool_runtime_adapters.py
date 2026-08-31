from dataclasses import replace
from threading import Event

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.modules.tool_v2 import ToolDispatch
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.context import execution_context
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import ToolIntent, ToolOutcome


def test_library_executes_canonical_intent_with_compiled_feedback():
    @tool_config(return_direct=True)
    def double(value: int) -> int:
        return value * 2

    library = ToolLibrary(name="test", tools=[double])
    intent = ToolIntent(id="call_1", name="double", arguments={"value": 4})

    outcomes = library.execute_intents([intent])

    assert len(outcomes) == 1
    assert outcomes[0].result == 8
    assert outcomes[0].status == "completed"
    assert outcomes[0].feedback.name == "direct"


@pytest.mark.asyncio
async def test_library_aexecutes_canonical_intent_and_normalizes_failure():
    async def explode() -> None:
        raise RuntimeError("boom")

    library = ToolLibrary(name="test", tools=[explode])
    intent = ToolIntent(id="call_1", name="explode")

    outcomes = await library.aexecute_intents((intent,))

    assert outcomes[0].status == "execution_failed"
    assert outcomes[0].error.code == "tool_execution_failed"
    assert "boom" in outcomes[0].error.message


def test_library_normalizes_unknown_tool_as_not_found():
    library = ToolLibrary(name="test", tools=[])
    intent = ToolIntent(id="call_1", name="missing")

    outcomes = library.execute_intents([intent])

    assert outcomes[0].status == "not_found"
    assert outcomes[0].error.code == "tool_not_found"


def test_library_preserves_custom_compiled_feedback_mode():
    @tool_config(feedback="approval")
    def deploy(environment: str) -> str:
        return environment

    library = ToolLibrary(name="test", tools=[deploy])
    intent = ToolIntent(
        id="call_1",
        name="deploy",
        arguments={"environment": "staging"},
    )

    outcome = library.execute_intents([intent])[0]

    assert outcome.result == "staging"
    assert outcome.feedback.name == "approval"


def test_canonical_outcomes_preserve_intent_order_across_execution_paths():
    def double(value: int) -> int:
        return value * 2

    library = ToolLibrary(name="test", tools=[double])
    intents = (
        ToolIntent(id="missing", name="unknown"),
        ToolIntent(id="executed", name="double", arguments={"value": 3}),
    )

    outcomes = library.execute_intents(intents)

    assert [outcome.intent_id for outcome in outcomes] == ["missing", "executed"]
    assert [outcome.status for outcome in outcomes] == ["not_found", "completed"]
    assert outcomes[1].result == 6


def test_canonical_execution_preserves_arguments_transformed_by_hook():
    def double(value: int) -> int:
        return value * 2

    library = ToolLibrary(name="test", tools=[double])
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, arguments={"value": 5}),
    ).register(library)

    outcome = library.execute_intents(
        [ToolIntent(id="call_1", name="double", arguments={"value": 2})]
    )[0]

    assert outcome.result == 10
    assert outcome.metadata["arguments"] == {"value": 5}


def test_canonical_execution_returns_structured_blocked_outcome():
    def dangerous(command: str) -> str:
        raise AssertionError("blocked tool must not execute")

    library = ToolLibrary(name="test", tools=[dangerous])
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, block="Command denied."),
    ).register(library)

    outcome = library.execute_intents(
        [
            ToolIntent(
                id="call_1",
                name="dangerous",
                arguments={"command": "rm -rf ~"},
            )
        ]
    )[0]

    assert outcome.status == "blocked"
    assert outcome.error.code == "tool_blocked"
    assert outcome.error.message == "Command denied."


def test_call_as_response_produces_outcome_without_running_implementation():
    executed = False

    @tool_config(call_as_response=True)
    def request_confirmation(action: str) -> str:
        nonlocal executed
        executed = True
        return action

    library = ToolLibrary(name="test", tools=[request_confirmation])
    intent = ToolIntent(
        id="call_1",
        name="request_confirmation",
        arguments={"action": "deploy"},
    )

    outcome = library.execute_intents([intent])[0]
    legacy = library([("call_1", "request_confirmation", {"action": "deploy"})])

    assert executed is False
    assert outcome.status == "completed"
    assert outcome.feedback.name == "call_as_response"
    assert outcome.result is None
    assert legacy.return_directly is True
    assert legacy.tool_calls[0].parameters == {"action": "deploy"}


def test_detached_dispatch_produces_dispatched_outcome():
    completed = Event()

    @tool_config(detached=True)
    def refresh_index(name: str) -> str:
        completed.set()
        return name

    library = ToolLibrary(name="test", tools=[refresh_index])

    outcome = library.execute_intents(
        [
            ToolIntent(
                id="call_1",
                name="refresh_index",
                arguments={"name": "products"},
            )
        ]
    )[0]

    assert outcome.status == "dispatched"
    assert outcome.feedback.name == "model"
    assert outcome.metadata["arguments"] == {"name": "products"}
    assert completed.wait(timeout=1)


@pytest.mark.asyncio
async def test_canonical_async_execution_propagates_abort():
    started = False

    async def wait_for_work() -> str:
        nonlocal started
        started = True
        return "done"

    library = ToolLibrary(name="test", tools=[wait_for_work])
    signal = AbortSignal()
    signal.abort("operator stopped the run")

    with (
        execution_context(abort_signal=signal),
        pytest.raises(AbortRequestedError, match="operator stopped the run"),
    ):
        await library.aexecute_intents([ToolIntent(id="call_1", name="wait_for_work")])

    assert started is False


class QueueDispatch(ToolDispatch):
    def __init__(self):
        super().__init__("dispatch_queue", dispatch_name="queue")
        self.requests = []

    async def dispatch(self, request):
        self.requests.append(request)
        return ToolOutcome.dispatched(
            request.plan.intent,
            result="queued externally",
            metadata={"queue": "priority"},
        )


def test_custom_dispatch_extension_routes_sync_execution():
    queue = QueueDispatch()

    @tool_config(dispatch="queue")
    def publish_report(report_id: str) -> str:
        raise AssertionError("custom dispatcher must own execution")

    library = ToolLibrary(
        name="test",
        tools=[publish_report],
        extensions=[queue],
    )

    outcome = library.execute_intents(
        [
            ToolIntent(
                id="call_1",
                name="publish_report",
                arguments={"report_id": "rpt_42"},
            )
        ]
    )[0]

    assert library.has_extension("dispatch_queue")
    assert outcome.status == "dispatched"
    assert outcome.result == "queued externally"
    assert outcome.metadata == {
        "queue": "priority",
        "arguments": {"report_id": "rpt_42"},
    }
    assert queue.requests[0].context.get("handle") is library.get_handle()


@pytest.mark.asyncio
async def test_custom_dispatch_extension_routes_async_execution():
    queue = QueueDispatch()

    @tool_config(dispatch="queue")
    async def publish_report(report_id: str) -> str:
        raise AssertionError("custom dispatcher must own execution")

    library = ToolLibrary(
        name="test",
        tools=[publish_report],
        extensions=[queue],
    )

    outcome = (
        await library.aexecute_intents(
            [
                ToolIntent(
                    id="call_1",
                    name="publish_report",
                    arguments={"report_id": "rpt_42"},
                )
            ]
        )
    )[0]

    assert outcome.status == "dispatched"
    assert len(queue.requests) == 1
