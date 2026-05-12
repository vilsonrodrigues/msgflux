from unittest.mock import MagicMock, Mock

import msgflux as mf
from msgflux.context import execution_context
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, Module, ToolLibrary
from msgflux.runtime import (
    EventStream,
    EventType,
    emit_compaction_post,
    emit_compaction_pre,
)
from msgflux.tasks import TaskStore


def _tool_call_response(
    tool_name: str,
    parameters: dict,
    *,
    call_id: str = "call_1",
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, mf.msgspec_dumps(parameters))
    response.add(agg)
    response.reasoning = None
    response.metadata = {}
    return response


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _ScriptedModel:
    def __init__(self, responses):
        self.model_type = "chat_completion"
        self._responses = list(responses)

    def __call__(self, **kwargs):
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        return self._responses.pop(0)


def test_agent_stream_events_emits_model_and_tool_events():
    def add(a: int, b: int) -> int:
        """Add two values."""
        return a + b

    model = _ScriptedModel(
        [
            _tool_call_response("add", {"a": 2, "b": 3}),
            _text_response("5"),
        ]
    )
    agent = Agent(name="Assistant", model=model, tools=[add])

    events = agent.stream_events("Calculate 2 + 3.")
    names = [event.name for event in events]

    assert EventType.AGENT_START in names
    assert EventType.MODEL_REQUEST in names
    assert EventType.MODEL_RESPONSE in names
    assert EventType.TOOL_CALL in names
    assert EventType.TOOL_STARTED in names
    assert EventType.TOOL_RESULT in names
    assert EventType.AGENT_COMPLETE in names
    model_responses = [
        event.attributes["response_type"]
        for event in events
        if event.name == EventType.MODEL_RESPONSE
    ]
    assert model_responses == ["tool_call", "text_generation"]

    tool_started = next(
        event for event in events if event.name == EventType.TOOL_STARTED
    )
    assert tool_started.attributes["tool_call_id"] == "call_1"
    assert tool_started.attributes["tool_name"] == "add"
    assert tool_started.attributes["caller_name"] == "Assistant"
    assert tool_started.attributes["caller_namespace"] == "Assistant"
    assert tool_started.attributes["arguments"] == {"a": 2, "b": 3}

    tool_result = next(event for event in events if event.name == EventType.TOOL_RESULT)
    assert tool_result.attributes["caller_name"] == "Assistant"
    assert tool_result.attributes["caller_namespace"] == "Assistant"


def test_stream_events_callback_returns_result_and_receives_events():
    class Echo(Module):
        def forward(self, value: str) -> str:
            return value

    seen = []
    result = Echo().stream_events("ok", callback=seen.append)

    assert result == "ok"
    assert [event.name for event in seen] == [
        EventType.MODULE_START,
        EventType.MODULE_COMPLETE,
    ]


def test_task_store_and_inbox_emit_runtime_events():
    store = TaskStore()
    inbox = mf.AgentInbox()

    with EventStream() as stream:
        task = store.create("worker", task_id="task_1")
        store.set_running(task.task_id)
        store.update_progress(task.task_id, message="half", percent=50)
        store.complete(task.task_id, "done")
        inbox.user_message("new instruction")
        stream.close()
        events = stream.events

    names = [event.name for event in events]
    assert EventType.TASK_CREATED in names
    assert EventType.TASK_RUNNING in names
    assert EventType.TASK_PROGRESS in names
    assert EventType.TASK_COMPLETED in names
    assert EventType.INBOX_NOTIFICATION in names


def test_tool_metadata_reaches_otel_but_not_tool_impl():
    def inspect_payload(x: str) -> str:
        """Return a value without accepting runtime-only kwargs."""
        return x

    library = ToolLibrary("tools", [inspect_payload])

    with execution_context(namespace="Caller", session_id="s1", run_id="r1"):
        events = library.stream_events([("call_1", "inspect_payload", {"x": "ok"})])

    names = [event.name for event in events]
    assert EventType.TOOL_STARTED in names
    result = next(event for event in events if event.name == EventType.TOOL_RESULT)
    assert result.attributes["tool_call_id"] == "call_1"
    assert result.attributes["caller_name"] == "Caller"
    assert result.attributes["caller_namespace"] == "Caller"
    assert result.attributes["caller_session_id"] == "s1"


def test_checkpoint_save_emits_runtime_event():
    model = MagicMock()
    model.model_type = "chat_completion"
    response = Mock(spec=ModelResponse)
    response.response_type = "text_generation"
    response.consume.return_value = "ok"
    response.data = "ok"
    response.reasoning = None
    response.metadata = {}
    model.return_value = response

    checkpointer = mf.InMemoryCheckpointStore()
    agent = Agent(name="Assistant", model=model, checkpointer=checkpointer)
    messages = mf.ChatMessages(session_id="session_1")

    events = agent.stream_events(
        "hello",
        messages=messages,
        scope=mf.ExecutionScope(session_id="session_1", run_id="run_1"),
    )

    checkpoint_event = next(
        event for event in events if event.name == EventType.CHECKPOINT_SAVED
    )
    assert checkpoint_event.attributes["checkpoint_namespace"] == "Assistant"
    assert checkpoint_event.attributes["checkpoint_session_id"] == "session_1"
    assert checkpoint_event.attributes["status"] == "completed"


def test_compaction_events_are_available_for_hooks():
    with EventStream() as stream:
        emit_compaction_pre(
            target="messages",
            strategy="summarize",
            message_count=12,
            token_count=4096,
            metadata={"hook": "context_window"},
        )
        emit_compaction_post(
            target="messages",
            strategy="summarize",
            message_count_before=12,
            message_count_after=4,
            token_count_before=4096,
            token_count_after=900,
            metadata={"hook": "context_window"},
        )
        stream.close()
        events = stream.events

    assert [event.name for event in events] == [
        EventType.COMPACTION_PRE,
        EventType.COMPACTION_POST,
    ]
    assert events[0].attributes["target"] == "messages"
    assert events[0].attributes["strategy"] == "summarize"
    assert events[0].attributes["message_count"] == 12
    assert events[1].attributes["message_count_before"] == 12
    assert events[1].attributes["message_count_after"] == 4
