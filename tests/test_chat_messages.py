import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import AbortRequestedError
from msgflux.models.response import ModelStreamResponse
from msgflux.runtime import AbortSignal
from msgflux.runtime.context import (
    DEFAULT_NAMESPACE,
    ExecutionScope,
    execution_context,
    get_execution_context,
    get_execution_scope,
    thread_context,
)


def test_default_execution_scope_is_available():
    scope = get_execution_scope()

    assert scope.thread_id is None
    assert scope.namespace == DEFAULT_NAMESPACE


def test_execution_context_accepts_scope_and_explicit_overrides():
    base_scope = ExecutionScope(
        thread_id="thread_1",
        namespace="root",
        run_id="run_1",
    )

    with execution_context(scope=base_scope, namespace="agent"):
        scope = get_execution_scope()

    assert scope.thread_id == "thread_1"
    assert scope.namespace == "agent"
    assert scope.run_id == "run_1"
    assert scope.root_run_id == "run_1"


def test_execution_context_exposes_abort_signal():
    abort_signal = AbortSignal()
    scope = ExecutionScope(
        thread_id="thread_1",
        namespace="root",
        run_id="run_1",
        abort_signal=abort_signal,
    )

    with execution_context(scope=scope):
        context = get_execution_context()

    assert context["abort_signal"] is abort_signal
    assert context["scope"].abort_signal is abort_signal
    assert "abort_signal" not in scope.to_dict()


def test_model_stream_response_finalizer_runs_once():
    stream = ModelStreamResponse(mode="sync")
    final_states = []

    stream.add_finalizer(final_states.append)
    stream.set_response_type("text_generation")
    stream.add("hello")
    stream.finish()
    stream.finish()

    assert len(final_states) == 1
    assert final_states[0].status == "completed"
    assert final_states[0].response_type == "text_generation"
    assert final_states[0].output == "hello"
    assert list(stream._pending_chunks) == ["hello", None]


def test_model_stream_response_finish_closes_without_public_add():
    stream = ModelStreamResponse(mode="sync")
    stream.set_response_type("text_generation")
    stream.add("hello")
    stream.add = lambda data: (_ for _ in ()).throw(AssertionError(data))

    stream.finish()

    assert stream.data == "hello"
    assert list(stream._pending_chunks) == ["hello", None]
    assert list(stream._reasoning_pending_chunks) == [None]


def test_model_stream_response_can_finish_reasoning_before_content():
    stream = ModelStreamResponse(mode="sync")
    stream.add_reasoning("thinking")

    stream.finish_reasoning()
    stream.add("answer")
    stream.finish()

    assert stream.reasoning is None
    assert stream.data == "answer"
    assert list(stream._reasoning_pending_chunks) == ["thinking", None]
    assert list(stream._pending_chunks) == ["answer", None]


def test_model_stream_response_rejects_chunks_after_channel_close():
    stream = ModelStreamResponse(mode="sync")
    stream.add_reasoning("thinking")
    stream.finish_reasoning()

    with pytest.raises(RuntimeError, match="closed stream"):
        stream.add_reasoning("late thinking")

    stream.add("answer")
    stream.finish()

    with pytest.raises(RuntimeError, match="closed stream"):
        stream.add("late answer")


def test_model_stream_response_finalizer_added_after_finish_runs_once():
    stream = ModelStreamResponse(mode="sync")
    stream.set_response_type("text_generation")
    stream.add("done")
    stream.finish()
    final_states = []

    stream.add_finalizer(final_states.append)

    assert len(final_states) == 1
    assert final_states[0].status == "completed"
    assert final_states[0].output == "done"


def test_model_stream_response_finish_with_abort_sets_interrupted_state():
    stream = ModelStreamResponse(mode="sync")
    final_states = []
    stream.add_finalizer(final_states.append)

    stream.finish(
        error=AbortRequestedError("user pressed esc"),
        status="interrupted",
    )

    assert len(final_states) == 1
    assert final_states[0].status == "interrupted"
    assert isinstance(final_states[0].error, AbortRequestedError)


def test_chat_messages_inherit_thread_context():
    with thread_context(thread_id="user_42", namespace="support"):
        chat = ChatMessages()

    assert chat.thread_id == "user_42"
    assert chat.namespace == "support"


def test_chat_messages_state_roundtrip():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    turn_id = chat.begin_turn(inputs="What is 2+2?", vars={"temperature": 0.2})
    chat.add_user("What is 2+2?")
    chat.add_assistant_response("4", reasoning_content="Simple arithmetic.")
    chat.end_turn(assistant_output="4", response_type="text_generation")

    state = chat._to_state()

    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.thread_id == "thread_1"
    assert restored.namespace == "agent:test"
    assert restored.turns[0]["turn_id"] == turn_id
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_disabled_chat_message_is_persisted_but_not_rendered():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.add_user("old context")
    chat.add_user("current context")
    chat.set_item_disabled(0, reason="compacted")

    state = chat._to_state()
    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.to_items()[0]["disabled"] is True
    assert restored.to_items()[0]["metadata"]["disabled_reason"] == "compacted"
    assert [item["content"] for item in restored.to_chatml()] == ["current context"]


def test_chat_messages_close_interrupted_tool_calls():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.append(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        }
    )

    closed = chat.close_interrupted_tool_calls(reason="user pressed interrupt")
    closed_again = chat.close_interrupted_tool_calls(reason="user pressed interrupt")

    items = chat.to_items()
    assert closed == 1
    assert closed_again == 0
    assert items[-1]["type"] == "function_call_output"
    assert items[-1]["call_id"] == "call_1"
    assert items[-1]["status"] == "interrupted"
    assert items[-1]["output"]["status"] == "interrupted"
    assert items[-1]["output"]["details"] == "user pressed interrupt"
