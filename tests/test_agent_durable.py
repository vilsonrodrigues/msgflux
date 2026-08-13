from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import AbortRequestedError, TaskInterruptRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime import AbortSignal
from msgflux.runtime.context import ExecutionScope


def _mock_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


def _text_response(text="Hello!"):
    response = Mock(spec=ModelResponse)
    response.response_type = "text_generation"
    response.data = text
    response.reasoning = None
    response.metadata = {"model": "test"}
    response.consume.return_value = text
    return response


def _tool_call_response():
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "call_lookup", "lookup", '{"query":"status"}')
    response = Mock(spec=ModelResponse)
    response.response_type = "tool_call"
    response.data = tool_calls
    response.reasoning = None
    response.metadata = {"model": "test"}
    return response


def _make_agent(checkpoint_store=None, **kwargs):
    return Agent(
        name="test_agent",
        model=_mock_model(),
        checkpoint_store=checkpoint_store,
        **kwargs,
    )


def test_agent_accepts_checkpoint_store_with_stream():
    store = InMemoryCheckpointStore()

    agent = _make_agent(checkpoint_store=store, config={"stream": True})

    assert agent.checkpoint_store is store


def test_agent_saves_completed_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(return_value=_text_response("42"))

    scope = ExecutionScope(
        thread_id="user_42", namespace="test_agent", run_id="run_math"
    )
    result = agent("What is 6*7?", scope=scope)

    assert result == "42"
    state = store.load_state("test_agent", "user_42", "run_math")
    assert state is not None
    assert state["status"] == "completed"
    assert state["messages"]["thread_id"] == "user_42"
    assert "turns" not in state["messages"]
    assert "vars" not in state
    assert [
        item
        for item in state["messages"]["items"]
        if item.get("role") == "assistant" and item.get("content") == "42"
    ] == [{"type": "message", "role": "assistant", "content": "42"}]


def test_agent_accepts_execution_scope_for_checkpoint_identity():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(return_value=_text_response("scoped"))

    scope = ExecutionScope(thread_id="user_42", namespace="outer", run_id="run_scope")
    result = agent("Use the scoped identity", scope=scope)

    assert result == "scoped"
    state = store.load_state("test_agent", "user_42", "run_scope")
    assert state is not None
    assert state["status"] == "completed"
    assert state["messages"]["namespace"] == "test_agent"


def test_agent_resumes_exact_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_resume")
    chat.add_user("What is 2+2?")
    store.save_state(
        "test_agent",
        "user_42",
        "run_resume",
        {
            "status": "running",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("4"))
    result = agent(
        "this input should be ignored on resume",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_resume",
        ),
    )

    assert result == "4"
    state = store.load_state("test_agent", "user_42", "run_resume")
    assert state is not None
    assert state["status"] == "completed"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_agent_resumes_failed_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_failed")
    chat.add_user("Call flaky backend")
    store.save_state(
        "test_agent",
        "user_42",
        "run_failed",
        {
            "status": "failed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("recovered"))
    result = agent(
        "this retry input should be ignored on resume",
        vars={"attempt": 2},
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_failed",
        ),
    )

    assert result == "recovered"
    state = store.load_state("test_agent", "user_42", "run_failed")
    assert state is not None
    assert state["status"] == "completed"
    assert "vars" not in state

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[0]["content"] == "Call flaky backend"
    assert restored.to_chatml()[-1]["content"] == "recovered"


def test_agent_interrupt_closes_active_tool_calls_in_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        raise TaskInterruptRequestedError("task_1", f"user pressed interrupt: {query}")

    agent.tool_library.add(lookup)
    agent.generator.forward = Mock(return_value=_tool_call_response())

    with pytest.raises(TaskInterruptRequestedError):
        agent(
            "Check status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_interrupt",
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_interrupt")
    assert state is not None
    assert state["status"] == "interrupted"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert any(
        item.get("role") == "assistant"
        and item.get("tool_calls", [{}])[0].get("id") == "call_lookup"
        for item in chatml
    )
    interrupted_outputs = [
        item
        for item in chatml
        if item.get("role") == "tool" and item.get("tool_call_id") == "call_lookup"
    ]
    assert len(interrupted_outputs) == 1
    assert "interrupted" in interrupted_outputs[0]["content"]
    assert restored.turns[-1]["status"] == "interrupted"


def test_agent_persists_tool_reasoning_once_in_canonical_history():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        return f"status for {query}"

    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.reasoning = "Check the current status."
    tool_response.history_items = [
        {
            "type": "reasoning",
            "role": "assistant",
            "text": "Check the current status.",
            "provider_state": {
                "provider": "openrouter",
                "data": [{"type": "reasoning.text", "text": "opaque order"}],
            },
        }
    ]
    agent.generator.forward = Mock(
        side_effect=[tool_response, _text_response("All systems operational.")]
    )

    result = agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_tool_reasoning",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_tool_reasoning")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    reasoning_items = [item for item in restored if item.get("type") == "reasoning"]

    assert result == "All systems operational."
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["text"] == "Check the current status."
    assert all("<think>" not in str(item.get("content", "")) for item in restored)


def test_agent_preserves_responses_function_call_without_duplicate():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        return f"status for {query}"

    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.history_items = [
        {
            "type": "function_call",
            "id": "fc_lookup",
            "status": "completed",
            "call_id": "call_lookup",
            "name": "lookup",
            "arguments": '{"query":"status"}',
        }
    ]
    agent.generator.forward = Mock(
        side_effect=[tool_response, _text_response("All systems operational.")]
    )

    result = agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_responses_tool",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_responses_tool")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    calls = [item for item in restored if item.get("type") == "function_call"]
    outputs = [item for item in restored if item.get("type") == "function_call_output"]

    assert result == "All systems operational."
    assert calls == [tool_response.history_items[0]]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_lookup"


def test_agent_preserves_responses_message_without_synthetic_duplicate():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("Final answer")
    response.history_items = [
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Final answer"}],
            "provider_state": {
                "provider": "openai",
                "api_mode": "responses",
                "data": {"id": "msg_1", "status": "completed"},
            },
        }
    ]
    agent.generator.forward = Mock(return_value=response)

    agent(
        "Answer once",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_message_once",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_message_once")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assistant_messages = [
        item
        for item in restored
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    assert assistant_messages == response.history_items


def test_agent_abort_signal_saves_interrupted_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(side_effect=AbortRequestedError("user pressed esc"))
    abort_signal = AbortSignal()
    abort_signal.abort("user pressed esc")

    with pytest.raises(TaskInterruptRequestedError, match="user pressed esc"):
        agent(
            "Check status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_abort",
                abort_signal=abort_signal,
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_abort")
    assert state is not None
    assert state["status"] == "interrupted"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.turns[-1]["status"] == "interrupted"


def test_agent_stream_checkpoint_completes_when_stream_finishes():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    stream_response.reasoning = "stream reasoning"
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_stream",
        ),
    )

    assert result is stream_response
    streaming_state = store.load_state("test_agent", "user_42", "run_stream")
    assert streaming_state is not None
    assert streaming_state["status"] == "streaming"

    stream_response.add("hello")
    stream_response.add(" world")
    stream_response.finish()

    completed_state = store.load_state("test_agent", "user_42", "run_stream")
    assert completed_state is not None
    assert completed_state["status"] == "completed"

    restored = ChatMessages()
    restored._hydrate_state(completed_state["messages"])
    chatml = restored.to_chatml()
    assert chatml[-1]["content"] == "hello world"
    assert restored.turns[-1]["status"] == "completed"
    assert "assistant_output" not in restored.turns[-1]
    assert [item["type"] for item in restored if item.get("type") == "reasoning"] == [
        "reasoning"
    ]
    assert (
        sum(
            item.get("role") == "assistant" and item.get("content") == "hello world"
            for item in restored
        )
        == 1
    )


def test_agent_stream_updates_the_caller_chat_messages_when_finished():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")

    result = agent(
        "Stream status",
        messages=messages,
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_visible_stream",
        ),
    )
    result.add("visible")
    result.finish()

    assert messages.to_chatml()[-1]["content"] == "visible"
    assert messages.turns[-1]["status"] == "completed"


def test_agent_stream_finalizer_saves_sqlite_checkpoint_from_worker(tmp_path):
    store = SQLiteCheckpointStore(path=str(tmp_path / "checkpoints.sqlite3"))
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream to SQLite",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_sqlite_stream",
        ),
    )

    def finish_from_worker():
        result.add("persisted")
        result.finish()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(finish_from_worker).result()

    state = store.load_state("test_agent", "user_42", "run_sqlite_stream")
    assert state["status"] == "completed"
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[-1]["content"] == "persisted"
    store.close()


def test_agent_stream_checkpoint_preserves_partial_output_on_failure():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_stream_failed",
        ),
    )

    assert result is stream_response
    stream_response.add("partial")
    stream_response.finish(error=RuntimeError("stream disconnected"), status="failed")

    state = store.load_state("test_agent", "user_42", "run_stream_failed")
    assert state is not None
    assert state["status"] == "failed"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert chatml[-1]["content"] == "partial"
    assert restored.turns[-1]["status"] == "failed"
    assert "assistant_output" not in restored.turns[-1]


def test_agent_stream_checkpoint_marks_pre_output_abort_interrupted():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.finish(
        error=AbortRequestedError("user pressed esc"),
        status="interrupted",
    )
    agent.generator.forward = Mock(return_value=stream_response)

    with pytest.raises(TaskInterruptRequestedError, match="user pressed esc"):
        agent(
            "Stream status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_stream_abort",
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_stream_abort")
    assert state is not None
    assert state["status"] == "interrupted"


def test_agent_continues_latest_thread_with_new_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_1")
    chat.add_user("Open a support ticket")
    chat.add_assistant("Ticket opened.")
    chat.end_turn()
    store.save_state(
        "test_agent",
        "user_42",
        "run_1",
        {
            "status": "completed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("Added note."))
    result = agent(
        "Add note: customer called back",
        vars={"customer": "current"},
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_2",
        ),
    )

    assert result == "Added note."
    state = store.load_state("test_agent", "user_42", "run_2")
    assert state is not None
    assert state["status"] == "completed"
    assert "vars" not in state

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert [item["content"] for item in chatml] == [
        "Open a support ticket",
        "Ticket opened.",
        "<task>Add note: customer called back</task>",
        "Added note.",
    ]


def test_agent_rejects_completed_run_id_retry():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_done")
    chat.add_user("Original")
    chat.add_assistant("Done.")
    chat.end_turn()
    store.save_state(
        "test_agent",
        "user_42",
        "run_done",
        {
            "status": "completed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("Fresh result."))
    with pytest.raises(ValueError, match="Use a new run_id"):
        agent(
            "Fresh input",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_done",
            ),
        )

    agent.generator.forward.assert_not_called()
    state = store.load_state("test_agent", "user_42", "run_done")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert [item["content"] for item in chatml] == [
        "Original",
        "Done.",
    ]
