from unittest.mock import Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.runtime.context import ExecutionScope
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent


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


def _make_agent(checkpointer=None, **kwargs):
    return Agent(
        name="test_agent",
        model=_mock_model(),
        checkpointer=checkpointer,
        **kwargs,
    )


def test_agent_rejects_checkpointer_with_stream():
    store = InMemoryCheckpointStore()

    with pytest.raises(ValueError, match="checkpointer"):
        _make_agent(checkpointer=store, config={"stream": True})


def test_agent_saves_completed_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpointer=store)
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


def test_agent_accepts_execution_scope_for_checkpoint_identity():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpointer=store)
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
    agent = _make_agent(checkpointer=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(inputs="What is 2+2?", turn_id="run_resume")
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
    agent = _make_agent(checkpointer=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(inputs="Call flaky backend", turn_id="run_failed")
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


def test_agent_continues_latest_thread_with_new_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpointer=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(inputs="Open a support ticket", turn_id="run_1")
    chat.add_user("Open a support ticket")
    chat.add_assistant("Ticket opened.")
    chat.end_turn(
        assistant_output="Ticket opened.",
        response_type="text_generation",
        status="completed",
    )
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
    agent = _make_agent(checkpointer=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(inputs="Original", turn_id="run_done")
    chat.add_user("Original")
    chat.add_assistant("Done.")
    chat.end_turn(
        assistant_output="Done.",
        response_type="text_generation",
        status="completed",
    )
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
