"""Tests for Agent durable execution (resume from checkpoint)."""

import time

import pytest
from unittest.mock import Mock

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


def _text_response(text="Hello!"):
    resp = Mock(spec=ModelResponse)
    resp.response_type = "text_generation"
    resp.data = text
    resp.metadata = {"model": "test"}
    resp.consume.return_value = text
    return resp


def _make_agent(checkpointer=None, **kw):
    return Agent(
        name="test_agent", model=_mock_model(), checkpointer=checkpointer, **kw
    )


# ── Initialization ───────────────────────────────────────────────────────────


class TestDurableInit:
    def test_checkpointer_stored(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)
        assert agent.checkpointer is store

    def test_no_checkpointer_by_default(self):
        agent = _make_agent()
        assert agent.checkpointer is None

    def test_stream_and_checkpointer_raises(self):
        store = InMemoryCheckpointStore()
        with pytest.raises(ValueError, match="checkpointer.*stream"):
            _make_agent(checkpointer=store, config={"stream": True})


# ── Resume from checkpoint ───────────────────────────────────────────────────


class TestResumeFromCheckpoint:
    def test_no_resume_when_no_checkpointer(self):
        agent = _make_agent()
        result = agent._try_resume_from_checkpoint(None)
        assert result is None

    def test_no_resume_when_no_incomplete_runs(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)
        chat = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(chat)
        assert result is None

    def test_resume_rehydrates_messages(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Simulate a previous incomplete run
        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="What is 2+2?")
        chat.add_user("What is 2+2?")

        store.save_state(
            "test_agent",
            "s1",
            chat.turns[0]["turn_id"],
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {"temperature": 0.7},
            },
        )

        # Try resume with a fresh ChatMessages (same session)
        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)

        assert result is not None
        restored = result["messages"]
        assert isinstance(restored, ChatMessages)
        # ChatML only has user message (turn markers are internal)
        chatml = restored.to_chatml()
        assert len(chatml) == 1
        assert chatml[0]["content"] == "What is 2+2?"
        assert result["vars"] == {"temperature": 0.7}

    def test_resume_uses_default_session_when_no_chat_messages(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="default")
        chat.begin_turn(inputs="test")
        chat.add_user("test")

        store.save_state(
            "test_agent",
            "default",
            chat.turns[0]["turn_id"],
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {},
            },
        )

        # Pass None (no messages kwarg)
        result = agent._try_resume_from_checkpoint(None)
        assert result is not None

    def test_resume_skips_completed_runs(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        store.save_state(
            "test_agent",
            "s1",
            "run_done",
            {
                "status": "completed",
                "messages": ChatMessages(session_id="s1")._to_state(),
                "vars": {},
            },
        )

        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)
        assert result is None

    def test_resume_skips_failed_runs(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        store.save_state(
            "test_agent",
            "s1",
            "run_fail",
            {
                "status": "failed",
                "messages": ChatMessages(session_id="s1")._to_state(),
                "vars": {},
            },
        )

        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)
        assert result is None

    def test_resume_picks_most_recent_incomplete(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Old incomplete
        chat_old = ChatMessages(session_id="s1")
        chat_old.begin_turn(inputs="old")
        chat_old.add_user("old")
        store.save_state(
            "test_agent",
            "s1",
            "run_old",
            {
                "status": "running",
                "messages": chat_old._to_state(),
                "vars": {},
            },
        )
        time.sleep(0.01)

        # New incomplete
        chat_new = ChatMessages(session_id="s1")
        chat_new.begin_turn(inputs="new")
        chat_new.add_user("new")
        store.save_state(
            "test_agent",
            "s1",
            "run_new",
            {
                "status": "running",
                "messages": chat_new._to_state(),
                "vars": {},
            },
        )

        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)
        assert result is not None
        restored = result["messages"]
        assert restored.to_chatml()[0]["content"] == "new"


# ── Error checkpointing ─────────────────────────────────────────────────────


class TestErrorCheckpointing:
    def test_error_saves_failed_status(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="fail me")
        chat.add_user("fail me")

        inputs = {"messages": chat, "vars": {"k": "v"}}
        agent._checkpoint_save_on_error(inputs)

        runs = store.list_runs("test_agent", "s1")
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"

    def test_error_checkpoint_no_op_without_checkpointer(self):
        agent = _make_agent()
        # Should not raise
        agent._checkpoint_save_on_error({"messages": [], "vars": {}})

    def test_forward_saves_failed_on_model_error(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Mock generator.forward to raise
        agent.generator.forward = Mock(side_effect=RuntimeError("API down"))

        chat = ChatMessages(session_id="s1")

        with pytest.raises(RuntimeError, match="API down"):
            agent("test question", messages=chat)

        runs = store.list_runs("test_agent", "s1")
        assert len(runs) == 1
        assert runs[0]["status"] == "failed"


# ── Forward with durable ────────────────────────────────────────────────────


class TestForwardDurable:
    def test_forward_normal_flow_with_checkpointer(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        resp = _text_response("42")
        agent.generator.forward = Mock(return_value=resp)

        chat = ChatMessages(session_id="s1")
        result = agent("What is 6*7?", messages=chat)

        assert result == "42"
        # Should have a completed checkpoint
        runs = store.list_runs("test_agent", "s1")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"

    def test_forward_resumes_incomplete_run(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Simulate crashed run: messages with user question, status=running
        chat_crashed = ChatMessages(session_id="s1")
        chat_crashed.begin_turn(inputs="What is 2+2?")
        chat_crashed.add_user("What is 2+2?")
        turn_id = chat_crashed.turns[0]["turn_id"]

        store.save_state(
            "test_agent",
            "s1",
            turn_id,
            {
                "status": "running",
                "messages": chat_crashed._to_state(),
                "vars": {},
            },
        )

        # Agent should detect and resume
        resp = _text_response("4")
        agent.generator.forward = Mock(return_value=resp)

        fresh_chat = ChatMessages(session_id="s1")
        result = agent("ignored because resuming", messages=fresh_chat)

        assert result == "4"

        # Checkpoint should be updated to completed
        runs = store.list_runs("test_agent", "s1")
        # The completed run should exist
        completed = [r for r in runs if r["status"] == "completed"]
        assert len(completed) >= 1

    def test_forward_no_resume_for_different_session(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Incomplete run for session "s1"
        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="hello")
        chat.add_user("hello")
        store.save_state(
            "test_agent",
            "s1",
            "run1",
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {},
            },
        )

        # Call with session "s2" -> should not resume
        resp = _text_response("Hi!")
        agent.generator.forward = Mock(return_value=resp)

        chat_s2 = ChatMessages(session_id="s2")
        result = agent("Hi there", messages=chat_s2)
        assert result == "Hi!"

    def test_forward_without_checkpointer_unchanged(self):
        agent = _make_agent()  # no checkpointer
        resp = _text_response("works")
        agent.generator.forward = Mock(return_value=resp)

        result = agent("test")
        assert result == "works"


# ── Async resume ─────────────────────────────────────────────────────────────


class TestAsyncResume:
    @pytest.mark.asyncio
    async def test_async_resume_rehydrates(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Incomplete run
        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="async question")
        chat.add_user("async question")
        store.save_state(
            "test_agent",
            "s1",
            chat.turns[0]["turn_id"],
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {"mode": "async"},
            },
        )

        result = await agent._atry_resume_from_checkpoint(ChatMessages(session_id="s1"))
        assert result is not None
        assert result["vars"] == {"mode": "async"}

    @pytest.mark.asyncio
    async def test_async_no_resume_when_empty(self):
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)
        result = await agent._atry_resume_from_checkpoint(ChatMessages(session_id="s1"))
        assert result is None


# ── Tool call checkpoint cycle ───────────────────────────────────────────────


class TestToolCallCheckpointCycle:
    def test_checkpoint_save_called_with_running(self):
        """Verify _checkpoint_save is called with running status during tool loop."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="step 1")
        chat.add_user("step 1")
        # Simulate tool response appended
        chat.extend(
            [
                {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        )

        agent._checkpoint_save(chat, {"k": "v"}, status="running")

        runs = store.list_runs("test_agent", "s1")
        assert len(runs) == 1
        assert runs[0]["status"] == "running"

        # Save completed
        agent._checkpoint_save(chat, {"k": "v"}, status="completed")
        runs = store.list_runs("test_agent", "s1")
        assert runs[0]["status"] == "completed"

    def test_checkpoint_preserves_tool_messages(self):
        """Verify that checkpointed messages include tool call history."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="task")
        chat.add_user("Do something")
        chat.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "t", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
            ]
        )

        agent._checkpoint_save(chat, {}, status="running")

        # Load and rehydrate
        turn_id = chat.turns[0]["turn_id"]
        state = store.load_state("test_agent", "s1", turn_id)
        restored = ChatMessages()
        restored._hydrate_state(state["messages"])

        chatml = restored.to_chatml()
        # user + assistant (tool_calls) + tool
        assert len(chatml) == 3
        assert chatml[2]["role"] == "tool"
        assert chatml[2]["content"] == "tool result"


# ── Namespace isolation ──────────────────────────────────────────────────────


class TestNamespaceIsolation:
    def test_different_agents_dont_resume_each_other(self):
        store = InMemoryCheckpointStore()
        agent_a = Agent(name="agent_a", model=_mock_model(), checkpointer=store)
        agent_b = Agent(name="agent_b", model=_mock_model(), checkpointer=store)

        # Incomplete run for agent_a
        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="hello")
        chat.add_user("hello")
        store.save_state(
            "agent_a",
            "s1",
            "run1",
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {},
            },
        )

        # agent_b should not find it
        result = agent_b._try_resume_from_checkpoint(ChatMessages(session_id="s1"))
        assert result is None

        # agent_a should find it
        result = agent_a._try_resume_from_checkpoint(ChatMessages(session_id="s1"))
        assert result is not None


# ── SQLite integration ───────────────────────────────────────────────────────


class TestSQLiteIntegration:
    def test_resume_with_sqlite(self, tmp_path):
        from msgflux.data.stores import SQLiteCheckpointStore

        path = str(tmp_path / "durable.sqlite3")
        store = SQLiteCheckpointStore(path=path)

        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="persisted")
        chat.add_user("persisted")

        store.save_state(
            "test_agent",
            "s1",
            chat.turns[0]["turn_id"],
            {
                "status": "running",
                "messages": chat._to_state(),
                "vars": {"db": "sqlite"},
            },
        )
        store.close()

        # Reopen — simulates process restart
        store2 = SQLiteCheckpointStore(path=path)
        agent2 = _make_agent(checkpointer=store2)

        result = agent2._try_resume_from_checkpoint(ChatMessages(session_id="s1"))
        assert result is not None
        assert result["vars"] == {"db": "sqlite"}

        restored = result["messages"]
        assert restored.to_chatml()[0]["content"] == "persisted"
        store2.close()

    def test_crash_and_resume_e2e(self, tmp_path):
        """Simulate crash mid-execution and resume with SQLite."""
        from msgflux.data.stores import SQLiteCheckpointStore

        path = str(tmp_path / "crash.sqlite3")
        store = SQLiteCheckpointStore(path=path)

        # First execution: agent saves running checkpoint, then crashes
        agent1 = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="user_42")
        chat.begin_turn(inputs="complex task")
        chat.add_user("complex task")

        # Manually save running checkpoint (simulates checkpoint after tool call)
        agent1._checkpoint_save(chat, {}, status="running")

        store.close()

        # Process restart
        store2 = SQLiteCheckpointStore(path=path)
        agent2 = _make_agent(checkpointer=store2)

        resp = _text_response("Resumed!")
        agent2.generator.forward = Mock(return_value=resp)

        fresh_chat = ChatMessages(session_id="user_42")
        result = agent2("this message ignored", messages=fresh_chat)
        assert result == "Resumed!"

        # Check completed
        runs = store2.list_runs("test_agent", "user_42")
        completed = [r for r in runs if r["status"] == "completed"]
        assert len(completed) >= 1

        store2.close()


# ── Recovery scenarios (additional coverage) ─────────────────────────────────


class TestRecoveryScenarios:
    def test_failed_run_does_not_block_new_execution(self):
        """After a failed run, a new call with the same session starts fresh.

        Note: fresh ChatMessages produces the same turn_id ("turn_1"), so
        the completed run UPSERTS over the failed one — only one run remains.
        """
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # First call fails
        agent.generator.forward = Mock(side_effect=RuntimeError("API error"))
        chat1 = ChatMessages(session_id="s1")
        with pytest.raises(RuntimeError):
            agent("question 1", messages=chat1)

        # Verify failed run exists
        runs = store.list_runs("test_agent", "s1")
        assert runs[0]["status"] == "failed"

        # Second call should work (fresh execution, not resume)
        resp = _text_response("works now")
        agent.generator.forward = Mock(return_value=resp)
        chat2 = ChatMessages(session_id="s1")
        result = agent("question 2", messages=chat2)
        assert result == "works now"

        # UPSERT: same turn_id ("turn_1") overwrites the failed run
        runs = store.list_runs("test_agent", "s1")
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"

    def test_multi_turn_crash_resumes_correct_turn(self):
        """Turn 1 completes, turn 2 crashes → resume picks turn 2."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Turn 1: completes normally
        resp1 = _text_response("turn 1 answer")
        agent.generator.forward = Mock(return_value=resp1)
        chat = ChatMessages(session_id="s1")
        result1 = agent("turn 1 question", messages=chat)
        assert result1 == "turn 1 answer"

        # Turn 2: simulate crash (save running checkpoint manually)
        chat.begin_turn(inputs="turn 2 question")
        chat.add_user("turn 2 question")
        agent._checkpoint_save(chat, {"turn": 2}, status="running")

        # Verify: turn 1 = completed, turn 2 = running
        runs = store.list_runs("test_agent", "s1")
        statuses = {r["run_id"]: r["status"] for r in runs}
        assert any(s == "completed" for s in statuses.values())
        assert any(s == "running" for s in statuses.values())

        # New agent instance (simulates restart) should resume turn 2
        agent2 = _make_agent(checkpointer=store)
        resp2 = _text_response("turn 2 resumed")
        agent2.generator.forward = Mock(return_value=resp2)

        fresh_chat = ChatMessages(session_id="s1")
        result2 = agent2("ignored", messages=fresh_chat)
        assert result2 == "turn 2 resumed"

    def test_resumed_messages_contain_full_history(self):
        """Resumed ChatMessages include all items from the crashed run."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Build a chat with user + assistant(tool_call) + tool result
        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="What is 2+2?")
        chat.add_user("What is 2+2?")
        chat.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calc", "arguments": '{"a":2,"b":2}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            ]
        )

        # Save as running (mid-tool-loop checkpoint)
        agent._checkpoint_save(chat, {}, status="running")

        # Resume and verify history is complete
        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)
        assert result is not None

        restored = result["messages"]
        chatml = restored.to_chatml()

        # Should have: user, assistant(tool_call), tool
        assert len(chatml) == 3
        assert chatml[0]["role"] == "user"
        assert chatml[0]["content"] == "What is 2+2?"
        assert chatml[1]["role"] == "assistant"
        assert "tool_calls" in chatml[1]
        assert chatml[2]["role"] == "tool"
        assert chatml[2]["content"] == "4"

    def test_vars_preserved_through_resume(self):
        """Vars from checkpoint are passed to resumed execution."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        chat = ChatMessages(session_id="s1")
        chat.begin_turn(inputs="test")
        chat.add_user("test")

        # Save with specific vars
        agent._checkpoint_save(
            chat,
            {"temperature": 0.5, "custom_key": "custom_value"},
            status="running",
        )

        # Resume
        fresh = ChatMessages(session_id="s1")
        result = agent._try_resume_from_checkpoint(fresh)
        assert result is not None
        assert result["vars"]["temperature"] == 0.5
        assert result["vars"]["custom_key"] == "custom_value"

    def test_checkpoint_error_does_not_mask_original_error(self):
        """If checkpoint save on error also fails, original error propagates."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Make model fail
        agent.generator.forward = Mock(side_effect=RuntimeError("Model crashed"))

        # Make checkpoint save also fail (broken store)
        store.save_state = Mock(side_effect=Exception("Store broken"))

        chat = ChatMessages(session_id="s1")
        # The original RuntimeError should propagate, not the store error
        with pytest.raises(RuntimeError, match="Model crashed"):
            agent("test", messages=chat)

    def test_concurrent_sessions_isolate_checkpoints(self):
        """Concurrent sessions under different IDs don't interfere."""
        store = InMemoryCheckpointStore()
        agent = _make_agent(checkpointer=store)

        # Session A: running (incomplete)
        chat_a = ChatMessages(session_id="session_a")
        chat_a.begin_turn(inputs="task A")
        chat_a.add_user("task A")
        agent._checkpoint_save(chat_a, {"session": "a"}, status="running")

        # Session B: completed
        resp_b = _text_response("B done")
        agent.generator.forward = Mock(return_value=resp_b)
        chat_b = ChatMessages(session_id="session_b")
        agent("task B", messages=chat_b)

        # Session B should NOT resume session A
        runs_a = store.list_runs("test_agent", "session_a")
        runs_b = store.list_runs("test_agent", "session_b")
        assert any(r["status"] == "running" for r in runs_a)
        assert all(r["status"] == "completed" for r in runs_b)

        # Session A resume still works
        result = agent._try_resume_from_checkpoint(ChatMessages(session_id="session_a"))
        assert result is not None
        assert result["vars"]["session"] == "a"

        # Session B has nothing to resume
        result = agent._try_resume_from_checkpoint(ChatMessages(session_id="session_b"))
        assert result is None
