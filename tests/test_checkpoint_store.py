"""Tests for CheckpointStore implementations."""

import tempfile
import time
from pathlib import Path
from threading import Thread

import pytest

from msgflux.data.stores import (
    CheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

NS = "test_agent"
SID = "session_1"
RID = "run_001"


@pytest.fixture
def mem_store():
    return InMemoryCheckpointStore()


@pytest.fixture
def sqlite_store(tmp_path):
    path = str(tmp_path / "test.sqlite3")
    store = SQLiteCheckpointStore(path=path)
    yield store
    store.close()


# ── Parametrized store fixture ───────────────────────────────────────────────


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryCheckpointStore()
    else:
        path = str(tmp_path / "test.sqlite3")
        s = SQLiteCheckpointStore(path=path)
        yield s
        s.close()


# ══════════════════════════════════════════════════════════════════════════════
# Shared tests (run on both implementations)
# ══════════════════════════════════════════════════════════════════════════════


class TestStateOperations:
    def test_save_and_load_roundtrip(self, store):
        state = {"status": "running", "step": 1, "data": {"key": "value"}}
        store.save_state(NS, SID, RID, state)
        loaded = store.load_state(NS, SID, RID)
        assert loaded == state

    def test_load_nonexistent_returns_none(self, store):
        assert store.load_state(NS, SID, "no_such_run") is None

    def test_save_state_upsert(self, store):
        store.save_state(NS, SID, RID, {"status": "running", "step": 1})
        store.save_state(NS, SID, RID, {"status": "completed", "step": 3})
        loaded = store.load_state(NS, SID, RID)
        assert loaded["status"] == "completed"
        assert loaded["step"] == 3

    def test_different_namespaces_are_isolated(self, store):
        store.save_state("agent_a", SID, RID, {"status": "a"})
        store.save_state("agent_b", SID, RID, {"status": "b"})
        assert store.load_state("agent_a", SID, RID)["status"] == "a"
        assert store.load_state("agent_b", SID, RID)["status"] == "b"

    def test_different_sessions_are_isolated(self, store):
        store.save_state(NS, "s1", RID, {"status": "s1"})
        store.save_state(NS, "s2", RID, {"status": "s2"})
        assert store.load_state(NS, "s1", RID)["status"] == "s1"
        assert store.load_state(NS, "s2", RID)["status"] == "s2"


class TestEventOperations:
    def test_append_and_load_events(self, store):
        store.save_state(NS, SID, RID, {"status": "running"})
        store.append_event(NS, SID, RID, {"event_type": "step", "step": 1})
        store.append_event(NS, SID, RID, {"event_type": "step", "step": 2})
        events = store.load_events(NS, SID, RID)
        assert len(events) == 2
        assert events[0]["step"] == 1
        assert events[1]["step"] == 2

    def test_load_events_empty(self, store):
        assert store.load_events(NS, SID, "nonexistent") == []

    def test_events_preserve_order(self, store):
        store.save_state(NS, SID, RID, {"status": "running"})
        for i in range(5):
            store.append_event(NS, SID, RID, {"event_type": "step", "n": i})
        events = store.load_events(NS, SID, RID)
        assert [e["n"] for e in events] == [0, 1, 2, 3, 4]


class TestSaveWithEvent:
    def test_atomic_save(self, store):
        state = {"status": "running", "step": 1}
        event = {"event_type": "started"}
        store.save_with_event(NS, SID, RID, state, event)
        loaded = store.load_state(NS, SID, RID)
        assert loaded["status"] == "running"
        events = store.load_events(NS, SID, RID)
        assert len(events) == 1
        assert events[0]["event_type"] == "started"


class TestListRuns:
    def test_list_runs_ordered_by_recency(self, store):
        store.save_state(NS, SID, "r1", {"status": "completed"})
        time.sleep(0.01)
        store.save_state(NS, SID, "r2", {"status": "running"})
        time.sleep(0.01)
        store.save_state(NS, SID, "r3", {"status": "completed"})
        runs = store.list_runs(NS, SID)
        assert [r["run_id"] for r in runs] == ["r3", "r2", "r1"]

    def test_list_runs_with_status_filter(self, store):
        store.save_state(NS, SID, "r1", {"status": "completed"})
        store.save_state(NS, SID, "r2", {"status": "running"})
        store.save_state(NS, SID, "r3", {"status": "completed"})
        runs = store.list_runs(NS, SID, status="completed")
        assert len(runs) == 2
        assert all(r["status"] == "completed" for r in runs)

    def test_list_runs_with_limit(self, store):
        for i in range(5):
            store.save_state(NS, SID, f"r{i}", {"status": "completed"})
            time.sleep(0.01)
        runs = store.list_runs(NS, SID, limit=2)
        assert len(runs) == 2

    def test_list_runs_empty(self, store):
        assert store.list_runs(NS, SID) == []


class TestDeleteRun:
    def test_delete_existing(self, store):
        store.save_state(NS, SID, RID, {"status": "running"})
        store.append_event(NS, SID, RID, {"event_type": "step"})
        assert store.delete_run(NS, SID, RID) is True
        assert store.load_state(NS, SID, RID) is None
        assert store.load_events(NS, SID, RID) == []

    def test_delete_nonexistent(self, store):
        assert store.delete_run(NS, SID, "no_such") is False


class TestConvenienceQueries:
    def test_load_latest_run(self, store):
        store.save_state(NS, SID, "r1", {"status": "completed", "v": 1})
        time.sleep(0.01)
        store.save_state(NS, SID, "r2", {"status": "running", "v": 2})
        latest = store.load_latest_run(NS, SID)
        assert latest["v"] == 2

    def test_load_latest_run_empty(self, store):
        assert store.load_latest_run(NS, SID) is None

    def test_find_incomplete_runs(self, store):
        store.save_state(NS, SID, "r1", {"status": "completed"})
        store.save_state(NS, SID, "r2", {"status": "running"})
        store.save_state(NS, SID, "r3", {"status": "running"})
        incomplete = store.find_incomplete_runs(NS, SID)
        assert len(incomplete) == 2

    def test_find_incomplete_excludes_failed(self, store):
        store.save_state(NS, SID, "r1", {"status": "failed"})
        store.save_state(NS, SID, "r2", {"status": "running"})
        incomplete = store.find_incomplete_runs(NS, SID)
        assert len(incomplete) == 1
        assert incomplete[0]["run_id"] == "r2"


class TestClear:
    def test_clear_all(self, store):
        store.save_state(NS, SID, "r1", {"status": "a"})
        store.save_state(NS, SID, "r2", {"status": "b"})
        removed = store.clear()
        assert removed == 2
        assert store.list_runs(NS, SID) == []

    def test_clear_by_namespace(self, store):
        store.save_state("ns1", SID, "r1", {"status": "a"})
        store.save_state("ns2", SID, "r2", {"status": "b"})
        removed = store.clear(namespace="ns1")
        assert removed == 1
        assert store.load_state("ns1", SID, "r1") is None
        assert store.load_state("ns2", SID, "r2") is not None

    def test_clear_by_session(self, store):
        store.save_state(NS, "s1", "r1", {"status": "a"})
        store.save_state(NS, "s2", "r2", {"status": "b"})
        removed = store.clear(namespace=NS, session_id="s1")
        assert removed == 1

    def test_clear_older_than(self, store):
        store.save_state(NS, SID, "old", {"status": "a"})
        time.sleep(0.1)
        store.save_state(NS, SID, "new", {"status": "b"})
        removed = store.clear(older_than=0.05)
        assert removed == 1
        assert store.load_state(NS, SID, "old") is None
        assert store.load_state(NS, SID, "new") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Memory-specific tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMemoryIsolation:
    def test_save_returns_deep_copy(self, mem_store):
        original = {"status": "running", "nested": {"key": "val"}}
        mem_store.save_state(NS, SID, RID, original)
        original["nested"]["key"] = "mutated"
        loaded = mem_store.load_state(NS, SID, RID)
        assert loaded["nested"]["key"] == "val"

    def test_load_returns_deep_copy(self, mem_store):
        mem_store.save_state(NS, SID, RID, {"status": "running", "items": [1, 2]})
        a = mem_store.load_state(NS, SID, RID)
        b = mem_store.load_state(NS, SID, RID)
        a["items"].append(3)
        assert b["items"] == [1, 2]


class TestMemoryThreadSafety:
    def test_concurrent_writes(self, mem_store):
        errors = []

        def writer(store, n):
            try:
                for i in range(50):
                    store.save_state(NS, SID, f"run_{n}_{i}", {"status": "ok", "n": n})
            except Exception as e:
                errors.append(e)

        threads = [Thread(target=writer, args=(mem_store, n)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        runs = mem_store.list_runs(NS, SID)
        assert len(runs) == 200


# ══════════════════════════════════════════════════════════════════════════════
# SQLite-specific tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSQLiteWAL:
    def test_wal_mode_active(self, sqlite_store):
        row = sqlite_store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"


class TestSQLiteForeignKeys:
    def test_cascade_delete_events(self, sqlite_store):
        sqlite_store.save_state(NS, SID, RID, {"status": "running"})
        sqlite_store.append_event(NS, SID, RID, {"event_type": "step"})
        sqlite_store.append_event(NS, SID, RID, {"event_type": "done"})
        sqlite_store.delete_run(NS, SID, RID)
        events = sqlite_store.load_events(NS, SID, RID)
        assert events == []


class TestSQLitePersistence:
    def test_persists_across_reopen(self, tmp_path):
        path = str(tmp_path / "persist.sqlite3")
        store = SQLiteCheckpointStore(path=path)
        store.save_state(NS, SID, RID, {"status": "saved"})
        store.close()

        store2 = SQLiteCheckpointStore(path=path)
        loaded = store2.load_state(NS, SID, RID)
        store2.close()
        assert loaded["status"] == "saved"


class TestSQLiteUpsert:
    def test_upsert_updates_not_duplicates(self, sqlite_store):
        sqlite_store.save_state(NS, SID, RID, {"status": "v1"})
        sqlite_store.save_state(NS, SID, RID, {"status": "v2"})
        runs = sqlite_store.list_runs(NS, SID)
        assert len(runs) == 1
        assert runs[0]["status"] == "v2"


# ══════════════════════════════════════════════════════════════════════════════
# Integration: ChatMessages round-trip
# ══════════════════════════════════════════════════════════════════════════════


class TestChatMessagesIntegration:
    def test_chat_messages_state_roundtrip(self, store):
        from msgflux.chat_messages import ChatMessages

        chat = ChatMessages(session_id="s1")
        chat.add_user("Hello")
        chat.add_assistant("Hi there!")

        state = {
            "status": "completed",
            "messages": chat._to_state(),
            "vars": {"temperature": 0.7},
        }
        store.save_state(NS, SID, RID, state)

        loaded = store.load_state(NS, SID, RID)
        restored = ChatMessages()
        restored._hydrate_state(loaded["messages"])

        assert len(restored) == 2
        chatml = restored.to_chatml()
        assert chatml[0]["role"] == "user"
        assert chatml[0]["content"] == "Hello"
        assert chatml[1]["role"] == "assistant"
        assert chatml[1]["content"] == "Hi there!"
        assert restored.session_id == "s1"

    def test_chat_messages_with_turns(self, store):
        from msgflux.chat_messages import ChatMessages

        chat = ChatMessages(session_id="demo")
        chat.begin_turn(inputs="What is 2+2?")
        chat.add_user("What is 2+2?")
        chat.add_assistant("4")
        chat.end_turn(assistant_output="4")

        state = {"status": "completed", "messages": chat._to_state()}
        store.save_state(NS, SID, RID, state)

        loaded = store.load_state(NS, SID, RID)
        restored = ChatMessages()
        restored._hydrate_state(loaded["messages"])

        assert len(restored.turns) == 1
        assert restored.turns[0]["assistant_output"] == "4"

    def test_time_travel_with_multiple_runs(self, store):
        from msgflux.chat_messages import ChatMessages

        # Run 1: completed
        chat1 = ChatMessages()
        chat1.add_user("Run 1 question")
        chat1.add_assistant("Run 1 answer")
        store.save_state(
            NS,
            SID,
            "r1",
            {
                "status": "completed",
                "messages": chat1._to_state(),
            },
        )
        time.sleep(0.01)

        # Run 2: failed
        chat2 = ChatMessages()
        chat2.add_user("Run 2 question")
        store.save_state(
            NS,
            SID,
            "r2",
            {
                "status": "failed",
                "messages": chat2._to_state(),
            },
        )
        time.sleep(0.01)

        # Run 3: completed
        chat3 = ChatMessages()
        chat3.add_user("Run 3 question")
        chat3.add_assistant("Run 3 answer")
        store.save_state(
            NS,
            SID,
            "r3",
            {
                "status": "completed",
                "messages": chat3._to_state(),
            },
        )

        # Time-travel: load run 1
        old_state = store.load_state(NS, SID, "r1")
        restored = ChatMessages()
        restored._hydrate_state(old_state["messages"])
        chatml = restored.to_chatml()
        assert chatml[0]["content"] == "Run 1 question"

        # Latest run is r3
        latest = store.load_latest_run(NS, SID)
        assert latest["status"] == "completed"
        r3_chat = ChatMessages()
        r3_chat._hydrate_state(latest["messages"])
        assert r3_chat.to_chatml()[0]["content"] == "Run 3 question"


# ══════════════════════════════════════════════════════════════════════════════
# ABC contract
# ══════════════════════════════════════════════════════════════════════════════


class TestABCContract:
    def test_is_checkpoint_store(self, store):
        assert isinstance(store, CheckpointStore)
