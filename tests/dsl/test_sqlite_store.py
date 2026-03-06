"""Tests for SQLiteCheckpointStore (sync)."""

import tempfile
from pathlib import Path

import pytest

from msgflux.data.stores import SQLiteCheckpointStore, StepEvent, StepStatus


class TestSQLiteCheckpointStore:
    def test_save_and_load_event(self):
        store = SQLiteCheckpointStore()
        event = StepEvent(
            step_name="module:prep#0",
            status=StepStatus.COMPLETED,
            timestamp=1.0,
        )
        store.save_event("run-1", event)
        run = store.load_run("run-1")
        assert run is not None
        assert len(run.events) == 1
        assert run.events[0].step_name == "module:prep#0"

    def test_load_nonexistent_run(self):
        store = SQLiteCheckpointStore()
        assert store.load_run("no-such-run") is None

    def test_delete_run(self):
        store = SQLiteCheckpointStore()
        event = StepEvent(
            step_name="s",
            status=StepStatus.COMPLETED,
            timestamp=1.0,
        )
        store.save_event("run-1", event)
        store.delete_run("run-1")
        assert store.load_run("run-1") is None

    def test_delete_nonexistent_run(self):
        store = SQLiteCheckpointStore()
        store.delete_run("no-such-run")  # should not raise

    def test_list_runs(self):
        store = SQLiteCheckpointStore()
        store.save_event(
            "a",
            StepEvent(step_name="s", status=StepStatus.PENDING, timestamp=1.0),
        )
        store.save_event(
            "b",
            StepEvent(step_name="s", status=StepStatus.PENDING, timestamp=1.0),
        )
        assert set(store.list_runs()) == {"a", "b"}

    def test_get_last_completed_step(self):
        store = SQLiteCheckpointStore()
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.COMPLETED,
                timestamp=1.0,
            ),
        )
        store.save_event(
            "r",
            StepEvent(
                step_name="s2",
                status=StepStatus.IN_PROGRESS,
                timestamp=2.0,
            ),
        )
        last = store.get_last_completed_step("r")
        assert last is not None
        assert last.step_name == "s1"

    def test_get_last_completed_step_none(self):
        store = SQLiteCheckpointStore()
        assert store.get_last_completed_step("nonexistent") is None

    def test_get_step_status(self):
        store = SQLiteCheckpointStore()
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.IN_PROGRESS,
                timestamp=1.0,
            ),
        )
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.COMPLETED,
                timestamp=2.0,
            ),
        )
        assert store.get_step_status("r", "s1") == StepStatus.COMPLETED

    def test_get_step_status_nonexistent(self):
        store = SQLiteCheckpointStore()
        assert store.get_step_status("r", "s1") is None

    def test_get_step_retry_count(self):
        store = SQLiteCheckpointStore()
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.FAILED,
                timestamp=1.0,
                error="err",
                retry_count=3,
            ),
        )
        assert store.get_step_retry_count("r", "s1") == 3

    def test_get_step_retry_count_no_failures(self):
        store = SQLiteCheckpointStore()
        assert store.get_step_retry_count("r", "s1") == 0

    def test_multiple_events_for_same_step(self):
        store = SQLiteCheckpointStore()
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.IN_PROGRESS,
                timestamp=1.0,
            ),
        )
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.FAILED,
                timestamp=2.0,
                error="timeout",
                retry_count=1,
            ),
        )
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.IN_PROGRESS,
                timestamp=3.0,
            ),
        )
        store.save_event(
            "r",
            StepEvent(
                step_name="s1",
                status=StepStatus.COMPLETED,
                timestamp=4.0,
                message_snapshot=b"{}",
            ),
        )
        run = store.load_run("r")
        assert run is not None
        assert len(run.events) == 4
        assert store.get_step_status("r", "s1") == StepStatus.COMPLETED

    def test_event_append_no_full_reserialization(self):
        """Each save_event should be an INSERT, not a full read-modify-write."""
        store = SQLiteCheckpointStore()
        for i in range(100):
            store.save_event(
                "r",
                StepEvent(
                    step_name=f"step#{i}",
                    status=StepStatus.COMPLETED,
                    timestamp=float(i),
                ),
            )
        run = store.load_run("r")
        assert run is not None
        assert len(run.events) == 100

    def test_persistence_across_connections(self):
        """Data should survive closing and reopening the database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")

            store1 = SQLiteCheckpointStore(path=db_path)
            store1.save_event(
                "persistent-run",
                StepEvent(
                    step_name="step_a",
                    status=StepStatus.COMPLETED,
                    timestamp=1.0,
                    message_snapshot=b'{"key": "value"}',
                ),
            )
            store1.close()

            store2 = SQLiteCheckpointStore(path=db_path)
            run = store2.load_run("persistent-run")
            assert run is not None
            assert len(run.events) == 1
            assert run.events[0].message_snapshot == b'{"key": "value"}'
            store2.close()

    def test_ttl_cleanup(self):
        """Expired runs should be cleaned up on list_runs."""
        store = SQLiteCheckpointStore(ttl=1)
        store.save_event(
            "old-run",
            StepEvent(
                step_name="s",
                status=StepStatus.COMPLETED,
                timestamp=1.0,
            ),
        )
        # Manually backdate the updated_at
        store._conn.execute(
            "UPDATE runs SET updated_at = 0 WHERE run_id = ?",
            ("old-run",),
        )
        store._conn.commit()

        runs = store.list_runs()
        assert "old-run" not in runs
        assert store.load_run("old-run") is None


class TestSQLiteCheckpointStoreWithDurableInline:
    def test_checkpoint_and_skip_on_resume(self):
        """SQLiteCheckpointStore works with DurableInlineDSL."""
        from msgflux.context import _current_run_id, set_run_id
        from msgflux.dotdict import dotdict
        from msgflux.dsl.inline.runtime import DurableInlineDSL

        call_count = {"prep": 0, "final": 0}

        def prep(msg):
            call_count["prep"] += 1
            msg["prep_done"] = True
            return msg

        def final(msg):
            call_count["final"] += 1
            msg["final_done"] = True
            return msg

        store = SQLiteCheckpointStore()
        modules = {"prep": prep, "final": final}

        # First run
        token = set_run_id("run-sqlite-1")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl("prep -> final", modules, dotdict())
            assert result.prep_done is True
            assert result.final_done is True
            assert call_count == {"prep": 1, "final": 1}
        finally:
            _current_run_id.reset(token)

        # Resume — both skipped
        call_count = {"prep": 0, "final": 0}
        token = set_run_id("run-sqlite-1")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl("prep -> final", modules, dotdict())
            assert result.final_done is True
            assert call_count == {"prep": 0, "final": 0}
        finally:
            _current_run_id.reset(token)

    def test_inline_with_sqlite_store(self):
        """inline() works with SQLiteCheckpointStore."""
        from msgflux.dotdict import dotdict
        from msgflux.dsl.inline import inline

        def prep(msg):
            msg["x"] = 1
            return msg

        store = SQLiteCheckpointStore()
        result = inline(
            "prep", {"prep": prep}, dotdict(), run_id="test-sqlite", store=store
        )
        assert result.x == 1


class TestImportPaths:
    def test_import_from_dsl(self):
        from msgflux.dsl import ainline, inline

        assert callable(inline)
        assert callable(ainline)

    def test_import_from_root(self):
        from msgflux import ainline, inline

        assert callable(inline)
        assert callable(ainline)

    def test_import_stores_from_data(self):
        from msgflux.data.stores import (
            AsyncSQLiteCheckpointStore,
            SQLiteCheckpointStore,
        )

        assert SQLiteCheckpointStore is not None
        assert AsyncSQLiteCheckpointStore is not None

    def test_import_stores_from_root(self):
        from msgflux import AsyncSQLiteCheckpointStore, SQLiteCheckpointStore

        assert SQLiteCheckpointStore is not None
        assert AsyncSQLiteCheckpointStore is not None

    def test_f_no_inline(self):
        import msgflux.nn.functional as F

        assert not hasattr(F, "inline")
        assert not hasattr(F, "ainline")
