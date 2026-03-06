"""Tests for checkpoint store implementations."""

import msgspec
import pytest

from msgflux.data.stores import (
    MemoryCheckpointStore,
    RunState,
    StepEvent,
    StepStatus,
)


class TestStepEventSerialization:
    def test_encode_decode(self):
        event = StepEvent(
            step_name="module:agent#0",
            status=StepStatus.COMPLETED,
            timestamp=12345.0,
            message_snapshot=b'{"key": "value"}',
        )
        encoded = msgspec.json.encode(event)
        decoded = msgspec.json.decode(encoded, type=StepEvent)
        assert decoded.step_name == "module:agent#0"
        assert decoded.status == StepStatus.COMPLETED
        assert decoded.message_snapshot == b'{"key": "value"}'
        assert decoded.retry_count == 0

    def test_with_retry_count(self):
        event = StepEvent(
            step_name="module:prep#0",
            status=StepStatus.FAILED,
            timestamp=1.0,
            error="connection timeout",
            retry_count=2,
        )
        encoded = msgspec.json.encode(event)
        decoded = msgspec.json.decode(encoded, type=StepEvent)
        assert decoded.retry_count == 2
        assert decoded.error == "connection timeout"


class TestRunStateSerialization:
    def test_encode_decode(self):
        run = RunState(
            run_id="run-1",
            expression="prep -> final",
            events=[
                StepEvent(
                    step_name="module:prep#0",
                    status=StepStatus.COMPLETED,
                    timestamp=1.0,
                ),
            ],
            created_at=1.0,
            updated_at=2.0,
        )
        encoded = msgspec.json.encode(run)
        decoded = msgspec.json.decode(encoded, type=RunState)
        assert decoded.run_id == "run-1"
        assert len(decoded.events) == 1
        assert decoded.events[0].step_name == "module:prep#0"


class TestMemoryCheckpointStore:
    def test_save_and_load_event(self):
        store = MemoryCheckpointStore()
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
        store = MemoryCheckpointStore()
        assert store.load_run("no-such-run") is None

    def test_delete_run(self):
        store = MemoryCheckpointStore()
        event = StepEvent(
            step_name="s",
            status=StepStatus.COMPLETED,
            timestamp=1.0,
        )
        store.save_event("run-1", event)
        store.delete_run("run-1")
        assert store.load_run("run-1") is None

    def test_delete_nonexistent_run(self):
        store = MemoryCheckpointStore()
        store.delete_run("no-such-run")  # should not raise

    def test_list_runs(self):
        store = MemoryCheckpointStore()
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
        store = MemoryCheckpointStore()
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
        store = MemoryCheckpointStore()
        assert store.get_last_completed_step("nonexistent") is None

    def test_get_step_status(self):
        store = MemoryCheckpointStore()
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
        store = MemoryCheckpointStore()
        assert store.get_step_status("r", "s1") is None

    def test_get_step_retry_count(self):
        store = MemoryCheckpointStore()
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
        store = MemoryCheckpointStore()
        assert store.get_step_retry_count("r", "s1") == 0

    def test_multiple_events_for_same_step(self):
        store = MemoryCheckpointStore()
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
