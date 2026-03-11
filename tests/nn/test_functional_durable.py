"""Tests for durable gather execution (one-for-one supervision)."""

import asyncio
import threading

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskError
from msgflux.nn import functional as F
from msgflux._private.supervision import (
    _PENDING,
    _backoff_delay,
    _deserialize_gather_result,
    _serialize_gather_result,
    gather_durable_async,
    gather_durable_sync,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ok(x):
    return x * 2


def _fail_then_ok(failures_left):
    """Returns a callable that fails *failures_left* times then succeeds."""
    counter = {"n": failures_left}

    def fn(x):
        if counter["n"] > 0:
            counter["n"] -= 1
            raise ValueError(f"temporary failure ({counter['n']} left)")
        return x * 2

    return fn


def _always_fail(x):
    raise RuntimeError(f"permanent failure for {x}")


async def _async_ok(x):
    await asyncio.sleep(0.01)
    return x * 2


def _async_fail_then_ok(failures_left):
    counter = {"n": failures_left}

    async def fn(x):
        if counter["n"] > 0:
            counter["n"] -= 1
            raise ValueError("async temporary failure")
        return x * 2

    return fn


async def _async_always_fail(x):
    raise RuntimeError(f"async permanent failure for {x}")


# ── Serialization ────────────────────────────────────────────────────────────


class TestSerialization:
    def test_serialize_pending(self):
        result = _serialize_gather_result(_PENDING)
        assert result == {"__pending__": True}

    def test_serialize_task_error(self):
        err = TaskError(exception=ValueError("boom"), index=2)
        result = _serialize_gather_result(err)
        assert result["__task_error__"] is True
        assert result["index"] == 2
        assert "boom" in result["error"]

    def test_serialize_normal_value(self):
        assert _serialize_gather_result(42) == 42
        assert _serialize_gather_result("hello") == "hello"
        assert _serialize_gather_result({"key": "val"}) == {"key": "val"}

    def test_deserialize_pending(self):
        assert _deserialize_gather_result({"__pending__": True}) is _PENDING

    def test_deserialize_task_error(self):
        raw = {"__task_error__": True, "index": 1, "error": "oops"}
        result = _deserialize_gather_result(raw)
        assert isinstance(result, TaskError)
        assert result.index == 1

    def test_deserialize_normal_value(self):
        assert _deserialize_gather_result(42) == 42
        assert _deserialize_gather_result("hello") == "hello"

    def test_roundtrip(self):
        values = [42, "text", _PENDING, TaskError(Exception("err"), 0)]
        for v in values:
            serialized = _serialize_gather_result(v)
            deserialized = _deserialize_gather_result(serialized)
            if v is _PENDING:
                assert deserialized is _PENDING
            elif isinstance(v, TaskError):
                assert isinstance(deserialized, TaskError)
            else:
                assert deserialized == v


class TestBackoffDelay:
    def test_first_attempt(self):
        delay = _backoff_delay(1, 1.0)
        assert 1.0 <= delay <= 1.3  # base + up to 30% jitter

    def test_exponential_growth(self):
        d1 = _backoff_delay(1, 1.0)
        d2 = _backoff_delay(2, 1.0)
        d3 = _backoff_delay(3, 1.0)
        # d2 base is 2.0, d3 base is 4.0
        assert d2 > d1
        assert d3 > d2


# ── scatter_gather with max_retries ──────────────────────────────────────────


class TestScatterGatherDurable:
    def test_all_succeed_no_retry(self):
        results = F.scatter_gather(
            [_ok, _ok, _ok],
            args_list=[(1,), (2,), (3,)],
            max_retries=3,
        )
        assert results == (2, 4, 6)

    def test_one_retries_and_succeeds(self):
        flaky = _fail_then_ok(2)
        results = F.scatter_gather(
            [_ok, flaky, _ok],
            args_list=[(1,), (5,), (3,)],
            max_retries=3,
            retry_delay=0.01,
        )
        assert results == (2, 10, 6)

    def test_one_exhausts_retries(self):
        results = F.scatter_gather(
            [_ok, _always_fail, _ok],
            args_list=[(1,), (5,), (3,)],
            max_retries=2,
            retry_delay=0.01,
        )
        assert results[0] == 2
        assert results[2] == 6
        assert isinstance(results[1], TaskError)
        assert results[1].index == 1

    def test_max_retries_zero(self):
        results = F.scatter_gather(
            [_always_fail],
            args_list=[(1,)],
            max_retries=0,
        )
        assert isinstance(results[0], TaskError)

    def test_without_max_retries_unchanged(self):
        """No max_retries → original fast path."""
        results = F.scatter_gather(
            [_ok, _ok],
            args_list=[(1,), (2,)],
        )
        assert results == (2, 4)


# ── scatter_gather with store ────────────────────────────────────────────────


class TestScatterGatherStore:
    def test_store_saves_completed(self):
        store = InMemoryCheckpointStore()
        results = F.scatter_gather(
            [_ok, _ok],
            args_list=[(1,), (2,)],
            max_retries=0,
            store=store,
            namespace="test",
            session_id="s1",
            run_id="run1",
        )
        assert results == (2, 4)

        state = store.load_state("test", "s1", "run1")
        assert state is not None
        assert state["status"] == "completed"
        assert state["results"] == [2, 4]

    def test_store_resume_skips_completed_workers(self):
        store = InMemoryCheckpointStore()

        # Save partial results (worker 0 done, worker 1 pending)
        store.save_state("test", "s1", "run1", {
            "status": "running",
            "results": [42, {"__pending__": True}],
            "total": 2,
        })

        results = F.scatter_gather(
            [_always_fail, _ok],  # worker 0 would fail if re-run
            args_list=[(1,), (2,)],
            max_retries=0,
            store=store,
            namespace="test",
            session_id="s1",
            run_id="run1",
        )
        # Worker 0 should use stored result (42), worker 1 executes
        assert results == (42, 4)

    def test_store_auto_resume_without_run_id(self):
        store = InMemoryCheckpointStore()

        # Save incomplete run (no explicit run_id match needed)
        store.save_state("test", "s1", "auto_run", {
            "status": "running",
            "results": [100, {"__pending__": True}, {"__pending__": True}],
            "total": 3,
        })

        results = F.scatter_gather(
            [_always_fail, _ok, _ok],
            args_list=[(1,), (2,), (3,)],
            max_retries=0,
            store=store,
            namespace="test",
            session_id="s1",
            # run_id not specified → auto-detects incomplete
        )
        assert results[0] == 100  # resumed
        assert results[1] == 4
        assert results[2] == 6


# ── map_gather with max_retries ──────────────────────────────────────────────


class TestMapGatherDurable:
    def test_retry_succeeds(self):
        flaky = _fail_then_ok(1)
        results = F.map_gather(
            flaky,
            args_list=[(5,)],
            max_retries=3,
            retry_delay=0.01,
        )
        assert results == (10,)

    def test_without_max_retries_unchanged(self):
        results = F.map_gather(_ok, args_list=[(1,), (2,)])
        assert results == (2, 4)


# ── bcast_gather with max_retries ────────────────────────────────────────────


class TestBcastGatherDurable:
    def test_retry_succeeds(self):
        flaky = _fail_then_ok(1)
        results = F.bcast_gather(
            [_ok, flaky],
            5,
            max_retries=3,
            retry_delay=0.01,
        )
        assert results == (10, 10)

    def test_without_max_retries_unchanged(self):
        results = F.bcast_gather([_ok, _ok], 3)
        assert results == (6, 6)


# ── wait_for with max_retries ───────────────────────────────────────────────


class TestWaitForDurable:
    def test_retry_succeeds(self):
        counter = {"n": 2}

        def flaky(x):
            if counter["n"] > 0:
                counter["n"] -= 1
                raise ValueError("temp")
            return x * 3

        result = F.wait_for(flaky, 4, max_retries=3, retry_delay=0.01)
        assert result == 12

    def test_retry_exhausted(self):
        def always_fail(x):
            raise RuntimeError("permanent")

        result = F.wait_for(always_fail, 1, max_retries=2, retry_delay=0.01)
        assert isinstance(result, TaskError)


# ── Async durable tests ─────────────────────────────────────────────────────


class TestAsyncScatterGatherDurable:
    @pytest.mark.asyncio
    async def test_all_succeed(self):
        results = await F.ascatter_gather(
            [_async_ok, _async_ok],
            args_list=[(1,), (2,)],
            max_retries=3,
        )
        assert results == (2, 4)

    @pytest.mark.asyncio
    async def test_retry_and_succeed(self):
        flaky = _async_fail_then_ok(2)
        results = await F.ascatter_gather(
            [_async_ok, flaky],
            args_list=[(1,), (5,)],
            max_retries=3,
            retry_delay=0.01,
        )
        assert results == (2, 10)

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        results = await F.ascatter_gather(
            [_async_ok, _async_always_fail],
            args_list=[(1,), (5,)],
            max_retries=1,
            retry_delay=0.01,
        )
        assert results[0] == 2
        assert isinstance(results[1], TaskError)

    @pytest.mark.asyncio
    async def test_store_saves_completed(self):
        store = InMemoryCheckpointStore()
        results = await F.ascatter_gather(
            [_async_ok, _async_ok],
            args_list=[(1,), (2,)],
            max_retries=0,
            store=store,
            namespace="async_test",
            session_id="s1",
            run_id="arun1",
        )
        assert results == (2, 4)

        state = store.load_state("async_test", "s1", "arun1")
        assert state is not None
        assert state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_store_resume(self):
        store = InMemoryCheckpointStore()

        store.save_state("async_test", "s1", "arun2", {
            "status": "running",
            "results": [99, {"__pending__": True}],
            "total": 2,
        })

        results = await F.ascatter_gather(
            [_async_always_fail, _async_ok],
            args_list=[(1,), (2,)],
            max_retries=0,
            store=store,
            namespace="async_test",
            session_id="s1",
            run_id="arun2",
        )
        assert results[0] == 99  # resumed
        assert results[1] == 4


class TestAsyncMapGatherDurable:
    @pytest.mark.asyncio
    async def test_retry_succeeds(self):
        flaky = _async_fail_then_ok(1)
        results = await F.amap_gather(
            flaky,
            args_list=[(5,)],
            max_retries=3,
            retry_delay=0.01,
        )
        assert results == (10,)


# ── Core gather_durable_sync tests ──────────────────────────────────────────


class TestGatherDurableSyncCore:
    def test_thread_safety(self):
        """Multiple workers running concurrently in thread pool."""
        results_tracker = []

        def slow_ok(x):
            import time
            time.sleep(0.05)
            return x

        workers = [(slow_ok, (i,), {}) for i in range(5)]
        results = gather_durable_sync(workers, max_retries=0)
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}

    def test_mixed_success_and_failure(self):
        workers = [
            (_ok, (1,), {}),
            (_always_fail, (2,), {}),
            (_ok, (3,), {}),
        ]
        results = gather_durable_sync(workers, max_retries=0)
        assert results[0] == 2
        assert isinstance(results[1], TaskError)
        assert results[2] == 6

    def test_no_workers_pending_after_resume(self):
        store = InMemoryCheckpointStore()
        store.save_state("ns", "s", "r", {
            "status": "running",
            "results": [10, 20],
            "total": 2,
        })

        workers = [(_always_fail, (1,), {}), (_always_fail, (2,), {})]
        results = gather_durable_sync(
            workers, store=store, namespace="ns", session_id="s", run_id="r",
        )
        # Both resumed from store, neither re-executed
        assert results == [10, 20]


# ── Core gather_durable_async tests ──────────────────────────────────────────


class TestGatherDurableAsyncCore:
    @pytest.mark.asyncio
    async def test_concurrent_async_workers(self):
        """Workers run concurrently as async tasks."""

        async def slow(x):
            await asyncio.sleep(0.05)
            return x

        workers = [(slow, (i,), {}) for i in range(5)]
        results = await gather_durable_async(workers, max_retries=0)
        assert set(results) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_sync_callable_in_async_gather(self):
        """Sync callables work in async gather via run_in_executor."""
        workers = [(_ok, (3,), {})]
        results = await gather_durable_async(workers, max_retries=0)
        assert results == [6]


# ── Namespace isolation ──────────────────────────────────────────────────────


class TestNamespaceIsolation:
    def test_different_namespaces_dont_interfere(self):
        store = InMemoryCheckpointStore()

        # Save for namespace "a"
        store.save_state("a", "s", "r", {
            "status": "running",
            "results": [100, {"__pending__": True}],
            "total": 2,
        })

        # Gather with namespace "b" should not resume from "a"
        results = F.scatter_gather(
            [_ok, _ok],
            args_list=[(1,), (2,)],
            max_retries=0,
            store=store,
            namespace="b",
            session_id="s",
            run_id="r",
        )
        assert results == (2, 4)  # fresh execution, not resumed
