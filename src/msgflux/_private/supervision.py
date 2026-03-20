"""One-for-one supervision helpers for durable gather execution.

Implements Erlang-style one-for-one supervision: each worker is monitored
independently.  When a worker fails, only that worker is restarted with
exponential backoff + jitter.  Other workers continue unaffected.
"""

import asyncio
import concurrent.futures
import random
import time
import uuid
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from msgflux._private.executor import Executor
from msgflux.exceptions import TaskError
from msgflux.logger import logger

if TYPE_CHECKING:
    from msgflux.data.stores.base import CheckpointStore

# Sentinel for pending worker slots (never serialized directly).
_PENDING = type(
    "_PENDING",
    (),
    {"__repr__": lambda self: "<PENDING>", "__bool__": lambda self: False},  # noqa: ARG005
)()


# ── Async call dispatch ──────────────────────────────────────────────────────


async def _async_call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call *fn* handling ``acall``, coroutine, and sync callables."""
    if hasattr(fn, "acall"):
        return await fn.acall(*args, **kwargs)
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    import functools  # noqa: PLC0415

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# ── Serialization ────────────────────────────────────────────────────────────


def _serialize_gather_result(r: Any) -> Any:
    if r is _PENDING:
        return {"__pending__": True}
    if isinstance(r, TaskError):
        return {"__task_error__": True, "index": r.index, "error": str(r.exception)}
    return r


def _deserialize_gather_result(r: Any) -> Any:
    if isinstance(r, dict):
        if r.get("__pending__"):
            return _PENDING
        if r.get("__task_error__"):
            return TaskError(exception=Exception(r["error"]), index=r["index"])
    return r


# ── Store helpers ─────────────────────────────────────────────────────────────


def _save_gather_state(
    store: "CheckpointStore",
    namespace: str,
    session_id: str,
    run_id: str,
    results: List[Any],
    total: int,
    status: str = "running",
) -> None:
    store.save_state(
        namespace,
        session_id,
        run_id,
        {
            "status": status,
            "results": [_serialize_gather_result(r) for r in results],
            "total": total,
        },
    )


async def _asave_gather_state(
    store: "CheckpointStore",
    namespace: str,
    session_id: str,
    run_id: str,
    results: List[Any],
    total: int,
    status: str = "running",
) -> None:
    state = {
        "status": status,
        "results": [_serialize_gather_result(r) for r in results],
        "total": total,
    }
    if hasattr(store, "asave_state"):
        await store.asave_state(namespace, session_id, run_id, state)
    else:
        store.save_state(namespace, session_id, run_id, state)


# ── Resume helpers ────────────────────────────────────────────────────────────


def _hydrate_results(state: Optional[dict], total: int) -> Optional[List[Any]]:
    """Extract and pad results from a stored state snapshot."""
    if state is None:
        return None
    if state.get("status") != "running":
        return None
    raw = state.get("results", [])
    results = [_deserialize_gather_result(r) for r in raw]
    # Pad / trim to match current total
    while len(results) < total:
        results.append(_PENDING)
    return results[:total]


def _default_results(total: int, run_id: Optional[str]) -> Tuple[List[Any], str]:
    return [_PENDING] * total, run_id or str(uuid.uuid4())


def _try_resume_gather(
    store: "CheckpointStore",
    namespace: str,
    session_id: str,
    run_id: Optional[str],
    total: int,
) -> Tuple[List[Any], str]:
    """Return ``(results, effective_run_id)``."""
    if run_id is not None:
        results = _hydrate_results(
            store.load_state(namespace, session_id, run_id),
            total,
        )
        if results is not None:
            return results, run_id
    else:
        incomplete = store.find_incomplete_runs(namespace, session_id)
        if incomplete:
            rid = incomplete[0]["run_id"]
            results = _hydrate_results(
                store.load_state(namespace, session_id, rid),
                total,
            )
            if results is not None:
                return results, rid
    return _default_results(total, run_id)


async def _atry_resume_gather(
    store: "CheckpointStore",
    namespace: str,
    session_id: str,
    run_id: Optional[str],
    total: int,
) -> Tuple[List[Any], str]:
    """Async version of :func:`_try_resume_gather`."""

    async def _load(ns: str, sid: str, rid: str) -> Any:
        if hasattr(store, "aload_state"):
            return await store.aload_state(ns, sid, rid)
        return store.load_state(ns, sid, rid)

    async def _find(ns: str, sid: str) -> List[Any]:
        if hasattr(store, "afind_incomplete_runs"):
            return await store.afind_incomplete_runs(ns, sid)
        return store.find_incomplete_runs(ns, sid)

    if run_id is not None:
        results = _hydrate_results(
            await _load(namespace, session_id, run_id),
            total,
        )
        if results is not None:
            return results, run_id
    else:
        inc = await _find(namespace, session_id)
        if inc:
            rid = inc[0]["run_id"]
            results = _hydrate_results(
                await _load(namespace, session_id, rid),
                total,
            )
            if results is not None:
                return results, rid
    return _default_results(total, run_id)


def _backoff_delay(attempt: int, base_delay: float) -> float:
    """Exponential backoff with jitter."""
    delay = base_delay * (2 ** (attempt - 1))
    return delay + random.uniform(0, delay * 0.3)  # noqa: S311


# ── Core durable gather ──────────────────────────────────────────────────────


def gather_durable_sync(  # noqa: C901
    workers: List[Tuple[Callable, tuple, dict]],
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    store: Optional["CheckpointStore"] = None,
    namespace: str = "gather",
    session_id: str = "default",
    run_id: Optional[str] = None,
) -> List[Any]:
    """One-for-one supervised gather (sync).

    *workers* is a list of ``(callable, args_tuple, kwargs_dict)`` tuples.
    Each worker is monitored independently.  On failure, only the failed
    worker is restarted with exponential backoff + jitter.
    """
    n = len(workers)

    # Resume from store
    if store is not None:
        results, effective_run_id = _try_resume_gather(
            store,
            namespace,
            session_id,
            run_id,
            n,
        )
    else:
        results: List[Any] = [_PENDING] * n
        effective_run_id = run_id or str(uuid.uuid4())

    # Submit pending workers
    executor = Executor.get_instance()
    active: Dict[Future, int] = {}
    retries_count = [0] * n

    for i, (fn, args, kwargs) in enumerate(workers):
        if results[i] is not _PENDING:
            continue
        active[executor.submit(fn, *args, **kwargs)] = i

    if not active:
        return results

    # Monitor loop (one-for-one)
    start = time.monotonic()
    while active:
        remaining = None
        if timeout is not None:
            remaining = max(0, timeout - (time.monotonic() - start))
            if remaining <= 0:
                break

        done, _ = concurrent.futures.wait(
            active.keys(),
            timeout=remaining,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )

        if not done:
            break

        for future in done:
            idx = active.pop(future)
            try:
                results[idx] = future.result()
            except Exception as e:
                if max_retries is not None and retries_count[idx] < max_retries:
                    retries_count[idx] += 1
                    time.sleep(_backoff_delay(retries_count[idx], retry_delay))
                    fn, args, kwargs = workers[idx]
                    active[executor.submit(fn, *args, **kwargs)] = idx
                else:
                    logger.error(str(e))
                    results[idx] = TaskError(exception=e, index=idx)

            if store is not None:
                _save_gather_state(
                    store,
                    namespace,
                    session_id,
                    effective_run_id,
                    results,
                    n,
                )

    # Timeout: mark remaining as failed
    for _future, idx in list(active.items()):
        results[idx] = TaskError(
            exception=TimeoutError("Worker timed out"),
            index=idx,
        )

    if store is not None:
        _save_gather_state(
            store,
            namespace,
            session_id,
            effective_run_id,
            results,
            n,
            status="completed",
        )

    return results


async def gather_durable_async(  # noqa: C901
    workers: List[Tuple[Callable, tuple, dict]],
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    store: Optional["CheckpointStore"] = None,
    namespace: str = "gather",
    session_id: str = "default",
    run_id: Optional[str] = None,
) -> List[Any]:
    """One-for-one supervised gather (async).

    Each worker runs as an independent ``asyncio.Task`` with its own retry
    loop.  Failures are isolated — one worker crashing does not affect others.
    """
    n = len(workers)

    # Resume from store
    if store is not None:
        results, effective_run_id = await _atry_resume_gather(
            store,
            namespace,
            session_id,
            run_id,
            n,
        )
    else:
        results: List[Any] = [_PENDING] * n
        effective_run_id = run_id or str(uuid.uuid4())

    # Supervised worker coroutine
    async def _supervised(fn: Callable, args: tuple, kwargs: dict) -> Any:
        retries = 0
        while True:
            try:
                return await _async_call(fn, *args, **kwargs)
            except Exception:
                if max_retries is not None and retries < max_retries:
                    retries += 1
                    await asyncio.sleep(_backoff_delay(retries, retry_delay))
                else:
                    raise

    # Create tasks for pending workers
    task_map: Dict[asyncio.Task, int] = {}
    for i, (fn, args, kwargs) in enumerate(workers):
        if results[i] is not _PENDING:
            continue
        task = asyncio.create_task(_supervised(fn, args, kwargs))
        task_map[task] = i

    if not task_map:
        return results

    # Monitor loop (one-for-one)
    pending = set(task_map.keys())
    start = time.monotonic()
    while pending:
        remaining = None
        if timeout is not None:
            remaining = max(0, timeout - (time.monotonic() - start))
            if remaining <= 0:
                break

        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=remaining,
        )

        if not done:
            break

        for task in done:
            idx = task_map[task]
            try:
                results[idx] = task.result()
            except Exception as e:
                logger.error(str(e))
                results[idx] = TaskError(exception=e, index=idx)

            if store is not None:
                await _asave_gather_state(
                    store,
                    namespace,
                    session_id,
                    effective_run_id,
                    results,
                    n,
                )

    # Timeout: cancel remaining
    for task in pending:
        task.cancel()
        idx = task_map[task]
        results[idx] = TaskError(
            exception=TimeoutError("Worker timed out"),
            index=idx,
        )

    if store is not None:
        await _asave_gather_state(
            store,
            namespace,
            session_id,
            effective_run_id,
            results,
            n,
            status="completed",
        )

    return results
