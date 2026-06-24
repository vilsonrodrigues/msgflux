from __future__ import annotations

import time
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, List, Mapping

from msgflux.data.stores.base import CheckpointStore
from msgflux.data.stores.registry import register_store
from msgflux.data.stores.types import CheckpointStoreType


@register_store()
class InMemoryCheckpointStore(CheckpointStore, CheckpointStoreType):
    """In-memory checkpoint store for tests and local prototyping."""

    provider = "in_memory"

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._lock = RLock()

    def _get_run(
        self, namespace: str, thread_id: str, run_id: str
    ) -> Dict[str, Any] | None:
        return self._data.get(namespace, {}).get(thread_id, {}).get(run_id)

    def _ensure_run(
        self, namespace: str, thread_id: str, run_id: str
    ) -> Dict[str, Any]:
        ns = self._data.setdefault(namespace, {})
        thread = ns.setdefault(thread_id, {})
        run = thread.get(run_id)
        if run is None:
            run = {"state": {}, "events": [], "updated_at": time.time()}
            thread[run_id] = run
        return run

    def save_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = deepcopy(dict(state))
            run["updated_at"] = time.time()

    def load_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return None
            return deepcopy(run["state"])

    def append_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["events"].append(deepcopy(dict(event)))

    def load_events(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return []
            return deepcopy(run["events"])

    def save_with_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = deepcopy(dict(state))
            run["updated_at"] = time.time()
            run["events"].append(deepcopy(dict(event)))

    def list_runs(
        self,
        namespace: str,
        thread_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            thread_runs = self._data.get(namespace, {}).get(thread_id, {})
            entries: List[Dict[str, Any]] = []
            for run_id, run in thread_runs.items():
                run_status = run["state"].get("status")
                if status is not None and run_status != status:
                    continue
                entries.append(
                    {
                        "run_id": run_id,
                        "status": run_status,
                        "updated_at": run["updated_at"],
                    }
                )
            entries.sort(key=lambda e: e["updated_at"], reverse=True)
            if limit is not None:
                entries = entries[:limit]
            return entries

    def delete_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> bool:
        with self._lock:
            thread_runs = self._data.get(namespace, {}).get(thread_id, {})
            if run_id in thread_runs:
                del thread_runs[run_id]
                return True
            return False

    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        cutoff = time.time() - older_than if older_than is not None else None
        removed = 0
        with self._lock:
            namespaces = (
                [namespace] if namespace is not None else list(self._data.keys())
            )
            for ns in namespaces:
                ns_data = self._data.get(ns)
                if ns_data is None:
                    continue
                threads = (
                    [thread_id] if thread_id is not None else list(ns_data.keys())
                )
                for sid in threads:
                    thread = ns_data.get(sid)
                    if thread is None:
                        continue
                    to_delete = []
                    for rid, run in thread.items():
                        if cutoff is not None and run["updated_at"] >= cutoff:
                            continue
                        to_delete.append(rid)
                    for rid in to_delete:
                        del thread[rid]
                        removed += 1
                    if not thread:
                        del ns_data[sid]
                if not ns_data:
                    del self._data[ns]
        return removed
