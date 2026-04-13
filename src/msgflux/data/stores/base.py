from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

_TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped"})


class CheckpointStore(ABC):
    """Unified store for agent and pipeline checkpoints.

    The key is always `(namespace, session_id, run_id)`. State snapshots use
    UPSERT semantics while events remain append-only.
    """

    @abstractmethod
    def save_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def append_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_events(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    def save_with_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        self.save_state(namespace, session_id, run_id, state)
        self.append_event(namespace, session_id, run_id, event)

    @abstractmethod
    def list_runs(
        self,
        namespace: str,
        session_id: str,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete_run(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> bool:
        raise NotImplementedError

    def load_latest_run(
        self,
        namespace: str,
        session_id: str,
    ) -> Optional[Mapping[str, Any]]:
        runs = self.list_runs(namespace, session_id, limit=1)
        if not runs:
            return None
        return self.load_state(namespace, session_id, runs[0]["run_id"])

    def find_incomplete_runs(
        self,
        namespace: str,
        session_id: str,
    ) -> List[Mapping[str, Any]]:
        all_runs = self.list_runs(namespace, session_id)
        return [r for r in all_runs if r.get("status") not in _TERMINAL_STATUSES]

    @abstractmethod
    def clear(
        self,
        namespace: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        older_than: Optional[float] = None,
    ) -> int:
        raise NotImplementedError


class AsyncCheckpointStore(ABC):
    """Async mirror of :class:`CheckpointStore`."""

    @abstractmethod
    async def asave_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def aappend_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_events(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    async def asave_with_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        await self.asave_state(namespace, session_id, run_id, state)
        await self.aappend_event(namespace, session_id, run_id, event)

    @abstractmethod
    async def alist_runs(
        self,
        namespace: str,
        session_id: str,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def adelete_run(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> bool:
        raise NotImplementedError

    async def aload_latest_run(
        self,
        namespace: str,
        session_id: str,
    ) -> Optional[Mapping[str, Any]]:
        runs = await self.alist_runs(namespace, session_id, limit=1)
        if not runs:
            return None
        return await self.aload_state(namespace, session_id, runs[0]["run_id"])

    async def afind_incomplete_runs(
        self,
        namespace: str,
        session_id: str,
    ) -> List[Mapping[str, Any]]:
        all_runs = await self.alist_runs(namespace, session_id)
        return [r for r in all_runs if r.get("status") not in _TERMINAL_STATUSES]

    @abstractmethod
    async def aclear(
        self,
        namespace: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        older_than: Optional[float] = None,
    ) -> int:
        raise NotImplementedError
