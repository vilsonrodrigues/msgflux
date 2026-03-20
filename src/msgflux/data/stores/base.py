from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Optional

_TERMINAL_STATUSES = frozenset({"completed", "failed"})


class CheckpointStore(ABC):
    """Unified store for agent and pipeline checkpoints.

    Uses a composite key ``(namespace, session_id, run_id)`` to partition
    state across components, sessions, and individual runs.

    Two storage layers:

    * **State** - Snapshot of the current execution (UPSERT semantics).
    * **Events** - Append-only audit trail for debugging and replay.
    """

    # --- State operations ---

    @abstractmethod
    def save_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        """Persist a state snapshot (upsert)."""
        raise NotImplementedError

    @abstractmethod
    def load_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> Optional[Mapping[str, Any]]:
        """Load the latest state snapshot for a run."""
        raise NotImplementedError

    # --- Event operations ---

    @abstractmethod
    def append_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        """Append an execution event."""
        raise NotImplementedError

    @abstractmethod
    def load_events(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        """Load all events for a run in chronological order."""
        raise NotImplementedError

    # --- Atomic state + event ---

    def save_with_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        """Persist state and event together.

        Default: two sequential calls.  Backends with transaction support
        (e.g. SQLite) should override for atomicity.
        """
        self.save_state(namespace, session_id, run_id, state)
        self.append_event(namespace, session_id, run_id, event)

    # --- Query operations ---

    @abstractmethod
    def list_runs(
        self,
        namespace: str,
        session_id: str,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Mapping[str, Any]]:
        """List runs, most recent first.

        Each entry contains at least ``run_id``, ``status``, ``updated_at``.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_run(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> bool:
        """Delete a run and its events.  Returns ``True`` if anything was removed."""
        raise NotImplementedError

    # --- Convenience queries ---

    def load_latest_run(
        self,
        namespace: str,
        session_id: str,
    ) -> Optional[Mapping[str, Any]]:
        """Load the most recent run state (for resume)."""
        runs = self.list_runs(namespace, session_id, limit=1)
        if not runs:
            return None
        return self.load_state(namespace, session_id, runs[0]["run_id"])

    def find_incomplete_runs(
        self,
        namespace: str,
        session_id: str,
    ) -> List[Mapping[str, Any]]:
        """Find runs that did not reach a terminal status."""
        all_runs = self.list_runs(namespace, session_id)
        return [r for r in all_runs if r.get("status") not in _TERMINAL_STATUSES]

    # --- Cleanup ---

    @abstractmethod
    def clear(
        self,
        namespace: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        older_than: Optional[float] = None,
    ) -> int:
        """Remove runs matching the filter and return the count removed.

        Parameters
        ----------
        namespace:
            Restrict to a specific namespace (``None`` = all).
        session_id:
            Restrict to a specific session (``None`` = all).
        older_than:
            Remove runs whose ``updated_at`` is older than this many
            seconds ago.  ``None`` = no time filter.
        """
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
