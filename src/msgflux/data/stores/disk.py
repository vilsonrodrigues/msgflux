"""Disk-based checkpoint store using diskcache (deprecated)."""

import time
from typing import List, Optional

import msgspec

from msgflux.data.stores.base import CheckpointStore
from msgflux.data.stores.schemas import RunState, StepEvent


class DiskCheckpointStore(CheckpointStore):
    """Persistent checkpoint store using diskcache.

    .. deprecated::
        Use :class:`SQLiteCheckpointStore` instead.
    """

    def __init__(self, directory: Optional[str] = None, ttl: Optional[int] = 86400):
        """Args:
        directory: Path for the diskcache directory. None uses default.
        ttl: Time-to-live for entries in seconds (default 24h).
        """
        import warnings  # noqa: PLC0415

        warnings.warn(
            "DiskCheckpointStore is deprecated, use SQLiteCheckpointStore",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from diskcache import Cache  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "`diskcache` is not available. Install with `pip install diskcache`"
            ) from None
        self.ttl = ttl
        self._cache = Cache(directory, timeout=1)

    def save_event(self, run_id: str, event: StepEvent) -> None:
        now = time.time()
        raw = self._cache.get(run_id)
        if raw is not None:
            run = msgspec.json.decode(raw, type=RunState)
        else:
            run = RunState(
                run_id=run_id,
                expression="",
                events=[],
                created_at=now,
                updated_at=now,
            )
        run.events.append(event)
        run.updated_at = now
        self._cache.set(run_id, msgspec.json.encode(run), expire=self.ttl)

    def load_run(self, run_id: str) -> Optional[RunState]:
        raw = self._cache.get(run_id)
        if raw is None:
            return None
        return msgspec.json.decode(raw, type=RunState)

    def delete_run(self, run_id: str) -> None:
        self._cache.delete(run_id)

    def list_runs(self) -> List[str]:
        return list(self._cache)
