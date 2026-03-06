"""SQLite-based checkpoint stores for durable pipeline execution.

Provides both synchronous (sqlite3 stdlib) and asynchronous (aiosqlite)
implementations using event-append storage instead of read-modify-write.
"""

import sqlite3
import time
from typing import List, Optional

from msgflux.data.stores.base import AsyncCheckpointStore, CheckpointStore
from msgflux.data.stores.schemas import RunState, StepEvent, StepStatus

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    expression TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    timestamp REAL NOT NULL,
    message_snapshot BLOB,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_events_run_step
    ON events(run_id, step_name);
"""


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema."""
    conn.executescript(_SCHEMA)


def _row_to_step_event(row: tuple) -> StepEvent:
    """Convert a database row to a StepEvent."""
    return StepEvent(
        step_name=row[0],
        status=row[1],
        timestamp=row[2],
        message_snapshot=row[3],
        error=row[4],
        retry_count=row[5],
    )


class SQLiteCheckpointStore(CheckpointStore):
    """Persistent checkpoint store backed by SQLite (stdlib).

    Uses event-append storage: each ``save_event`` call is a single
    INSERT — no read-modify-write cycle.

    Args:
        path: SQLite database path.  ``":memory:"`` for in-memory.
        ttl: Time-to-live for runs in seconds.  ``None`` disables cleanup.
    """

    def __init__(
        self,
        path: str = ":memory:",
        ttl: Optional[int] = 86400,
    ):
        self.path = path
        self.ttl = ttl
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        _init_db(self._conn)

    def save_event(self, run_id: str, event: StepEvent) -> None:
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs "
                "(run_id, expression, created_at, updated_at) "
                "VALUES (?, '', ?, ?)",
                (run_id, now, now),
            )
            self._conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
            self._conn.execute(
                "INSERT INTO events "
                "(run_id, step_name, status, timestamp, "
                "message_snapshot, error, retry_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    event.step_name,
                    event.status,
                    event.timestamp,
                    event.message_snapshot,
                    event.error,
                    event.retry_count,
                ),
            )

    def load_run(self, run_id: str) -> Optional[RunState]:
        row = self._conn.execute(
            "SELECT run_id, expression, created_at, updated_at "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None

        event_rows = self._conn.execute(
            "SELECT step_name, status, timestamp, message_snapshot, error, retry_count "
            "FROM events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()

        events = [_row_to_step_event(r) for r in event_rows]
        return RunState(
            run_id=row[0],
            expression=row[1],
            events=events,
            created_at=row[2],
            updated_at=row[3],
        )

    def delete_run(self, run_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def list_runs(self) -> List[str]:
        if self.ttl is not None:
            cutoff = time.time() - self.ttl
            with self._conn:
                expired = self._conn.execute(
                    "SELECT run_id FROM runs WHERE updated_at < ?", (cutoff,)
                ).fetchall()
                for (rid,) in expired:
                    self.delete_run(rid)

        rows = self._conn.execute("SELECT run_id FROM runs").fetchall()
        return [r[0] for r in rows]

    def get_step_status(self, run_id: str, step_name: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT status FROM events "
            "WHERE run_id = ? AND step_name = ? ORDER BY id DESC LIMIT 1",
            (run_id, step_name),
        ).fetchone()
        return row[0] if row else None

    def get_step_retry_count(self, run_id: str, step_name: str) -> int:
        row = self._conn.execute(
            "SELECT retry_count FROM events "
            "WHERE run_id = ? AND step_name = ? AND status = ? "
            "ORDER BY id DESC LIMIT 1",
            (run_id, step_name, StepStatus.FAILED),
        ).fetchone()
        return row[0] if row else 0

    def get_last_completed_step(self, run_id: str) -> Optional[StepEvent]:
        row = self._conn.execute(
            "SELECT step_name, status, timestamp, message_snapshot, error, retry_count "
            "FROM events WHERE run_id = ? AND status = ? ORDER BY id DESC LIMIT 1",
            (run_id, StepStatus.COMPLETED),
        ).fetchone()
        return _row_to_step_event(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self):
        try:
            self._conn.close()
        except Exception:  # noqa: S110
            pass


class AsyncSQLiteCheckpointStore(AsyncCheckpointStore):
    """Persistent async checkpoint store backed by SQLite via aiosqlite.

    Uses event-append storage with true async I/O.

    Args:
        path: SQLite database path.  ``":memory:"`` for in-memory.
        ttl: Time-to-live for runs in seconds.  ``None`` disables cleanup.
    """

    def __init__(
        self,
        path: str = ":memory:",
        ttl: Optional[int] = 86400,
    ):
        try:
            import aiosqlite  # noqa: PLC0415, F401
        except ImportError:
            raise ImportError(
                "`aiosqlite` is not available. Install with `pip install aiosqlite`"
            ) from None
        self.path = path
        self.ttl = ttl
        self._db = None

    async def _get_db(self):
        """Lazily initialize the database connection."""
        if self._db is None:
            import aiosqlite  # noqa: PLC0415

            self._db = await aiosqlite.connect(self.path)
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.executescript(_SCHEMA)
        return self._db

    async def asave_event(self, run_id: str, event: StepEvent) -> None:
        db = await self._get_db()
        now = time.time()
        await db.execute(
            "INSERT OR IGNORE INTO runs (run_id, expression, created_at, updated_at) "
            "VALUES (?, '', ?, ?)",
            (run_id, now, now),
        )
        await db.execute(
            "UPDATE runs SET updated_at = ? WHERE run_id = ?",
            (now, run_id),
        )
        await db.execute(
            "INSERT INTO events "
            "(run_id, step_name, status, timestamp, "
            "message_snapshot, error, retry_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                event.step_name,
                event.status,
                event.timestamp,
                event.message_snapshot,
                event.error,
                event.retry_count,
            ),
        )
        await db.commit()

    async def aload_run(self, run_id: str) -> Optional[RunState]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT run_id, expression, created_at, updated_at "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        cursor = await db.execute(
            "SELECT step_name, status, timestamp, message_snapshot, error, retry_count "
            "FROM events WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        event_rows = await cursor.fetchall()
        events = [_row_to_step_event(r) for r in event_rows]
        return RunState(
            run_id=row[0],
            expression=row[1],
            events=events,
            created_at=row[2],
            updated_at=row[3],
        )

    async def adelete_run(self, run_id: str) -> None:
        db = await self._get_db()
        await db.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
        await db.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        await db.commit()

    async def alist_runs(self) -> List[str]:
        db = await self._get_db()
        if self.ttl is not None:
            cutoff = time.time() - self.ttl
            cursor = await db.execute(
                "SELECT run_id FROM runs WHERE updated_at < ?", (cutoff,)
            )
            expired = await cursor.fetchall()
            for (rid,) in expired:
                await self.adelete_run(rid)

        cursor = await db.execute("SELECT run_id FROM runs")
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def aget_step_status(self, run_id: str, step_name: str) -> Optional[str]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT status FROM events "
            "WHERE run_id = ? AND step_name = ? ORDER BY id DESC LIMIT 1",
            (run_id, step_name),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def aget_step_retry_count(self, run_id: str, step_name: str) -> int:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT retry_count FROM events "
            "WHERE run_id = ? AND step_name = ? AND status = ? "
            "ORDER BY id DESC LIMIT 1",
            (run_id, step_name, StepStatus.FAILED),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def aget_last_completed_step(self, run_id: str) -> Optional[StepEvent]:
        db = await self._get_db()
        cursor = await db.execute(
            "SELECT step_name, status, timestamp, message_snapshot, error, retry_count "
            "FROM events WHERE run_id = ? AND status = ? ORDER BY id DESC LIMIT 1",
            (run_id, StepStatus.COMPLETED),
        )
        row = await cursor.fetchone()
        return _row_to_step_event(row) if row else None

    async def aclose(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None
