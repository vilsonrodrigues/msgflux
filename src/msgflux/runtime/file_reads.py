from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterator

from msgflux.context import get_execution_scope


@dataclass(frozen=True)
class FileReadRecord:
    path: str
    text_hash: str
    session_id: str
    namespace: str
    run_id: str | None = None


class FileReadTracker:
    """Tracks files read during the active runtime context."""

    def __init__(self) -> None:
        self._records: dict[str, FileReadRecord] = {}

    def mark_read(self, path: str | Path, content: str) -> FileReadRecord:
        resolved_path = str(Path(path).expanduser().resolve())
        scope = get_execution_scope()
        record = FileReadRecord(
            path=resolved_path,
            text_hash=hash_text(content),
            session_id=scope.session_id,
            namespace=scope.namespace,
            run_id=scope.run_id,
        )
        self._records[resolved_path] = record
        return record

    def get(self, path: str | Path) -> FileReadRecord | None:
        resolved_path = str(Path(path).expanduser().resolve())
        record = self._records.get(resolved_path)
        if record is None:
            return None
        scope = get_execution_scope()
        if (
            record.session_id != scope.session_id
            or record.namespace != scope.namespace
            or record.run_id != scope.run_id
        ):
            return None
        return record

    def clear(self) -> None:
        self._records.clear()


_FILE_READ_TRACKER: ContextVar[FileReadTracker | None] = ContextVar(
    "msgflux_file_read_tracker",
    default=None,
)


def get_file_read_tracker() -> FileReadTracker:
    tracker = _FILE_READ_TRACKER.get()
    if tracker is None:
        tracker = FileReadTracker()
        _FILE_READ_TRACKER.set(tracker)
    return tracker


@contextmanager
def file_read_tracker_context(tracker: FileReadTracker) -> Iterator[None]:
    token = _FILE_READ_TRACKER.set(tracker)
    try:
        yield
    finally:
        _FILE_READ_TRACKER.reset(token)


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
