"""Per-thread durable storage for the coding interface."""

from __future__ import annotations

import os
import re
from pathlib import Path
from uuid import uuid4

import msgspec

from msgflux.data.stores.providers.sqlite import SQLiteCheckpointStore
from msgflux.runtime.approvals.providers.sqlite import SQLiteApprovalStore
from msgflux.runtime.context import new_thread_id
from msgflux.utils.time import utc_now_isoformat

_THREAD_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)


class ThreadMetadata(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Small discovery metadata; conversation history stays in checkpoints."""

    thread_id: str
    created_at: str
    updated_at: str
    workspace: str | None = None
    title: str = ""


def validate_thread_id(thread_id: str) -> str:
    """Validate the complete ID format before it is used in a filesystem path."""
    if not isinstance(thread_id, str) or _THREAD_ID_RE.fullmatch(thread_id) is None:
        raise ValueError(
            "thread_id must contain 1-128 ASCII letters, digits, underscores or hyphens"
        )
    return thread_id


class ThreadStorage:
    """Manage ``threads/<thread_id>/`` directories under a storage root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def threads_dir(self) -> Path:
        return self.root / "threads"

    def thread_dir(self, thread_id: str) -> Path:
        thread_id = validate_thread_id(thread_id)
        return self.threads_dir / thread_id

    def new_thread(self) -> ThreadMetadata:
        """Create a new thread using the runtime's canonical ID generator."""
        return self.create_thread(new_thread_id())

    def create_thread(
        self, thread_id: str, *, workspace: str | None = None
    ) -> ThreadMetadata:
        """Create a thread directory and its initial metadata file."""
        directory = self.thread_dir(thread_id)
        self._prepare_parent()
        if directory.is_symlink():
            raise ValueError("thread directory must not be a symlink")
        directory.mkdir(parents=False, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            return self.read_metadata(thread_id)
        now = utc_now_isoformat()
        metadata = ThreadMetadata(thread_id, now, now, workspace)
        self._write_metadata(metadata_path, metadata)
        return metadata

    def read_metadata(self, thread_id: str) -> ThreadMetadata:
        """Read metadata for an existing thread, rejecting malformed content."""
        directory = self.thread_dir(thread_id)
        if directory.is_symlink():
            raise ValueError("thread directory must not be a symlink")
        path = directory / "metadata.json"
        if path.is_symlink():
            raise ValueError("thread metadata must not be a symlink")
        try:
            metadata = msgspec.json.decode(path.read_bytes(), type=ThreadMetadata)
            if metadata.thread_id != thread_id:
                raise ValueError(
                    "thread metadata identity does not match its directory"
                )
            return metadata
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"thread metadata not found: {thread_id}") from exc
        except msgspec.DecodeError as exc:
            raise ValueError(f"invalid metadata for thread {thread_id}") from exc

    def list_threads(
        self, *, workspace: str | None = None
    ) -> tuple[ThreadMetadata, ...]:
        """Discover valid threads without creating directories or databases."""
        if not self.threads_dir.exists():
            return ()
        if self.threads_dir.is_symlink():
            raise ValueError("threads directory must not be a symlink")
        threads = []
        for directory in self.threads_dir.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                metadata = self.read_metadata(directory.name)
            except (ValueError, OSError):
                continue
            checkpoint = directory / "checkpoints.sqlite3"
            if checkpoint.is_symlink() or not checkpoint.is_file():
                continue
            if workspace is None or metadata.workspace in {None, workspace}:
                threads.append(metadata)
        return tuple(
            sorted(
                threads,
                key=lambda item: (item.updated_at, item.thread_id),
                reverse=True,
            )
        )

    def touch(self, thread_id: str, *, title: str | None = None) -> ThreadMetadata:
        """Atomically update discovery metadata, without storing a transcript."""
        metadata = self.read_metadata(thread_id)
        updated = msgspec.structs.replace(
            metadata,
            updated_at=utc_now_isoformat(),
            title=metadata.title if title is None else title[:160],
        )
        path = self.thread_dir(thread_id) / "metadata.json"
        temporary = path.with_name(f".metadata-{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(msgspec.json.encode(updated))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return updated

    def checkpoint_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "checkpoints.sqlite3"

    def approval_path(self, thread_id: str) -> Path:
        return self.thread_dir(thread_id) / "approvals.sqlite3"

    def open_checkpoint_store(self, thread_id: str) -> SQLiteCheckpointStore:
        """Open the checkpoint SQLite store for a thread."""
        self._ensure_thread(thread_id)
        path = self.checkpoint_path(thread_id)
        self._prepare_db(path)
        return SQLiteCheckpointStore(str(path))

    def open_approval_store(self, thread_id: str) -> SQLiteApprovalStore:
        """Open the approval journal SQLite store for a thread."""
        self._ensure_thread(thread_id)
        path = self.approval_path(thread_id)
        self._prepare_db(path)
        return SQLiteApprovalStore(str(path))

    def _prepare_parent(self) -> None:
        self.threads_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.threads_dir.is_symlink():
            raise ValueError("threads directory must not be a symlink")
        self.threads_dir.chmod(0o700)

    def _ensure_thread(self, thread_id: str) -> None:
        directory = self.thread_dir(thread_id)
        self._prepare_parent()
        if directory.is_symlink():
            raise ValueError("thread directory must not be a symlink")
        if not directory.exists():
            self.create_thread(thread_id)

    @staticmethod
    def _write_metadata(path: Path, metadata: ThreadMetadata) -> None:
        # Exclusive creation avoids replacing metadata if another process wins
        # the creation race. Metadata intentionally contains no transcript.
        payload = msgspec.json.encode(metadata)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
        except FileExistsError:
            pass

    @staticmethod
    def _prepare_db(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("thread database must not be a symlink")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if not path.is_file() or path.is_symlink():
                raise ValueError("thread database must be a regular file") from None
            path.chmod(0o600)
        else:
            os.close(descriptor)
