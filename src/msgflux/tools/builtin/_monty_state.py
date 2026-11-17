"""Private, bounded, replace-in-place Monty session files.

This stays independent of AgentService: its process-lock helpers import the
service package and require POSIX, whereas these tools are usable by the SDK.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

import msgspec


class SessionRecord(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    version: int = 1
    variables: dict[str, object] = msgspec.field(default_factory=dict)
    invalid: bool = False
    call_id: str | None = None
    run_id: str | None = None
    code_hash: str | None = None
    completed_result: bytes | None = None


class SessionFiles:
    def __init__(self, directory: Path, max_bytes: int):
        self.directory = directory.expanduser().absolute()
        self.max_bytes = max_bytes

    def _prepare(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = self.directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Monty state directory must be a real directory")
        if os.name == "posix":
            if metadata.st_uid != os.getuid():
                raise PermissionError("Monty state directory must belong to this user")
            self.directory.chmod(0o700)

    @contextmanager
    def locked(self, key: str):
        self._prepare()
        descriptor = os.open(
            self.directory / f"{key}.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "r+b") as lock:
            metadata = os.fstat(lock.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Monty lock must be a regular file")
            if os.name == "posix" and metadata.st_uid != os.getuid():
                raise PermissionError("Monty lock must belong to this user")
            if os.name == "posix":
                import fcntl  # noqa: PLC0415

                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError("Monty session is busy") from exc
                try:
                    yield self.directory / f"{key}.json"
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            else:
                import msvcrt  # noqa: PLC0415

                lock.write(b"\0")
                lock.flush()
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise RuntimeError("Monty session is busy") from exc
                try:
                    yield self.directory / f"{key}.json"
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)

    def read(self, path: Path) -> bytes | None:
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("Monty state must be a regular file")
            if os.name == "posix" and (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise PermissionError("Monty state must be private to this user")
            data = source.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("Monty state file exceeds its size limit")
        return data

    def write(self, path: Path, data: bytes) -> None:
        if len(data) > self.max_bytes:
            raise ValueError("Monty state file exceeds its size limit")
        descriptor, temporary = tempfile.mkstemp(prefix=".monty-", dir=self.directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            if os.name == "posix":
                directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            Path(temporary).unlink(missing_ok=True)
