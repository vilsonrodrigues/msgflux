from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from threading import RLock
from typing import Literal

from msgflux.context import get_execution_context
from msgflux.runtime.events import (
    emit_file_edit_applied,
    emit_file_edit_failed,
    emit_file_edit_proposed,
    emit_file_edit_rejected,
)
from msgflux.runtime.file_reads import get_file_read_tracker, hash_text
from msgflux.runtime.permissions import (
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
    PermissionRuntimeError,
)

FileEditOperation = Literal["replace", "add", "update", "delete"]
DEFAULT_MAX_FILE_EDIT_DIFF_CHARS = 100_000


@dataclass(frozen=True)
class FileEditProposal:
    path: str
    operation: FileEditOperation
    diff: str
    old_text_hash: str
    new_text_hash: str
    lines_added: int
    lines_removed: int
    diff_truncated: bool = False
    diff_chars_original: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "operation": self.operation,
            "diff": self.diff,
            "old_text_hash": self.old_text_hash,
            "new_text_hash": self.new_text_hash,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "diff_truncated": self.diff_truncated,
            "diff_chars_original": self.diff_chars_original,
        }


@dataclass(frozen=True)
class _PatchOperation:
    operation: FileEditOperation
    path: Path
    old_text: str | None
    new_text: str | None
    line_ending: str = "\n"


class _FileEditCoordinator:
    """Serializes writes per file path without blocking unrelated files."""

    def __init__(self) -> None:
        self._locks: dict[Path, asyncio.Lock] = {}
        self._guard = RLock()

    @asynccontextmanager
    async def lock_paths(self, paths: list[Path]) -> AsyncIterator[None]:
        # Always acquire locks in path order. Multi-file patches can overlap, and
        # deterministic ordering avoids deadlocks when two patches touch the same
        # files in different orders.
        ordered_paths = sorted(set(paths), key=lambda path: path.as_posix())
        locks = [self._get_lock(path) for path in ordered_paths]
        for lock in locks:
            await lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()

    def _get_lock(self, path: Path) -> asyncio.Lock:
        with self._guard:
            lock = self._locks.get(path)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[path] = lock
            return lock


_FILE_EDIT_COORDINATOR = _FileEditCoordinator()


class FileEditRuntime:
    """Shared file-edit runtime for tools that propose and apply text changes."""

    def __init__(
        self,
        *,
        tool_name: str = "Edit",
        max_diff_chars: int = DEFAULT_MAX_FILE_EDIT_DIFF_CHARS,
    ) -> None:
        if max_diff_chars <= 0:
            raise ValueError(
                "FileEditRuntime max_diff_chars must be greater than zero."
            )
        self.tool_name = tool_name
        self.max_diff_chars = max_diff_chars

    async def replace(
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        path = Path(file_path).expanduser().resolve()
        try:
            proposal, new_content, line_ending = self._build_replace_proposal(
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
            emit_file_edit_proposed(proposal.to_dict())
            await self._enforce_permission(proposal)
            async with _FILE_EDIT_COORDINATOR.lock_paths([path]):
                # Approval can take an arbitrary amount of time. Re-check the
                # file under the path lock so a parallel edit cannot apply based
                # on the same stale read snapshot.
                self._ensure_file_unchanged_since_read(path=path, proposal=proposal)
                path.write_bytes(
                    new_content.replace("\n", line_ending).encode("utf-8")
                )
                get_file_read_tracker().mark_read(path, new_content)
            emit_file_edit_applied(proposal.to_dict())
            return f"Edit applied to {path}."
        except PermissionDeniedError:
            payload = {"path": str(path), "operation": "replace"}
            emit_file_edit_rejected(payload)
            raise
        except Exception as exc:
            payload = {"path": str(path), "operation": "replace", "error": str(exc)}
            emit_file_edit_failed(payload)
            raise

    async def apply_patch(self, *, patch: str) -> str:
        try:
            operations = self._parse_apply_patch(patch)
            proposals = [
                self._build_patch_proposal(operation) for operation in operations
            ]
            for proposal in proposals:
                emit_file_edit_proposed(proposal.to_dict())
            for proposal in proposals:
                await self._enforce_permission(proposal)
            async with _FILE_EDIT_COORDINATOR.lock_paths(
                [operation.path for operation in operations]
            ):
                # The patch proposal was built before permission resolution.
                # Revalidating inside the lock rejects concurrent edits instead
                # of applying a patch generated from an obsolete file state.
                for proposal in proposals:
                    self._ensure_patch_target_unchanged(proposal)
                self._write_patch_operations_atomically(operations)
                for operation in operations:
                    if operation.new_text is not None:
                        get_file_read_tracker().mark_read(
                            operation.path,
                            operation.new_text,
                        )
            for proposal in proposals:
                emit_file_edit_applied(proposal.to_dict())
            return f"Patch applied to {len(operations)} file(s)."
        except PermissionDeniedError:
            emit_file_edit_rejected({"path": None, "operation": "apply_patch"})
            raise
        except Exception as exc:
            emit_file_edit_failed(
                {"path": None, "operation": "apply_patch", "error": str(exc)}
            )
            raise

    def _build_replace_proposal(
        self,
        *,
        path: Path,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> tuple[FileEditProposal, str, str]:
        if not old_string:
            raise ValueError("Edit `old_string` cannot be empty.")
        if old_string == new_string:
            raise ValueError("Edit `old_string` and `new_string` are identical.")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        old_content, line_ending = self._read_text_preserving_newlines(path)
        read_record = get_file_read_tracker().get(path)
        if read_record is None:
            raise ValueError(
                "File must be read with the Read tool before it can be edited."
            )
        current_hash = hash_text(old_content)
        if read_record.text_hash != current_hash:
            raise ValueError(
                "File has changed since it was last read. Read the file again "
                "before editing."
            )
        occurrences = old_content.count(old_string)
        if occurrences == 0:
            raise ValueError("Edit `old_string` was not found in the file.")
        if occurrences > 1 and not replace_all:
            raise ValueError(
                "Edit `old_string` appears multiple times. Use replace_all=True "
                "or provide a more specific old_string."
            )

        count = -1 if replace_all else 1
        new_content = old_content.replace(old_string, new_string, count)
        diff = self._build_diff(
            path=path,
            old_content=old_content,
            new_content=new_content,
        )
        diff_for_payload, diff_truncated = self._truncate_diff(diff)
        return (
            FileEditProposal(
                path=str(path),
                operation="replace",
                diff=diff_for_payload,
                old_text_hash=current_hash,
                new_text_hash=hash_text(new_content),
                lines_added=self._count_diff_lines(diff, "+"),
                lines_removed=self._count_diff_lines(diff, "-"),
                diff_truncated=diff_truncated,
                diff_chars_original=len(diff),
            ),
            new_content,
            line_ending,
        )

    def _ensure_file_unchanged_since_read(
        self,
        *,
        path: Path,
        proposal: FileEditProposal,
    ) -> None:
        current_content, _line_ending = self._read_text_preserving_newlines(path)
        if hash_text(current_content) != proposal.old_text_hash:
            raise ValueError(
                "File changed after the edit was proposed. Read the file again "
                "before retrying the edit."
            )

    def _ensure_patch_target_unchanged(self, proposal: FileEditProposal) -> None:
        if proposal.operation == "add":
            if Path(proposal.path).exists():
                raise ValueError(f"File already exists: {proposal.path}")
            return
        path = Path(proposal.path)
        current_content, _line_ending = self._read_text_preserving_newlines(path)
        if hash_text(current_content) != proposal.old_text_hash:
            raise ValueError(
                "File changed after the patch was proposed. Read the file again "
                "before retrying the patch."
            )

    async def _enforce_permission(self, proposal: FileEditProposal) -> None:
        context = get_execution_context()
        manager = context.get("permission_manager")
        manager_missing = manager is None
        if manager is None:
            manager = PermissionManager()
        if not isinstance(manager, PermissionManager):
            raise TypeError("`permission_manager` must be a PermissionManager.")

        mode = self._resolve_permission_mode(manager)
        if manager_missing and mode == "ask_user":
            raise PermissionRuntimeError(
                "`ask_user` permission mode requires an active PermissionManager."
            )

        await manager.enforce_permission(
            PermissionRequest(
                action="file.edit",
                resource=proposal.path,
                tool_name=self.tool_name,
                risk="high",
                reason="Apply file edit",
                metadata={"preview": proposal.to_dict()},
            ),
            policy=mode,
        )

    def _resolve_permission_mode(self, manager: PermissionManager) -> str:
        scope = get_execution_context().get("scope")
        mode = getattr(scope, "permission_mode", None)
        if mode is None:
            mode = manager.default_mode
        if not isinstance(mode, str):
            raise TypeError("Permission mode must be a string.")
        return mode

    def _read_text_preserving_newlines(self, path: Path) -> tuple[str, str]:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        line_ending = "\r\n" if b"\r\n" in raw else "\n"
        return text.replace("\r\n", "\n"), line_ending

    def _parse_apply_patch(self, patch: str) -> list[_PatchOperation]:  # noqa: C901
        lines = patch.splitlines()
        if not lines or lines[0] != "*** Begin Patch":
            raise ValueError("Patch must start with `*** Begin Patch`.")
        if lines[-1] != "*** End Patch":
            raise ValueError("Patch must end with `*** End Patch`.")

        operations: list[_PatchOperation] = []
        index = 1
        while index < len(lines) - 1:
            line = lines[index]
            if line.startswith("*** Add File: "):
                path = self._resolve_patch_path(line.removeprefix("*** Add File: "))
                index += 1
                added_lines = []
                while index < len(lines) - 1 and not lines[index].startswith("*** "):
                    if not lines[index].startswith("+"):
                        raise ValueError("Add File lines must start with `+`.")
                    added_lines.append(lines[index][1:])
                    index += 1
                operations.append(
                    _PatchOperation(
                        operation="add",
                        path=path,
                        old_text=None,
                        new_text="\n".join(added_lines) + "\n",
                    )
                )
                continue
            if line.startswith("*** Delete File: "):
                path = self._resolve_patch_path(
                    line.removeprefix("*** Delete File: ")
                )
                old_text, line_ending = self._read_existing_patch_target(path)
                operations.append(
                    _PatchOperation(
                        operation="delete",
                        path=path,
                        old_text=old_text,
                        new_text=None,
                        line_ending=line_ending,
                    )
                )
                index += 1
                continue
            if line.startswith("*** Update File: "):
                path = self._resolve_patch_path(
                    line.removeprefix("*** Update File: ")
                )
                index += 1
                old_text, line_ending = self._read_existing_patch_target(path)
                new_text = old_text
                while index < len(lines) - 1 and not lines[index].startswith("*** "):
                    if lines[index].startswith("@@"):
                        index += 1
                        continue
                    old_block: list[str] = []
                    new_block: list[str] = []
                    while (
                        index < len(lines) - 1
                        and not lines[index].startswith("*** ")
                        and not lines[index].startswith("@@")
                    ):
                        patch_line = lines[index]
                        if patch_line.startswith(" "):
                            old_block.append(patch_line[1:])
                            new_block.append(patch_line[1:])
                        elif patch_line.startswith("-"):
                            old_block.append(patch_line[1:])
                        elif patch_line.startswith("+"):
                            new_block.append(patch_line[1:])
                        else:
                            raise ValueError(
                                "Update File hunk lines must start with space, "
                                "`-`, or `+`."
                            )
                        index += 1
                    old_chunk = "\n".join(old_block)
                    new_chunk = "\n".join(new_block)
                    if not old_chunk:
                        raise ValueError(
                            "Update File hunk must include context or removed lines."
                        )
                    if old_chunk not in new_text:
                        raise ValueError(
                            f"Patch hunk was not found in file: {path}"
                        )
                    new_text = new_text.replace(old_chunk, new_chunk, 1)
                operations.append(
                    _PatchOperation(
                        operation="update",
                        path=path,
                        old_text=old_text,
                        new_text=new_text,
                        line_ending=line_ending,
                    )
                )
                continue
            raise ValueError(f"Unknown patch operation: {line}")

        if not operations:
            raise ValueError("Patch does not contain any file operations.")
        self._validate_unique_patch_paths(operations)
        return operations

    def _validate_unique_patch_paths(self, operations: list[_PatchOperation]) -> None:
        seen_paths: set[Path] = set()
        for operation in operations:
            if operation.path in seen_paths:
                raise ValueError(
                    f"Patch contains multiple operations for file: {operation.path}"
                )
            seen_paths.add(operation.path)

    def _resolve_patch_path(self, path: str) -> Path:
        if not path.strip():
            raise ValueError("Patch file path cannot be empty.")
        return Path(path).expanduser().resolve()

    def _read_existing_patch_target(self, path: Path) -> tuple[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        content, line_ending = self._read_text_preserving_newlines(path)
        read_record = get_file_read_tracker().get(path)
        if read_record is None:
            raise ValueError(
                "File must be read with the Read tool before it can be patched."
            )
        if read_record.text_hash != hash_text(content):
            raise ValueError(
                "File has changed since it was last read. Read the file again "
                "before patching."
            )
        return content, line_ending

    def _build_patch_proposal(self, operation: _PatchOperation) -> FileEditProposal:
        old_content = operation.old_text or ""
        new_content = operation.new_text or ""
        diff = self._build_diff(
            path=operation.path,
            old_content=old_content,
            new_content=new_content,
        )
        diff_for_payload, diff_truncated = self._truncate_diff(diff)
        return FileEditProposal(
            path=str(operation.path),
            operation=operation.operation,
            diff=diff_for_payload,
            old_text_hash=hash_text(old_content),
            new_text_hash=hash_text(new_content),
            lines_added=self._count_diff_lines(diff, "+"),
            lines_removed=self._count_diff_lines(diff, "-"),
            diff_truncated=diff_truncated,
            diff_chars_original=len(diff),
        )

    def _write_patch_operation(self, operation: _PatchOperation) -> None:
        if operation.operation == "delete":
            operation.path.unlink()
            return
        if operation.new_text is None:
            raise ValueError("Patch operation has no new content to write.")
        operation.path.parent.mkdir(parents=True, exist_ok=True)
        operation.path.write_bytes(
            operation.new_text.replace("\n", operation.line_ending).encode("utf-8")
        )

    def _write_patch_operations_atomically(
        self,
        operations: list[_PatchOperation],
    ) -> None:
        snapshots = {
            operation.path: (
                operation.path.exists(),
                operation.path.read_bytes() if operation.path.exists() else None,
            )
            for operation in operations
        }
        written: list[Path] = []
        try:
            for operation in operations:
                self._write_patch_operation(operation)
                written.append(operation.path)
        except Exception:
            # A multi-file patch must be all-or-nothing. If a later write fails
            # after an earlier file was changed, restore every touched path to
            # its original bytes before surfacing the original error.
            for path in reversed(written):
                existed, content = snapshots[path]
                if existed and content is not None:
                    path.write_bytes(content)
                elif path.exists():
                    path.unlink()
            raise

    def _build_diff(self, *, path: Path, old_content: str, new_content: str) -> str:
        return "".join(
            unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{path} (before)",
                tofile=f"{path} (after)",
            )
        )

    def _count_diff_lines(self, diff: str, prefix: str) -> int:
        return sum(
            1
            for line in diff.splitlines()
            if line.startswith(prefix) and not line.startswith(prefix * 3)
        )

    def _truncate_diff(self, diff: str) -> tuple[str, bool]:
        if len(diff) <= self.max_diff_chars:
            return diff, False
        marker = "\n...[diff truncated]...\n"
        return diff[: self.max_diff_chars] + marker, True
