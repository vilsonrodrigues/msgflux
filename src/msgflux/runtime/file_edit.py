from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path
from typing import Literal

from msgflux.context import get_execution_context
from msgflux.runtime.events import (
    emit_file_edit_applied,
    emit_file_edit_failed,
    emit_file_edit_proposed,
    emit_file_edit_rejected,
)
from msgflux.runtime.permissions import (
    PermissionDeniedError,
    PermissionManager,
    PermissionRequest,
    PermissionRuntimeError,
)

FileEditOperation = Literal["replace"]


@dataclass(frozen=True)
class FileEditProposal:
    path: str
    operation: FileEditOperation
    diff: str
    old_text_hash: str
    new_text_hash: str
    lines_added: int
    lines_removed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "operation": self.operation,
            "diff": self.diff,
            "old_text_hash": self.old_text_hash,
            "new_text_hash": self.new_text_hash,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
        }


class FileEditRuntime:
    """Shared file-edit runtime for tools that propose and apply text changes."""

    def __init__(self, *, tool_name: str = "Edit") -> None:
        self.tool_name = tool_name

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
            proposal, new_content = self._build_replace_proposal(
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )
            emit_file_edit_proposed(proposal.to_dict())
            await self._enforce_permission(proposal)
            path.write_text(new_content, encoding="utf-8")
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

    def _build_replace_proposal(
        self,
        *,
        path: Path,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> tuple[FileEditProposal, str]:
        if not old_string:
            raise ValueError("Edit `old_string` cannot be empty.")
        if old_string == new_string:
            raise ValueError("Edit `old_string` and `new_string` are identical.")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        old_content = path.read_text(encoding="utf-8")
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
        return (
            FileEditProposal(
                path=str(path),
                operation="replace",
                diff=diff,
                old_text_hash=self._hash_text(old_content),
                new_text_hash=self._hash_text(new_content),
                lines_added=self._count_diff_lines(diff, "+"),
                lines_removed=self._count_diff_lines(diff, "-"),
            ),
            new_content,
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

    def _hash_text(self, text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
