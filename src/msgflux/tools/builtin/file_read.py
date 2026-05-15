from pathlib import Path

from msgflux.runtime.events import emit_file_read

FILE_READ_TOOL_NAME = "Read"

DEFAULT_FILE_READ_LIMIT = 2_000
DEFAULT_FILE_READ_MAX_CHARS = 120_000
HARD_FILE_READ_MAX_CHARS = 250_000

BLOCKED_DEVICE_PATHS = {
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/full",
    "/dev/stdin",
    "/dev/tty",
    "/dev/console",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/fd/0",
    "/dev/fd/1",
    "/dev/fd/2",
}

IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


class FileRead:
    """Read a text file with line and character limits."""

    name = FILE_READ_TOOL_NAME
    display_name = "Read"
    description = (
        "Read a text file. Use offset and limit to inspect large files in "
        "smaller chunks. The tool returns line-numbered content and truncates "
        "large outputs."
    )
    read_only = True
    concurrency_safe = True

    def __init__(
        self,
        *,
        default_limit: int = DEFAULT_FILE_READ_LIMIT,
        default_max_chars: int = DEFAULT_FILE_READ_MAX_CHARS,
        hard_max_chars: int = HARD_FILE_READ_MAX_CHARS,
    ) -> None:
        if default_limit <= 0:
            raise ValueError("FileRead default_limit must be greater than zero.")
        if default_max_chars <= 0:
            raise ValueError("FileRead default_max_chars must be greater than zero.")
        if hard_max_chars <= 0:
            raise ValueError("FileRead hard_max_chars must be greater than zero.")
        self.default_limit = default_limit
        self.default_max_chars = min(default_max_chars, hard_max_chars)
        self.hard_max_chars = hard_max_chars

    def __call__(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Read a text file.

        Args:
            file_path: Path to the file to read.
            offset: 1-based line number to start reading from.
            limit: Maximum number of lines to return.
            max_chars: Maximum number of characters to return.
        """
        resolved_path = self._resolve_path(file_path)
        self._validate_readable_path(resolved_path)

        line_start = 1 if offset is None else offset
        line_limit = self.default_limit if limit is None else limit
        requested_max_chars = self.default_max_chars if max_chars is None else max_chars
        char_limit = min(requested_max_chars, self.hard_max_chars)
        self._validate_limits(
            line_start=line_start,
            limit=line_limit,
            max_chars=char_limit,
        )

        content = resolved_path.read_bytes()
        self._ensure_text_content(resolved_path, content)
        text = content.decode("utf-8")
        lines = text.splitlines()

        start_index = min(line_start - 1, len(lines))
        selected_lines = lines[start_index : start_index + line_limit]

        rendered_lines = []
        chars_used = 0
        truncated_by_chars = False
        for line_number, line in enumerate(selected_lines, start=line_start):
            rendered = f"{line_number} | {line}"
            projected_chars = chars_used + len(rendered) + 1
            if projected_chars > char_limit:
                truncated_by_chars = True
                break
            rendered_lines.append(rendered)
            chars_used = projected_chars

        if rendered_lines:
            line_end = line_start + len(rendered_lines) - 1
        else:
            line_end = line_start - 1
        truncated_by_lines = start_index + len(selected_lines) < len(lines)
        truncated = truncated_by_chars or truncated_by_lines
        reason = self._truncation_reason(
            by_chars=truncated_by_chars,
            by_lines=truncated_by_lines,
        )

        emit_file_read(
            path=str(resolved_path),
            line_start=line_start,
            line_end=line_end,
            lines_returned=len(rendered_lines),
            chars_returned=sum(len(line) + 1 for line in rendered_lines),
            truncated=truncated,
            reason=reason,
        )

        return self._format_result(
            path=resolved_path,
            line_start=line_start,
            line_end=line_end,
            rendered_lines=rendered_lines,
            truncated=truncated,
            reason=reason,
            next_offset=line_end + 1 if truncated and line_end >= line_start else None,
        )

    async def acall(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        return self(
            file_path=file_path,
            offset=offset,
            limit=limit,
            max_chars=max_chars,
        )

    def _resolve_path(self, path: str) -> Path:
        if not path.strip():
            raise ValueError("FileRead `file_path` cannot be empty.")
        return Path(path).expanduser().resolve()

    def _validate_readable_path(self, path: Path) -> None:
        path_text = str(path)
        if path_text in BLOCKED_DEVICE_PATHS or self._is_stdio_fd_path(path_text):
            raise ValueError(f"Refusing to read blocked device path: {path_text}")
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_text}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path_text}")
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            raise ValueError(
                "File appears to be an image. FileRead only reads text files."
            )

    def _validate_limits(self, *, line_start: int, limit: int, max_chars: int) -> None:
        if line_start <= 0:
            raise ValueError("FileRead offset must be greater than zero.")
        if limit <= 0:
            raise ValueError("FileRead limit must be greater than zero.")
        if max_chars <= 0:
            raise ValueError("FileRead max_chars must be greater than zero.")

    def _ensure_text_content(self, path: Path, content: bytes) -> None:
        if b"\x00" in content[:8192]:
            raise ValueError(f"File appears to be binary: {path}")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"File is not valid UTF-8 text: {path}") from exc

    def _format_result(
        self,
        *,
        path: Path,
        line_start: int,
        line_end: int,
        rendered_lines: list[str],
        truncated: bool,
        reason: str | None,
        next_offset: int | None,
    ) -> str:
        if rendered_lines:
            header = f"Read {path} lines {line_start}-{line_end}."
        else:
            header = f"Read {path}; no lines returned."
        if truncated and next_offset is not None:
            header += (
                f" Output truncated by {reason}. "
                f"Use offset={next_offset} to continue."
            )
        elif truncated:
            header += f" Output truncated by {reason}."
        body = "\n".join(rendered_lines)
        return f"{header}\n\n{body}" if body else header

    def _truncation_reason(self, *, by_chars: bool, by_lines: bool) -> str | None:
        if by_chars and by_lines:
            return "max_chars and limit"
        if by_chars:
            return "max_chars"
        if by_lines:
            return "limit"
        return None

    def _is_stdio_fd_path(self, path: str) -> bool:
        if not path.startswith("/proc/") or "/fd/" not in path:
            return False
        return (
            path.endswith("/fd/0")
            or path.endswith("/fd/1")
            or path.endswith("/fd/2")
        )
