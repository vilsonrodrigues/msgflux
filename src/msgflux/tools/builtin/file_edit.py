from msgflux.runtime.file_edit import (
    DEFAULT_MAX_FILE_EDIT_DIFF_CHARS,
    FileEditRuntime,
)

FILE_EDIT_TOOL_NAME = "edit"


class FileEdit:
    """Edit a text file by replacing an exact string."""

    name = FILE_EDIT_TOOL_NAME
    display_name = "Edit"
    description = (
        "Edit a text file by replacing an exact old_string with new_string. "
        "Use replace_all only when every occurrence should be replaced."
    )
    read_only = False
    concurrency_safe = False

    def __init__(
        self,
        runtime: FileEditRuntime | None = None,
        *,
        max_diff_chars: int = DEFAULT_MAX_FILE_EDIT_DIFF_CHARS,
    ) -> None:
        self.runtime = runtime or FileEditRuntime(
            tool_name=self.name,
            max_diff_chars=max_diff_chars,
        )

    def __call__(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> str:
        """Edit a text file.

        Args:
            file_path: Path to the file to edit.
            old_string: Exact text to replace.
            new_string: Replacement text.
            replace_all: Replace every occurrence instead of exactly one.
        """
        _ = (file_path, old_string, new_string, replace_all)
        raise RuntimeError("Edit is async-only. Use `agent.acall(...)`.")

    async def acall(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> str:
        return await self.runtime.replace(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )


Edit = FileEdit
