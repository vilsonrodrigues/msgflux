from msgflux.runtime.file_edit import (
    DEFAULT_MAX_FILE_EDIT_DIFF_CHARS,
    FileEditRuntime,
)

APPLY_PATCH_TOOL_NAME = "apply_patch"


class ApplyPatch:
    """Apply a Codex-style patch to one or more files."""

    name = APPLY_PATCH_TOOL_NAME
    display_name = "Apply Patch"
    description = (
        "Apply a Codex-style patch. Use this for multi-file edits or when a "
        "patch is clearer than an exact string replacement."
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

    def __call__(self, patch: str) -> str:
        """Apply a Codex-style patch.

        Args:
            patch: Patch text starting with `*** Begin Patch` and ending with
                `*** End Patch`.
        """
        _ = patch
        raise RuntimeError("apply_patch is async-only. Use `agent.acall(...)`.")

    async def acall(self, patch: str) -> str:
        return await self.runtime.apply_patch(patch=patch)
