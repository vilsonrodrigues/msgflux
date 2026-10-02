"""Open the coding host's durable workspace with freshly selected permissions."""

from pathlib import Path

from msgflux.runtime import AgentWorkspace, PermissionSet, SandboxRequirements
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry


async def open_coding_workspace(root: Path, checkpoint_path: Path, *, read_only=False):
    """Keep direct host paths and cwd; the coding host does not imply isolation.

    Registry tables share the thread checkpoint database. No credentials or
    permission grants are stored in the registry.
    """
    registry = SQLiteWorkspaceRegistry(str(checkpoint_path))
    try:
        grants = {"filesystem.read", "filesystem.list"}
        if not read_only:
            grants.update(
                {
                    "filesystem.write",
                    "filesystem.delete",
                    "filesystem.mkdir",
                    "process.execute",
                }
            )
        workspace = await AgentWorkspace.open(
            LocalWorkspaceBackend("/", registry=registry, allow_processes=True),
            "coding-host",
            cwd=str(root),
            permissions=PermissionSet(grants),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
    except BaseException:
        registry.close()
        raise
    return workspace, registry
