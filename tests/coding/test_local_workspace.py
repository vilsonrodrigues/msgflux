"""Real coding workspace lifecycle, host paths and authority."""

import asyncio
import sys

import pytest

from msgflux.coding.workspace import open_coding_workspace
from msgflux.exceptions import AbortRequestedError
from msgflux.runtime import AbortSignal, ExecutionScope, execution_context


@pytest.mark.asyncio
async def test_local_workspace_host_paths_and_identity_survive_reopen(tmp_path):
    path = tmp_path / "workspace.sqlite3"
    first, registry = await open_coding_workspace(tmp_path, path)
    identity = first.identity
    try:
        first.write_text("created.txt", "created")
        assert first.read_text(str(tmp_path / "created.txt")) == "created"
        assert first.cwd == str(tmp_path)
    finally:
        await first.aclose()
        registry.close()
    second, registry = await open_coding_workspace(tmp_path, path)
    try:
        assert second.identity == identity
        assert second.read_text("created.txt") == "created"
    finally:
        await second.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_local_executor_streams_output(tmp_path):
    workspace, registry = await open_coding_workspace(
        tmp_path, tmp_path / "store.sqlite3"
    )
    chunks = []

    async def output(channel, data):
        chunks.append((channel, data))

    try:
        result = await workspace.arun(
            (sys.executable, "-c", "print('ok')"), on_output=output
        )
        assert result.returncode == 0
        assert (
            b"".join(data for channel, data in chunks if channel == "stdout") == b"ok\n"
        )
    finally:
        await workspace.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_local_executor_cancellation_terminates_child(tmp_path):
    workspace, registry = await open_coding_workspace(
        tmp_path, tmp_path / "store.sqlite3"
    )
    signal = AbortSignal()

    async def abort():
        await asyncio.sleep(0.1)
        signal.abort("test cancellation")

    try:
        with execution_context(
            scope=ExecutionScope(workspace=workspace, abort_signal=signal)
        ):
            task = asyncio.create_task(abort())
            with pytest.raises(AbortRequestedError):
                await workspace.arun(
                    (sys.executable, "-c", "import time; time.sleep(30)")
                )
            await task
    finally:
        await workspace.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_read_only_workspace_disallows_process_and_write(tmp_path):
    workspace, registry = await open_coding_workspace(
        tmp_path, tmp_path / "store.sqlite3", read_only=True
    )
    try:
        assert not workspace.permissions.allows(("process.execute",))
        with pytest.raises(PermissionError):
            await workspace.arun((sys.executable, "-c", "print('blocked')"))
        with pytest.raises(PermissionError):
            workspace.write_text("new.txt", "blocked")
        assert not (tmp_path / "new.txt").exists()
    finally:
        await workspace.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_resume_with_reduced_grants_keeps_resource_identity(tmp_path):
    path = tmp_path / "workspace.sqlite3"
    first, registry = await open_coding_workspace(tmp_path, path)
    identity = first.identity
    await first.aclose()
    registry.close()
    read_only, registry = await open_coding_workspace(tmp_path, path, read_only=True)
    try:
        assert read_only.identity == identity
        with pytest.raises(PermissionError):
            read_only.write_text("blocked.txt", "no")
    finally:
        await read_only.aclose()
        registry.close()
