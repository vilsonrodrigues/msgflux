"""The default backend resolves workspace and permissions lazily, without RPC."""

import pytest

from msgflux.coding.remote_backend import create_service


@pytest.mark.asyncio
async def test_blank_thread_is_lazy_and_read_only_profile_denies_writes_and_processes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    state = tmp_path / "state"
    runtime = state / "runtime"
    runtime.mkdir(parents=True)
    (state / "config.toml").write_text('default_model = "openai/gpt-6-luna"\n')
    workspace = tmp_path / "project"
    workspace.mkdir()
    service = create_service(runtime)
    try:
        thread = await service.open_thread("lite:read-only", cwd=workspace)
        assert not (state / "threads").exists()
        session = await service.session(thread.thread_id)
        assert session.agent.workspace.permissions.missing(("process.execute",))
        assert session.agent.workspace.permissions.missing(("filesystem.write",))
        assert session.agent.workspace.cwd == str(workspace)
        assert "bash" not in session.agent.tool_library.get_tool_names()
        with pytest.raises(PermissionError):
            await session.agent.workspace.awrite_text("forbidden.txt", "no")
        assert (state / "threads" / thread.thread_id / "checkpoints.sqlite3").exists()
    finally:
        await service.aclose()
        service.store.close()
