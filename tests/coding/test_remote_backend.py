"""The default backend resolves workspace and permissions lazily, without RPC."""

import sqlite3
from unittest.mock import Mock

import pytest

from msgflux.coding.config import CodingProfile, load_config
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
        assert not (state / "threads" / thread.thread_id / "approvals.sqlite3").exists()
    finally:
        await service.aclose()
        service.store.close()


def test_profile_approval_names_must_be_nonempty_and_unique(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[profiles.review]\napprovals = ["apply_patch", "apply_patch"]\n')
    with pytest.raises(ValueError, match="duplicate approval"):
        load_config(config)
    config.write_text('[profiles.review]\napprovals = ["", "apply_patch"]\n')
    with pytest.raises(ValueError, match="empty approval"):
        load_config(config)
    assert CodingProfile().approvals == ()


@pytest.mark.asyncio
async def test_approval_backend_uses_server_owner_and_closes_its_journal(
    tmp_path, monkeypatch
):
    from msgflux.coding import remote_backend

    state = tmp_path / "state"
    state.mkdir()
    (state / "runtime").mkdir()
    (state / "config.toml").write_text(
        'default_model = "openai/gpt-6-luna"\n'
        'default_profile = "review"\n'
        '[profiles.review]\napprovals = ["apply_patch"]\n'
        '[profiles.review.tools]\nactive = ["workspace"]\n'
    )
    project = tmp_path / "project"
    project.mkdir()
    model = Mock(model_type="chat_completion")
    model.provider = "openai"
    model.api_mode = "chat_completions"
    model.native_tools = False
    model.profile = None
    model.supports_reasoning_effort.return_value = False
    monkeypatch.setattr(
        remote_backend.Model, "chat_completion", lambda *_a, **_k: model
    )

    service = create_service(state / "runtime")
    try:
        thread = await service.open_thread("review", cwd=project)
        session = await service.session(thread.thread_id)
        journal_path = state / "threads" / thread.thread_id / "approvals.sqlite3"
        assert session.agent.approvals is not None
        assert session.agent.approvals.tools.keys() == {"apply_patch"}
        assert session.approval_reviewer == "local-user"
        assert session.scope(thread.thread_id).principal == "local-user"
        assert journal_path.exists()
        journal = session.agent.approvals.store
        readonly = await service.open_thread("review:read-only", cwd=project)
        readonly_session = await service.session(readonly.thread_id)
        assert readonly_session.agent.approvals is None
        assert "apply_patch" not in readonly_session.agent.tool_library.get_tool_names()
        assert not (
            state / "threads" / readonly.thread_id / "approvals.sqlite3"
        ).exists()
        await service.aclose()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            journal._conn.execute("SELECT 1")
    finally:
        await service.aclose()
        service.store.close()
