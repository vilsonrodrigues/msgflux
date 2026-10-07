"""The remote entry point constructs the actual Textual app."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from msgflux.coding import remote_cli, tui
from msgflux.coding.tui import CodingApp


@pytest.mark.asyncio
async def test_cli_builds_remote_app_without_embedded_approval_controller(
    tmp_path, monkeypatch
):
    client = SimpleNamespace(aclose=AsyncMock())
    session = SimpleNamespace(thread_id="thread", workspace_root=str(tmp_path))
    host = SimpleNamespace(
        select=AsyncMock(return_value=(session, None)),
        aclose=AsyncMock(),
        is_new=True,
        workspace=str(tmp_path),
    )
    connect = AsyncMock(return_value=client)
    monkeypatch.setattr(remote_cli, "connect_local_service", connect)
    monkeypatch.setattr(remote_cli, "RemoteCodingHost", lambda *_args, **_kwargs: host)
    apps = []

    class App(CodingApp):
        async def run_async(self):
            apps.append(self)

    monkeypatch.setattr(tui, "CodingApp", App)
    args = remote_cli._parser().parse_args(
        ["--state-dir", str(tmp_path), "--workspace", str(tmp_path)]
    )
    await remote_cli._run(args)
    assert len(apps) == 1
    assert apps[0].coding_session is session
    assert not apps[0]._observe_immediately
    host.aclose.assert_awaited_once()
    client.aclose.assert_awaited_once()
