"""Exercise deferred thread persistence with the existing SQLite session factory."""

import pytest

from msgflux.coding.tui import CodingApp
from tests.coding.test_reopen import command, make_host


@pytest.mark.asyncio
async def test_open_close_and_new_commands_do_not_persist_empty_threads(tmp_path):
    host, _calls, models, closed = make_host(tmp_path)
    host._lazy_new = True
    try:
        session, controller = await host.select()
        original_id = session.thread_id
        assert controller is None
        assert (await session.snapshot()).messages is None
        assert session.runs() == ()
        app = CodingApp(session, host=host)
        async with app.run_test() as pilot:
            await pilot.pause()
            await command(app, "/help")
            await command(app, "/session")
            await command(app, "/new")
            assert app.coding_session.thread_id != original_id
            assert models == []
            assert not host.storage.threads_dir.exists()
    finally:
        await host.aclose()
    assert closed == []
    assert not host.storage.threads_dir.exists()


@pytest.mark.asyncio
async def test_first_prompt_persists_once_and_reopens_history(tmp_path):
    host, _calls, models, closed = make_host(tmp_path)
    host._lazy_new = True
    session, _ = await host.select()
    try:
        events = [event async for event in session.stream("first request")]
        assert events
        assert len(models) == 1
        assert models[0].call_count == 1
        metadata = host.storage.read_metadata(session.thread_id)
        assert metadata.workspace == host.workspace
        assert host.storage.checkpoint_path(session.thread_id).is_file()
        assert host.storage.approval_path(session.thread_id).is_file()
        assert session.latest_run()["status"] == "completed"
        await host.select(session.thread_id)
        assert "first request" in str((await host.session.snapshot()).messages)
        assert len(models) == 2
        assert models[1].call_count == 0
        assert closed == [session.thread_id]
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_first_prompt_installs_approval_controller(tmp_path):
    host, _calls, _models, _closed = make_host(tmp_path, protected=True)
    host._lazy_new = True
    try:
        session, controller = await host.select()
        assert controller is None
        assert host.approval_controller is None
        [event async for event in session.stream("first request")]
        assert host.approval_controller is not None
    finally:
        await host.aclose()
