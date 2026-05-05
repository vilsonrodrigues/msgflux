import asyncio

import pytest

from msgflux.channels import RunManager


@pytest.mark.asyncio
async def test_run_manager_tracks_and_cancels_active_session_task():
    manager = RunManager()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = manager.create_active("session-1", work())

    await started.wait()
    assert manager.active_task("session-1") is task
    assert manager.cancel_session("session-1") is True

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert manager.active_task("session-1") is None


@pytest.mark.asyncio
async def test_run_manager_replaces_pending_session_task_and_items():
    manager = RunManager()
    release = asyncio.Event()

    async def wait_forever():
        await asyncio.Event().wait()

    async def wait_until_released():
        await release.wait()

    first = manager.replace_pending("session-1", wait_forever())
    manager.add_pending_item("session-1", "first")
    await asyncio.sleep(0)
    second = manager.replace_pending("session-1", wait_until_released())
    manager.add_pending_item("session-1", "second")
    await asyncio.sleep(0)

    assert first.cancelled()
    assert not second.done()
    assert manager.pop_pending_items("session-1") == ["first", "second"]

    release.set()
    await second
    manager.forget_pending_if_current("session-1", second)
    assert manager.cancel_session("session-1") is False
