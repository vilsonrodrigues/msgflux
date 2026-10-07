"""The remote TUI observes independently from prompt admission."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest

from msgflux.coding.tui.app import CodingApp


@dataclass
class Receipt:
    run_id: str


@dataclass
class Snapshot:
    messages: tuple[dict, ...] = ()
    active_runs: tuple[dict, ...] = ()


class FakeWatcher:
    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.events: asyncio.Queue = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.events.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item


class FakeRemoteSession:
    thread_id = "thread-1"
    workspace_root = "/workspace"

    def __init__(self):
        self.watchers: list[FakeWatcher] = []
        self.watch_started = asyncio.Event()
        self.prompt_started = asyncio.Event()
        self.finish_prompt = asyncio.Event()
        self.prompt_calls: list[tuple[str, str]] = []
        self.cancel_calls: list[str] = []
        self.steer_calls: list[tuple[str, str]] = []
        self.next_snapshot = Snapshot()

    @asynccontextmanager
    async def watch(self):
        watcher = FakeWatcher(self.next_snapshot)
        self.watchers.append(watcher)
        self.watch_started.set()
        try:
            yield watcher
        finally:
            watcher.closed = True

    async def prompt(self, text, *, request_id):
        self.prompt_calls.append((text, request_id))
        self.prompt_started.set()
        await self.finish_prompt.wait()
        return Receipt("root-run")

    async def steer(self, run_id, text):
        self.steer_calls.append((run_id, text))
        return None

    async def cancel(self, run_id):
        self.cancel_calls.append(run_id)
        return True

    async def latest_run(self):
        return None

    async def runs(self):
        return ()

    async def resume(self, run_id):
        return Receipt(run_id)


async def wait_for(predicate, *, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_observer_attaches_before_delayed_prompt_admission_and_keeps_receiving():
    session = FakeRemoteSession()
    app = CodingApp(session, observe_immediately=False)

    async with app.run_test() as pilot:
        await pilot.click("#composer")
        await pilot.press("h", "i", "enter")
        await asyncio.wait_for(session.watch_started.wait(), timeout=2)
        await asyncio.wait_for(session.prompt_started.wait(), timeout=2)
        assert len(session.prompt_calls) == 1
        assert session.prompt_calls[0][1]

        # Admission stays blocked while the independent watcher still renders.
        await session.watchers[0].events.put(
            {"type": "run.start", "run_id": "root-run", "data": {}}
        )
        await session.watchers[0].events.put(
            {
                "type": "message.delta",
                "run_id": "root-run",
                "data": {"delta": "partial"},
            }
        )
        await wait_for(lambda: app._assistant_text == "partial")
        assert not session.finish_prompt.is_set()

        session.finish_prompt.set()
        await wait_for(lambda: app._active_run_id == "root-run")
        assert len(session.prompt_calls) == 1


@pytest.mark.asyncio
async def test_reconnect_replaces_transcript_from_fresh_snapshot():
    session = FakeRemoteSession()
    app = CodingApp(session)

    async with app.run_test():
        await asyncio.wait_for(session.watch_started.wait(), timeout=2)
        await session.watchers[0].events.put(StopAsyncIteration)
        await wait_for(lambda: len(session.watchers) >= 2)
        session.next_snapshot = Snapshot(
            messages=(
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "tool", "content": "hidden"},
            )
        )
        # The second watch may have captured its snapshot before assignment; force
        # another EOF so its next connection obtains the prepared snapshot.
        await session.watchers[1].events.put(StopAsyncIteration)
        await wait_for(lambda: len(session.watchers) >= 3)
        await wait_for(lambda: len(app.query_one("#transcript").children) == 2)
        rendered = [str(row.content) for row in app.query_one("#transcript").children]
        assert any("User: question" in line for line in rendered)
        assert any("Assistant: answer" in line for line in rendered)
        assert all("hidden" not in line for line in rendered)


@pytest.mark.asyncio
async def test_close_does_not_cancel_the_service_run():
    session = FakeRemoteSession()
    session.finish_prompt.set()
    app = CodingApp(session, observe_immediately=False)

    async with app.run_test() as pilot:
        await pilot.click("#composer")
        await pilot.press("h", "i", "enter")
        await asyncio.wait_for(session.prompt_started.wait(), timeout=2)
        await session.watchers[0].events.put(
            {
                "type": "run.start",
                "run_id": "root-run",
                "source_path": ("main",),
                "data": {},
            }
        )
        await wait_for(lambda: app._active_run_id == "root-run")
        assert session.cancel_calls == []
        await pilot.press("ctrl+c")

    assert session.cancel_calls == []


@pytest.mark.asyncio
async def test_native_snapshot_partial_and_message_end_update_one_root_response():
    from msgflux.runtime.service.http.records import EventRecord

    session = FakeRemoteSession()
    session.next_snapshot = Snapshot(
        messages=(
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "channel": "commentary",
                "content": "thinking out loud",
            },
            {"role": "assistant", "channel": "analysis", "content": "secret reasoning"},
            {"role": "assistant", "channel": "final", "content": "prior answer"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "tool"}]},
        ),
        active_runs=(
            {
                "run_id": "root-run",
                "source_path": ["main"],
                "streaming_message": "partial",
            },
            {
                "run_id": "child-run",
                "source_path": ["main", "delegate"],
                "streaming_message": "child partial",
            },
        ),
    )
    app = CodingApp(session)

    async with app.run_test():
        await wait_for(lambda: app._active_run_id == "root-run")
        assert app._assistant_text == "partial"
        rendered = [str(row.content) for row in app.query_one("#transcript").children]
        assert any("Progress: thinking out loud" in row for row in rendered)
        assert any("Assistant: prior answer" in row for row in rendered)
        assert all("secret reasoning" not in row for row in rendered)
        await session.watchers[0].events.put(
            EventRecord(
                type="message.end",
                timestamp="2026-10-07T12:00:00Z",
                data={"content": "partial complete"},
                run_id="root-run",
                source_path=("main",),
            )
        )
        await wait_for(
            lambda: any(
                "partial complete" in str(row.content)
                for row in app.query_one("#transcript").children
            )
        )
        assistant_rows = [
            row
            for row in app.query_one("#transcript").children
            if "assistant-message" in row.classes
        ]
        assert len(assistant_rows) == 2
        final_rows = [row for row in assistant_rows if "partial" in str(row.content)]
        assert len(final_rows) == 1
        assert str(final_rows[0].content) == "Assistant: partial complete"
        assert app._active_run_id == "root-run"
        await session.watchers[0].events.put(
            EventRecord(
                type="run.end",
                timestamp="2026-10-07T12:00:01Z",
                data={"outcome": "completed"},
                run_id="child-run",
                source_path=("main", "delegate"),
            )
        )
        await asyncio.sleep(0)
        assert app._active_run_id == "root-run"
        await session.watchers[0].events.put(
            EventRecord(
                type="run.end",
                timestamp="2026-10-07T12:00:02Z",
                data={"outcome": "failed"},
                run_id="root-run",
                source_path=("main",),
            )
        )
        await wait_for(lambda: app._active_run_id is None)
        assert str(app.query_one("#status").content) == "Run failed"


@pytest.mark.asyncio
async def test_fast_run_end_before_prompt_receipt_does_not_resurrect_busy_state():
    session = FakeRemoteSession()
    app = CodingApp(session, observe_immediately=False)

    async with app.run_test() as pilot:
        await pilot.click("#composer")
        await pilot.press("f", "i", "r", "s", "t", "enter")
        await asyncio.wait_for(session.prompt_started.wait(), timeout=2)
        await session.watchers[0].events.put(
            {
                "type": "run.start",
                "run_id": "root-run",
                "source_path": ("main",),
                "data": {},
            }
        )
        await session.watchers[0].events.put(
            {
                "type": "run.end",
                "run_id": "root-run",
                "source_path": ("main",),
                "data": {"outcome": "completed"},
            }
        )
        await wait_for(lambda: app._active_run_id is None and not app._run_active)
        session.finish_prompt.set()
        await wait_for(
            lambda: app._admission_task is not None and app._admission_task.done()
        )

        assert app._active_run_id is None
        assert not app._run_active
        assert str(app.query_one("#status").content) == "Ready"

        await pilot.press("s", "e", "c", "o", "n", "d", "enter")
        await wait_for(lambda: len(session.prompt_calls) == 2)
        assert session.steer_calls == []
