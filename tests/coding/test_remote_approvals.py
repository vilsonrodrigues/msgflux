"""Remote approval controls stay bound to the server session and run."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from msgflux.coding.tui.app import CodingApp
from msgflux.coding.tui.approval import ApprovalCard
from msgflux.runtime.service.records import ApprovalReview


def review(request_id, tool_call_id, tool_name, *, status="pending", revision=1):
    return ApprovalReview(
        request_id=request_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=status,
        revision=revision,
        expires_at=9999999999,
        diff="verified diff",
    )


class Watcher:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.events = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.events.get()
        if item is StopAsyncIteration:
            raise StopAsyncIteration
        return item


class Remote:
    thread_id = "thread-1"
    workspace_root = "/workspace"

    def __init__(self):
        self.watchers = []
        self.reviews = {}
        self.decisions = []
        self.resumes = []
        self.review_gate = None
        self.next_snapshot = {"messages": (), "active_runs": (), "approvals": ()}

    @asynccontextmanager
    async def watch(self):
        watcher = Watcher(self.next_snapshot)
        self.watchers.append(watcher)
        yield watcher

    async def approval_reviews(self, run_id):
        if self.review_gate is not None:
            await self.review_gate.wait()
        return tuple(self.reviews.get(run_id, ()))

    async def decide_approval(self, run_id, request_id, *, approved, expected_revision):
        self.decisions.append((run_id, request_id, approved, expected_revision))
        existing = next(
            item for item in self.reviews[run_id] if item.request_id == request_id
        )
        updated = review(
            existing.request_id,
            existing.tool_call_id,
            existing.tool_name,
            status="approved" if approved else "denied",
            revision=expected_revision + 1,
        )
        self.reviews[run_id] = [
            updated if item.request_id == request_id else item
            for item in self.reviews[run_id]
        ]
        return updated

    async def resume(self, run_id):
        self.resumes.append(run_id)
        return object()


async def wait_for(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_multiple_reviews_decide_without_auto_resume_then_explicit_resume():
    remote = Remote()
    remote.reviews["run-1"] = [
        review("req-a", "call-a", "write_file"),
        review("req-b", "call-b", "run_shell"),
    ]
    app = CodingApp(remote)

    async with app.run_test() as pilot:
        await wait_for(lambda: remote.watchers)
        await remote.watchers[0].events.put(
            {
                "type": "run.start",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await remote.watchers[0].events.put(
            {
                "type": "tool.approval_required",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await remote.watchers[0].events.put(
            {
                "type": "run.paused",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await wait_for(lambda: len(app._approval_cards) == 2)
        await wait_for(
            lambda: app.query_one("#approval-req-a", ApprovalCard).is_mounted
        )
        await wait_for(lambda: app.query_one("#resume-run-1").is_mounted)
        assert "verified diff" in str(app.query_one("#approval-content-req-a").content)
        assert app.query_one("#resume-run-1").disabled
        app.query_one("#approve-req-a").focus()
        await pilot.press("enter")
        await wait_for(lambda: len(remote.decisions) == 1)
        app.query_one("#deny-req-b").focus()
        await pilot.press("enter")
        await wait_for(lambda: len(remote.decisions) == 2)
        await wait_for(
            lambda: (
                app._approval_cards[("run-1", "req-a")].status == "approved"
                and app._approval_cards[("run-1", "req-b")].status == "denied"
            )
        )
        await wait_for(lambda: not app.query_one("#resume-run-1").disabled)
        assert remote.resumes == []
        app.query_one("#resume-run-1").focus()
        await pilot.press("enter")
        await wait_for(lambda: remote.resumes == ["run-1"])
        assert remote.decisions == [
            ("run-1", "req-a", True, 1),
            ("run-1", "req-b", False, 1),
        ]


@pytest.mark.asyncio
async def test_snapshot_restores_reviews_and_reconnect_deduplicates_cards():
    remote = Remote()
    remote.reviews["run-1"] = [review("req-a", "call-a", "write_file")]
    app = CodingApp(remote)

    async with app.run_test():
        await wait_for(lambda: remote.watchers)
        remote.next_snapshot = {
            "messages": (),
            "active_runs": (),
            "approvals": (
                {
                    "request_id": "req-a",
                    "binding": {
                        "run_id": "run-1",
                        "tool_call_id": "call-a",
                        "tool_name": "write_file",
                    },
                },
            ),
        }
        await remote.watchers[0].events.put(StopAsyncIteration)
        await wait_for(lambda: len(remote.watchers) > 1)
        await wait_for(lambda: len(app._approval_cards) == 1)
        await wait_for(
            lambda: app.query_one("#approval-req-a", ApprovalCard).is_mounted
        )
        await remote.watchers[1].events.put(
            {
                "type": "run.paused",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await asyncio.sleep(0.05)
        assert len(app._approval_cards) == 1


@pytest.mark.asyncio
async def test_late_review_response_after_session_switch_is_discarded():
    remote = Remote()
    remote.reviews["run-1"] = [review("req-a", "call-a", "write_file")]
    remote.review_gate = asyncio.Event()
    app = CodingApp(remote)

    async with app.run_test():
        await wait_for(lambda: remote.watchers)
        await remote.watchers[0].events.put(
            {
                "type": "run.start",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await remote.watchers[0].events.put(
            {
                "type": "tool.approval_required",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await remote.watchers[0].events.put(
            {
                "type": "run.paused",
                "run_id": "run-1",
                "source_path": ["main"],
                "data": {},
            }
        )
        await wait_for(lambda: "run-1" in app._approval_refresh_tasks)
        app._session_generation += 1
        remote.review_gate.set()
        await asyncio.sleep(0.05)
        assert app._approval_cards == {}


@pytest.mark.asyncio
async def test_review_mount_finishing_after_view_replacement_is_removed(monkeypatch):
    remote = Remote()
    app = CodingApp(remote)
    entered, release = asyncio.Event(), asyncio.Event()
    async with app.run_test():
        await wait_for(lambda: remote.watchers)
        transcript = app.query_one("#transcript")
        original_mount = transcript.mount

        async def delayed_mount(widget):
            entered.set()
            await release.wait()
            await original_mount(widget)

        monkeypatch.setattr(transcript, "mount", delayed_mount)
        task = asyncio.create_task(
            app._upsert_approval("run-1", review("req-a", "call-a", "edit"))
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            app._session_generation += 1
            await app._replace_from_snapshot({"messages": (), "active_runs": ()})
            release.set()
            await asyncio.wait_for(task, 2)
            assert app._approval_cards == {}
            assert not app.query(ApprovalCard)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
