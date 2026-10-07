"""End-to-end approval review through the authenticated native service client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Static, TextArea

from msgflux.coding import CodingCheckpointExtension
from msgflux.coding.tui import CodingApp
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, AgentWorkspace, SQLiteApprovalStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import AgentSessionClient
from msgflux.tools.builtin import EditTool

from tests.coding.test_remote_socket import _server, _until


def _response(value, kind="text_generation"):
    response = ModelResponse()
    response.set_response_type(kind)
    response.add(value)
    return response


def _edit_call():
    calls = ToolCallAggregator()
    calls.process(
        0,
        "edit-call",
        "edit",
        '{"path":"note.txt","old":"before","new":"after"}',
    )
    return calls


def _text(app):
    return "\n".join(
        str(row.content) for row in app.query_one("#transcript").query(Static)
    )


async def _click_control(app, pilot, selector):
    control = app.query_one(selector)
    assert control.display
    assert not control.disabled
    app.query_one("#transcript").scroll_to_widget(control, animate=False)
    await pilot.pause()
    await pilot.click(selector)


def _make_service(tmp_path, *, multi=False, entered=None, release=None):
    target = tmp_path / "note.txt"
    target.write_text("before")
    second_target = tmp_path / "second.txt"
    if multi:
        second_target.write_text("second before")
    checkpoints = InMemoryCheckpointStore()
    approval_store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = AgentApprovals(approval_store, {"edit": "v1"}, "coding-v1")
    workspace = AgentWorkspace.local(tmp_path)
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="main",
        model=model,
        workspace=workspace,
        tools=[EditTool()],
        checkpoint_store=checkpoints,
        approvals=policy,
    )
    if multi:
        calls = _edit_call()
        calls.process(
            1,
            "second-edit-call",
            "edit",
            '{"path":"second.txt","old":"second before","new":"second after"}',
        )
        tool_reply = _response(calls, "tool_call")
    else:
        tool_reply = _response(_edit_call(), "tool_call")
    replies = iter((tool_reply, _response("edit handled")))

    async def respond(**_kwargs):
        if entered is not None and agent.generator.aforward.await_count == 1:
            entered.set()
            await release.wait()
        return next(replies)

    agent.generator.aforward = AsyncMock(side_effect=respond)
    agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            scope_factory=lambda scope: scope.with_overrides(principal="local-user"),
            approval_reviewer="local-user",
        ),
    )
    return service, agent, workspace, target


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_file"),
    [("approve", "after"), ("deny", "before")],
)
async def test_native_pilot_reviews_edit_and_explicitly_resumes_same_run(
    tmp_path, decision, expected_file
):
    service, agent, workspace, target = _make_service(tmp_path)
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            app = CodingApp(session, observe_immediately=False)
            async with app.run_test() as pilot:
                app.query_one("#composer", TextArea).load_text("edit note")
                await pilot.press("enter")
                await _until(lambda: len(app.query(".approval-card")) == 1)
                run_id = next(iter(app._approval_cards))[0]

                reviews = await session.approval_reviews(run_id)
                assert len(reviews) == 1
                review = reviews[0]
                assert review.tool_name == "edit"
                assert review.tool_call_id == "edit-call"
                assert "-before" in review.diff and "+after" in review.diff
                assert target.read_text() == "before"
                card_text = _text(app)
                assert review.request_id in card_text
                assert "-before" in card_text and "+after" in card_text

                await _click_control(app, pilot, f"#{decision}-{review.request_id}")
                await _until(
                    lambda: (
                        app._approval_cards[(run_id, review.request_id)].status.lower()
                        in {"approved", "denied"}
                    )
                )
                assert app._approval_cards[
                    (run_id, review.request_id)
                ].status.lower() == ("approved" if decision == "approve" else "denied")
                assert agent.generator.aforward.await_count == 1

                # Recording a decision never resumes the run. The user has a
                # separate resume action after the review state is settled.
                receipt = service.receipt_for_run(session.thread_id, run_id)
                assert receipt.status == "paused"
                resume = app.query_one(f"#resume-{run_id}")
                assert not resume.disabled
                await _click_control(app, pilot, f"#resume-{run_id}")
                await _until(lambda: not app._run_active)
                settled = await asyncio.wait_for(
                    service.wait(session.thread_id, receipt.request_id), 5
                )
                assert settled.run_id == run_id
                assert settled.status == "completed"
                assert target.read_text() == expected_file
                assert agent.generator.aforward.await_count == 2
    finally:
        await service.aclose()
        service.store.close()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_native_pilot_reopens_pending_approval_and_preserves_external_edit(
    tmp_path,
):
    service, agent, workspace, target = _make_service(tmp_path)
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            first = CodingApp(session, observe_immediately=False)
            async with first.run_test() as pilot:
                first.query_one("#composer", TextArea).load_text("edit note")
                await pilot.press("enter")
                await _until(lambda: len(first.query(".approval-card")) == 1)
                run_id = next(iter(first._approval_cards))[0]
                assert target.read_text() == "before"

            # Closing Textual tears down its watcher; the service owned run and
            # approval journal remain alive while no frontend is attached.
            receipt = service.receipt_for_run(session.thread_id, run_id)
            assert receipt.status == "paused"
            target.write_text("outside edit")
            snapshot = await session.snapshot()
            assert any(item.get("request_id") for item in snapshot.approvals)
            reopened = CodingApp(session)
            async with reopened.run_test() as pilot:
                await _until(lambda: len(reopened.query(".approval-card")) == 1)
                reviews = await session.approval_reviews(run_id)
                assert len(reviews) == 1
                review = reviews[0]
                assert review.status.lower() == "pending"
                assert reopened._approval_cards[(run_id, review.request_id)].diff
                await _click_control(reopened, pilot, f"#approve-{review.request_id}")
                await _until(
                    lambda: (
                        reopened._approval_cards[
                            (run_id, review.request_id)
                        ].status.lower()
                        == "approved"
                    )
                )
                assert target.read_text() == "outside edit"
                await _click_control(reopened, pilot, f"#resume-{run_id}")
                await _until(lambda: not reopened._run_active)
                settled = await asyncio.wait_for(
                    service.wait(session.thread_id, receipt.request_id), 5
                )
                assert target.read_text() == "outside edit"
                assert settled.run_id == run_id
                assert settled.status == "paused"
                assert agent.generator.aforward.await_count == 1
    finally:
        await service.aclose()
        service.store.close()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_native_pilot_waits_for_every_request_before_enabling_resume(tmp_path):
    service, agent, workspace, target = _make_service(tmp_path, multi=True)
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            app = CodingApp(session, observe_immediately=False)
            async with app.run_test() as pilot:
                app.query_one("#composer", TextArea).load_text("edit both files")
                await pilot.press("enter")
                await _until(lambda: len(app.query(".approval-card")) == 2)
                run_id = next(iter(app._approval_cards))[0]
                reviews = await session.approval_reviews(run_id)
                assert len(reviews) == 2
                resume = app.query_one(f"#resume-{run_id}")
                assert resume.disabled

                first, second = reviews
                await _click_control(app, pilot, f"#approve-{first.request_id}")
                await _until(
                    lambda: (
                        app._approval_cards[(run_id, first.request_id)].status.lower()
                        == "approved"
                    )
                )
                assert resume.disabled
                assert target.read_text() == "before"
                assert (tmp_path / "second.txt").read_text() == "second before"
                assert agent.generator.aforward.await_count == 1

                await _click_control(app, pilot, f"#deny-{second.request_id}")
                await _until(
                    lambda: (
                        app._approval_cards[(run_id, second.request_id)].status.lower()
                        == "denied"
                    )
                )
                assert not resume.disabled
                await _click_control(app, pilot, f"#resume-{run_id}")
                await _until(lambda: not app._run_active)
                receipt = service.receipt_for_run(session.thread_id, run_id)
                settled = await asyncio.wait_for(
                    service.wait(session.thread_id, receipt.request_id), 5
                )
                assert settled.run_id == run_id
                assert settled.status == "completed"
                assert target.read_text() == "after"
                assert (tmp_path / "second.txt").read_text() == "second before"
                assert agent.generator.aforward.await_count == 2
    finally:
        await service.aclose()
        service.store.close()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_closing_pilot_does_not_cancel_service_producer_before_approval_pause(
    tmp_path,
):
    model_entered, release_model = asyncio.Event(), asyncio.Event()
    service, agent, workspace, target = _make_service(
        tmp_path, entered=model_entered, release=release_model
    )
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            app = CodingApp(session, observe_immediately=False)
            async with app.run_test() as pilot:
                app.query_one("#composer", TextArea).load_text("edit note")
                await pilot.press("enter")
                await asyncio.wait_for(model_entered.wait(), 5)
                await _until(lambda: app._active_run_id is not None)
                run_id = app._active_run_id
                assert service.receipt_for_run(session.thread_id, run_id).status == (
                    "running"
                )

            release_model.set()
            running = service.receipt_for_run(session.thread_id, run_id)
            settled = await asyncio.wait_for(
                service.wait(session.thread_id, running.request_id), 5
            )
            assert settled.run_id == run_id
            assert settled.status == "paused"
            reviews = await session.approval_reviews(run_id)
            assert len(reviews) == 1
            assert agent.generator.aforward.await_count == 1
            assert target.read_text() == "before"
    finally:
        release_model.set()
        await service.aclose()
        service.store.close()
        await workspace.aclose()
