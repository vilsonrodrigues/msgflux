"""Exercise real durable session restoration and keyboard-driven UI commands."""

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Static, TextArea

from msgflux.coding.approval import CodingApprovalController
from msgflux.coding.extensions import CodingExtensions
from msgflux.coding.host import CodingHost
from msgflux.coding.session import CodingSession
from msgflux.coding.storage import ThreadStorage
from msgflux.coding.tui import CodingApp
from msgflux.coding.tui.pickers import CodingPicker
from msgflux.coding.workspace import open_coding_workspace
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals


def response(text):
    value = ModelResponse()
    value.set_response_type("text_generation")
    value.add(text)
    return value


def tool_response():
    calls = ToolCallAggregator()
    calls.process(0, "lookup-call", "lookup", '{"query":"cobalt"}')
    value = ModelResponse()
    value.set_response_type("tool_call")
    value.add(calls)
    return value


def make_host(tmp_path, *, protected=False):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    storage = ThreadStorage(tmp_path / "state")
    calls, models, closed = [], [], []

    async def factory(thread_id):
        store = storage.open_checkpoint_store(thread_id)
        journal = storage.open_approval_store(thread_id)
        workspace, registry = await open_coding_workspace(
            project, storage.checkpoint_path(thread_id)
        )

        def lookup(query: str) -> str:
            """Look up a verified word."""
            calls.append(query)
            return "verified: cobalt"

        policy = (
            AgentApprovals(journal, {"lookup": "v1"}, "test-policy")
            if protected
            else None
        )
        agent = Agent(
            name="main",
            model=Mock(model_type="chat_completion"),
            tools=[lookup],
            workspace=workspace,
            checkpoint_store=store,
            approvals=policy,
        )
        agent.generator.aforward = AsyncMock(return_value=response("restored reply"))
        models.append(agent.generator.aforward)
        session = CodingSession(
            agent,
            thread_id=thread_id,
            checkpoint_store=store,
            scope_factory=lambda scope: replace(
                scope,
                workspace=workspace,
                permissions=workspace.permissions,
                principal="local_user",
            ),
        )
        controller = (
            CodingApprovalController(agent, policy, principal=lambda: "local_user")
            if policy
            else None
        )

        async def close():
            closed.append(thread_id)
            await workspace.aclose()
            registry.close()
            journal.close()
            store.close()

        return session, controller, close

    return CodingHost(storage, str(project), factory), calls, models, closed


async def command(app, text):
    app.query_one("#composer", TextArea).load_text(text)
    await app._send_prompt()
    worker = app._stream_worker
    if worker is not None:
        await worker


def transcript(app):
    return "\n".join(
        str(row.content) for row in app.query_one("#transcript").query(Static)
    )


@pytest.mark.asyncio
async def test_sqlite_reopen_restores_tools_without_reexecution_and_new_prompt_uses_history(
    tmp_path,
):
    host, calls, models, _closed = make_host(tmp_path)
    first, _ = await host.select()
    thread_id = first.thread_id
    models[-1].side_effect = [tool_response(), response("first answer")]
    [event async for event in first.stream("remember cobalt")]
    await host.aclose()
    assert calls == ["cobalt"]

    restarted, calls_after_restart, new_models, _ = make_host(tmp_path)
    try:
        session, controller = await restarted.select(thread_id)
        app = CodingApp(session, host=restarted, approval_controller=controller)
        async with app.run_test() as pilot:
            await pilot.pause()
            restored = transcript(app)
            assert "remember cobalt" in restored
            assert "first answer" in restored
            assert "lookup" in restored
            assert "verified: cobalt" in restored
            assert not calls_after_restart
            assert new_models[-1].call_count == 0
            await command(app, "/help")
            await command(app, "/session")
            await command(app, "/runs")
            assert new_models[-1].call_count == 0
            assert "/resume" in transcript(app)
            assert "completed" in transcript(app)
            await command(app, "what word?")
            assert new_models[-1].call_count == 1
            history = new_models[-1].call_args.kwargs["messages"].to_chatml()
            assert any(
                "remember cobalt" in str(item.get("content")) for item in history
            )
            assert not any(
                str(item.get("content", "")).startswith("/help") for item in history
            )
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_sqlite_paused_approval_is_restored_and_review_resumes_once(tmp_path):
    host, calls, models, _ = make_host(tmp_path, protected=True)
    session, _ = await host.select()
    thread_id = session.thread_id
    models[-1].side_effect = [tool_response()]
    with pytest.raises(TaskPauseRequestedError):
        [event async for event in session.stream("look up cobalt")]
    await host.aclose()
    assert calls == []

    restarted, restarted_calls, models, _ = make_host(tmp_path, protected=True)
    try:
        session, controller = await restarted.select(thread_id)
        app = CodingApp(session, host=restarted, approval_controller=controller)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._waiting_for_approval
            assert "Approval required" in transcript(app)
            assert models[-1].call_count == 0
            await command(app, "/help")
            assert app._waiting_for_approval
            app._handle_approval_button("approval-approve-0")
            await app._stream_worker
            assert restarted_calls == ["cobalt"]
            assert models[-1].call_count == 1
            assert session.latest_run()["status"] == "completed"
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_resume_picker_new_unknown_and_failed_switch_preserve_session(tmp_path):
    host, _calls, models, closed = make_host(tmp_path)
    try:
        first, _ = await host.select()
        first_id = first.thread_id
        [event async for event in first.stream("first message")]
        second, controller = await host.select()
        second_id = second.thread_id
        app = CodingApp(second, host=host, approval_controller=controller)
        async with app.run_test() as pilot:
            await command(app, "/does-not-exist")
            assert "Unknown coding command" in transcript(app)
            assert models[-1].call_count == 0
            await command(app, "/resume missing-thread")
            assert app.coding_session.thread_id == second_id
            assert not host.storage.thread_dir("missing-thread").exists()
            await command(app, "/resume")
            await pilot.pause()
            assert isinstance(app.screen, CodingPicker)
            await pilot.press("escape")
            await pilot.pause()
            assert app.coding_session.thread_id == second_id
            await command(app, "/resume")
            await pilot.pause()
            app.screen.query_one("#picker-filter").value = first_id
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.coding_session.thread_id == first_id
            assert "first message" in transcript(app)
            assert second_id in closed
            await command(app, "/new")
            assert app.coding_session.thread_id not in {first_id, second_id}
            assert "first message" not in transcript(app)
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_slash_completion_palette_and_reserved_extension_names(tmp_path):
    host, _calls, _models, _closed = make_host(tmp_path)
    try:
        session, _ = await host.select()
        app = CodingApp(session, host=host)
        async with app.run_test() as pilot:
            composer = app.query_one("#composer", TextArea)
            composer.load_text("/res")
            await pilot.pause()
            assert "/resume" in str(app.query_one("#command-hints", Static).content)
            await pilot.press("tab")
            await pilot.pause()
            assert composer.text == "/resume "
            await app.action_command_picker()
            await pilot.pause()
            assert isinstance(app.screen, CodingPicker)
            app.screen.query_one("#picker-filter").value = "new conversation"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert composer.text == "/new "
        extensions = CodingExtensions()
        extensions.register_command("resume", lambda _: None)
        with pytest.raises(ValueError, match="Reserved"):
            CodingApp(session, extensions=extensions)
    finally:
        await host.aclose()


def test_catalog_skips_invalid_metadata_symlinks_and_other_projects(tmp_path):
    storage = ThreadStorage(tmp_path)
    storage.create_thread("valid", workspace="/project")
    storage.open_checkpoint_store("valid").close()
    storage.create_thread("foreign", workspace="/other")
    storage.open_checkpoint_store("foreign").close()
    storage.create_thread("broken")
    storage.open_checkpoint_store("broken").close()
    (storage.thread_dir("broken") / "metadata.json").write_text('{"thread_id":"wrong"}')
    (storage.threads_dir / "linked").symlink_to(
        storage.thread_dir("valid"), target_is_directory=True
    )
    assert [item.thread_id for item in storage.list_threads(workspace="/project")] == [
        "valid"
    ]


@pytest.mark.asyncio
async def test_unfinished_run_blocks_new_prompt_and_unknown_continue_without_model(
    tmp_path,
):
    host, _calls, models, _closed = make_host(tmp_path)
    try:
        session, _ = await host.select()
        [event async for event in session.stream("hello")]
        latest = session.latest_run()
        state = session.saved_state(latest["run_id"])
        state["status"] = "running"
        session.checkpoint_store.save_state(
            session.namespace, session.thread_id, latest["run_id"], state
        )
        model_calls = models[-1].call_count
        with pytest.raises(ValueError, match="unfinished"):
            [event async for event in session.stream("do not start again")]
        with pytest.raises(ValueError, match="Unknown run"):
            [event async for event in session.resume("missing")]
        assert models[-1].call_count == model_calls
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_terminal_checkpoint_with_uncertain_command_cannot_start_new_turn(
    tmp_path,
):
    from msgflux.runtime.workspace.receipts import new_command_receipt

    host, _calls, models, _closed = make_host(tmp_path)
    try:
        session, _ = await host.select()
        [event async for event in session.stream("hello")]
        latest = session.latest_run()
        state = session.saved_state(latest["run_id"])
        state["runtime"]["extensions"]["command_receipts"] = [
            new_command_receipt(
                workspace_reference=None,
                backend="local",
                run_id=latest["run_id"],
                tool_call_id="uncertain-call",
            ).to_dict()
        ]
        session.checkpoint_store.save_state(
            session.namespace, session.thread_id, latest["run_id"], state
        )
        model_calls = models[-1].call_count
        with pytest.raises(TaskPauseRequestedError, match="reconciliation"):
            [event async for event in session.stream("do not replay")]
        assert models[-1].call_count == model_calls
    finally:
        await host.aclose()


def _crash_during_generation(directory):
    import asyncio
    import os
    from pathlib import Path

    async def run():
        host, _calls, models, _closed = make_host(Path(directory))
        session, _ = await host.select()
        (Path(directory) / "thread-id").write_text(session.thread_id)

        async def crash(**_kwargs):
            assert session.latest_run()["status"] == "running"
            os._exit(23)

        models[-1].side_effect = crash
        [event async for event in session.stream("survive this controller crash")]

    asyncio.run(run())


@pytest.mark.asyncio
async def test_new_process_continues_committed_generation_checkpoint_without_new_user_turn(
    tmp_path,
):
    import multiprocessing

    process = multiprocessing.get_context("spawn").Process(
        target=_crash_during_generation, args=(str(tmp_path),)
    )
    process.start()
    process.join(20)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("Crash-test worker did not exit")
    assert process.exitcode == 23
    thread_id = (tmp_path / "thread-id").read_text()
    host, _calls, models, _closed = make_host(tmp_path)
    try:
        session, controller = await host.select(thread_id)
        run_id = session.latest_run()["run_id"]
        app = CodingApp(session, host=host, approval_controller=controller)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "survive this controller crash" in transcript(app)
            assert models[-1].call_count == 0
            await command(app, "/continue")
            assert models[-1].call_count == 1
            assert session.latest_run()["run_id"] == run_id
            assert session.latest_run()["status"] == "completed"
            messages = session.saved_state(run_id)["messages"]["items"]
            assert sum(item.get("role") == "user" for item in messages) == 1
    finally:
        await host.aclose()


@pytest.mark.asyncio
async def test_restore_failure_and_active_tasks_do_not_close_current_session(tmp_path):
    host, _calls, _models, closed = make_host(tmp_path)
    try:
        session, _ = await host.select()
        original_factory = host._factory

        async def fail(_thread_id):
            raise RuntimeError("provider unavailable")

        host._factory = fail
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await host.select()
        assert host.session is session
        assert closed == []
        host._factory = original_factory
        task_store = session.agent.tool_library.get_task_store()
        task_store.create(task_id="active", tool_name="bash", metadata={})
        with pytest.raises(RuntimeError, match="background tasks"):
            await host.select()
        assert host.session is session
        assert closed == []
    finally:
        await host.aclose()
