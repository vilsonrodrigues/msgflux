from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from msgflux.exceptions import TaskPauseRequestedError

textual = pytest.importorskip("textual")
from textual.widgets import Static, TextArea

from msgflux.coding.extensions import CodingExtensions
from msgflux.coding.tui import CodingApp


@dataclass
class Event:
    type: str
    data: dict
    run_id: str | None = None


class FakeSession:
    thread_id = "thread-1"

    def __init__(self):
        self.prompts = []
        self.cancelled = False

    async def stream(self, prompt):
        self.prompts.append(prompt)
        yield Event("message.delta", {"delta": "Hello"})
        yield Event("message.delta", {"delta": " there [bold red]unsafe[/]"})
        yield Event("message.end", {"content": "Hello there [bold red]unsafe[/]"})
        yield Event(
            "tool.start",
            {
                "tool_name": "read_file",
                "arguments": {"path": "[link=https://bad]payload[/link]"},
            },
        )

    async def cancel(self):
        self.cancelled = True


@pytest.mark.asyncio
async def test_coding_app_streams_prompt_and_keeps_sidebars_available():
    session = FakeSession()
    app = CodingApp(session, workspace_root="/project")

    async with app.run_test(size=(100, 35)) as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("inspect [bold red]project[/]")
        assert composer.text == "inspect [bold red]project[/]"
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        assert session.prompts == ["inspect [bold red]project[/]"]
        rows = app.query_one("#transcript").query(Static)
        user_rows = [row for row in rows if "user-message" in row.classes]
        assistant_rows = [row for row in rows if "assistant-message" in row.classes]
        tool_rows = [row for row in rows if "tool-card" in row.classes]
        assert user_rows[0].content == "You: inspect [bold red]project[/]"
        assert len(assistant_rows) == 1
        assert assistant_rows[0].content == (
            "Assistant: Hello there [bold red]unsafe[/]"
        )
        assert len(tool_rows) == 1
        assert "Tool: read_file · tool.start" in tool_rows[0].content
        assert "[link=https://bad]payload[/link]" in tool_rows[0].content
        assert app.query_one("#workspace-path").content == "/project"
        assert app.query_one("#left-sidebar").display
        assert not app.query_one("#right-sidebar").display
        assert str(app.query_one("#status").content) == "Ready"


@pytest.mark.asyncio
async def test_coding_app_restores_user_and_assistant_history():
    class History:
        def to_chatml(self):
            return [
                {"role": "user", "content": "Earlier question"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Earlier answer"}],
                },
                {"role": "tool", "content": "internal tool output"},
            ]

    class SnapshotSession(FakeSession):
        async def snapshot(self):
            return type("Snapshot", (), {"messages": History()})()

    app = CodingApp(SnapshotSession())

    async with app.run_test() as _pilot:
        rows = app.query_one("#transcript").query(Static)
        assert [row.content for row in rows] == [
            "User: Earlier question",
            "Assistant: Earlier answer",
            "Tool result (): internal tool output",
        ]


@pytest.mark.asyncio
async def test_coding_app_cancel_requests_session_cancellation():
    class SlowSession(FakeSession):
        async def stream(self, prompt):
            self.prompts.append(prompt)
            await asyncio.Event().wait()
            yield Event("message.delta", {"delta": "never"})

    session = SlowSession()
    app = CodingApp(session)

    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("wait")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.05)
        assert app._run_active
        await pilot.press("escape")
        await pilot.pause(0.05)
        assert session.cancelled
        assert str(app.query_one("#status").content) == "Cancelled"


@pytest.mark.asyncio
async def test_registered_right_panel_mounts_in_reserved_sidebar():
    extensions = CodingExtensions()
    extensions.register_panel(
        "files", "Files", lambda: Static("file list"), side="right"
    )
    app = CodingApp(FakeSession(), extensions=extensions)

    async with app.run_test(size=(110, 35)) as pilot:
        assert app.query_one("#right-extension-panels Static").content == "Files"
        await pilot.press("f3")
        await pilot.pause()
        assert app.query_one("#right-sidebar").display


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_id", "approved"),
    [("approval-approve-0", True), ("approval-deny-0", False)],
)
async def test_approval_review_decision_resumes_paused_run(button_id, approved):
    class ApprovalSession(FakeSession):
        async def stream(self, prompt):
            self.prompts.append(prompt)
            yield Event("tool.approval_required", {}, run_id="run-approval")
            yield Event("run.paused", {}, run_id="run-approval")
            raise TaskPauseRequestedError("approval-task")

        async def resume(self, run_id):
            self.resumed_run = run_id
            yield Event("message.delta", {"delta": "Approved and resumed"})
            yield Event("message.end", {"content": "Approved and resumed"})

    class Controller:
        def __init__(self):
            self.decisions = []

        def pending(self, thread_id, run_id):
            assert (thread_id, run_id) == ("thread-1", "run-approval")
            return (
                type(
                    "Review",
                    (),
                    {
                        "request_id": "request-1",
                        "tool_name": "write_file",
                        "expires_at": 1_800_000_000.0,
                        "diff": "--- a/file.py\n+++ b/file.py\n+safe change",
                    },
                )(),
            )

        def decide(self, thread_id, run_id, request_id, *, approved):
            self.decisions.append((thread_id, run_id, request_id, approved))

    session = ApprovalSession()
    controller = Controller()
    app = CodingApp(session, approval_controller=controller)

    async with app.run_test(size=(110, 40)) as pilot:
        app.query_one("#composer", TextArea).load_text("change file")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        card = app.query_one(".approval-card")
        assert len(app.query(".approval-card")) == 1
        card_text = "\n".join(str(row.content) for row in card.query(Static))
        assert "Approval required · write_file" in card_text
        assert "Prepared diff" in card_text
        assert "safe change" in card_text
        assert "Expires:" in card_text
        assert str(app.query_one("#status").content).startswith("Approval pending")
        assert app.query_one("#approval-approve-0")
        assert app.query_one("#approval-deny-0")
        assert str(app.query_one("#status").content).startswith("Approval pending")

        await pilot.click("#approval-approve-0")
        await pilot.pause(0.2)
        assert controller.decisions == [
            ("thread-1", "run-approval", "request-1", True)
        ], (str(app.query_one("#status").content), app._approval_actions)
        assert session.resumed_run == "run-approval"
        resumed = [row.content for row in app.query(".assistant-message")]
        assert "Assistant: Approved and resumed" in resumed
        assert str(app.query_one("#status").content) == "Ready"


@pytest.mark.asyncio
async def test_pause_without_controller_keeps_paused_status_and_single_notice():
    class PausedSession(FakeSession):
        async def stream(self, prompt):
            yield Event("tool.approval_required", {}, run_id="run-pending")
            yield Event("run.paused", {}, run_id="run-pending")
            raise TaskPauseRequestedError("approval-task")

    app = CodingApp(PausedSession())

    async with app.run_test() as pilot:
        app.query_one("#composer", TextArea).load_text("needs approval")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)
        notices = [
            row for row in app.query(".approval-card") if isinstance(row, Static)
        ]
        assert len(notices) == 1
        assert "no approval controller configured" in str(
            app.query_one("#status", Static).content
        )
        assert app._waiting_for_approval


@pytest.mark.asyncio
async def test_registered_slash_commands_receive_arguments_without_agent_call():
    extensions = CodingExtensions()
    seen = []

    def sync_command(argument_text):
        seen.append(("sync", argument_text))
        return f"sync result: {argument_text}"

    async def async_command(argument_text):
        await asyncio.sleep(0)
        seen.append(("async", argument_text))
        return f"async result: {argument_text}"

    extensions.register_command("sync", sync_command)
    extensions.register_command("async", async_command)
    session = FakeSession()
    app = CodingApp(session, extensions=extensions)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("/sync hello world")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        composer.load_text("/async another argument")
        await pilot.press("ctrl+enter")
        await pilot.pause(0.1)

        assert seen == [("sync", "hello world"), ("async", "another argument")]
        assert session.prompts == []
        results = [row.content for row in app.query(".command-result")]
        assert results == ["sync result: hello world", "async result: another argument"]


@pytest.mark.asyncio
async def test_workspace_file_list_uses_session_authorized_paths():
    class WorkspaceSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.requested_limit = None

        async def workspace_files(self, *, limit):
            self.requested_limit = limit
            return ("/README.md", "/src/[bold]literal.py")

    session = WorkspaceSession()
    app = CodingApp(session)

    async with app.run_test() as _pilot:
        listing = app.query_one("#workspace-files Static")
        assert session.requested_limit == 500
        assert listing.content == "/README.md\n/src/[bold]literal.py"


@pytest.mark.asyncio
async def test_commentary_is_rendered_separately_from_final_answer():
    app = CodingApp(FakeSession(), workspace_root="/project")
    async with app.run_test(size=(100, 35)):
        await app._render_event(
            Event("commentary.delta", {"delta": "Checking "}), assistant_started=False
        )
        await app._render_event(
            Event("commentary.delta", {"delta": "files"}), assistant_started=False
        )
        await app._render_event(
            Event("message.delta", {"delta": "Done"}), assistant_started=False
        )
        rows = app.query_one("#transcript").query(Static)
        commentary = [row for row in rows if "commentary-message" in row.classes]
        answers = [row for row in rows if "assistant-message" in row.classes]
        assert len(commentary) == 1
        assert commentary[0].content == "Progress: Checking files"
        assert answers[0].content == "Assistant: Done"


@pytest.mark.asyncio
async def test_f4_copies_displayed_error_and_conversation():
    app = CodingApp(FakeSession(), workspace_root="/project")
    async with app.run_test(size=(100, 35)) as pilot:
        await app._append_transcript("Assistant: hello", classes="assistant-message")
        await app._append_transcript(
            "Codex request failed with HTTP 400", classes="error"
        )
        await pilot.press("f4")
        assert "Assistant: hello" in app.clipboard
        assert "HTTP 400" in app.clipboard


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_tui_displays_real_bash_commands_in_both_dispatch_modes(
    tmp_path, background
):
    from msgflux.nn import ToolLibrary
    from msgflux.runtime import ExecutionScope, execution_context
    from msgflux.tools.builtin import BashTool
    from msgflux.coding.workspace import open_coding_workspace

    workspace, registry = await open_coding_workspace(
        tmp_path, tmp_path / "workspace.sqlite3"
    )
    bash = BashTool()
    bash.tool_config["allow_background"] = True
    library = ToolLibrary("bash-display", [bash])
    command = "printf 'bash-visible [bold]literal[/bold]'"
    scope = ExecutionScope(
        thread_id=f"bash-display-{background}",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    app = CodingApp(FakeSession())
    try:
        with execution_context(scope=scope):
            events = [
                event
                async for event in library.stream_events(
                    tool_callings=[
                        (
                            "bash-call",
                            "bash",
                            {"command": command, "run_in_background": background},
                        )
                    ]
                )
            ]
            if background:
                dispatched = next(
                    event for event in events if event.type == "task.start"
                )
                task_id = dispatched.data["task_id"]
                assert dispatched.data["tool_call_id"] == "bash-call"
                assert dispatched.data["arguments"]["command"] == command
                # Hidden dependencies must never appear in the transcript.
                assert "environment" not in dispatched.data["arguments"]
                await library.arun("task_wait", {"task_id": task_id, "timeout": 5.0})
                assert library.get_task_store().get(task_id).status == "completed"
            async with app.run_test() as _pilot:
                for event in events:
                    await app._render_event(event, assistant_started=False)
                cards = list(app.query_one("#transcript").query(".tool-card"))
                start_card = next(
                    card
                    for card in cards
                    if ("task.start" if background else "tool.start") in card.content
                )
                assert command in start_card.content
                if background:
                    assert task_id in start_card.content
                    assert "Tool: bash(run_in_background=true)" in start_card.content
    finally:
        await workspace.aclose()
        registry.close()
