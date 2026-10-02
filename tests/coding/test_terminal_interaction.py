"""End-to-end interaction coverage for the terminal composer."""

import pytest
from unittest.mock import AsyncMock

textual = pytest.importorskip("textual")
from textual.widgets import OptionList, TextArea

from msgflux.coding.tui import CodingApp
from tests.coding.test_tui import Event, FakeSession


@pytest.mark.asyncio
async def test_enter_sends_and_alt_enter_inserts_newline():
    session = FakeSession()
    app = CodingApp(session)
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("first line")
        composer.move_cursor((0, len(composer.text)))
        await pilot.press("alt+enter")
        assert composer.text == "first line\n"
        composer.load_text("send this")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert session.prompts == ["send this"]


@pytest.mark.asyncio
async def test_slash_completion_lists_every_builtin_and_navigation_works():
    app = CodingApp(FakeSession())
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.load_text("/")
        await pilot.pause()
        options = app.query_one("#command-hints", OptionList)
        expected = {
            "help",
            "resume",
            "new",
            "session",
            "runs",
            "continue",
            "sidebar",
            "copy",
            "quit",
        }
        assert {
            options.get_option_at_index(i).id for i in range(options.option_count)
        } == expected

        options.highlighted = 0
        await pilot.press("down")
        assert options.get_option_at_index(options.highlighted).id == "resume"
        await pilot.press("up")
        assert options.get_option_at_index(options.highlighted).id == "help"
        composer.load_text("/q")
        # TextArea emits Changed through Textual's message queue. Wait for the
        # filtered menu to settle before asserting Tab completion.
        for _ in range(10):
            await pilot.pause(0.01)
            if (
                options.option_count == 1
                and options.get_option_at_index(0).id == "quit"
            ):
                break
        assert options.option_count == 1
        assert options.get_option_at_index(0).id == "quit"
        await pilot.press("tab")
        assert composer.text == "/quit "


@pytest.mark.asyncio
async def test_tool_start_and_end_update_same_card_and_preserve_falsey_result():
    app = CodingApp(FakeSession())
    async with app.run_test():
        await app._render_event(
            Event(
                "tool.start",
                {
                    "tool_call_id": "call-1",
                    "tool_name": "search",
                    "arguments": {"query": "x"},
                },
            ),
            assistant_started=False,
        )
        await app._render_event(
            Event(
                "tool.end",
                {"tool_call_id": "call-1", "result": "", "status": "completed"},
            ),
            assistant_started=False,
        )
        cards = list(app.query(".tool-card"))
        assert len(cards) == 1
        assert "Tool: search · completed" in cards[0].content
        assert cards[0].content.endswith("\n")


@pytest.mark.asyncio
async def test_copy_command_exports_transcript_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "msgflux.coding.tui.app.copy_system_clipboard", AsyncMock(return_value=False)
    )
    destination = tmp_path / "transcript.txt"
    app = CodingApp(FakeSession())
    async with app.run_test() as pilot:
        await app._append_transcript(
            "Assistant: saved text", classes="assistant-message"
        )
        composer = app.query_one("#composer", TextArea)
        composer.load_text(f"/copy --file {destination}")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert destination.read_text(encoding="utf-8") == "Assistant: saved text"
        assert str(app.query_one("#status").content) == "Ready"


@pytest.mark.asyncio
async def test_right_click_uses_osc52_fallback_when_native_clipboard_fails(monkeypatch):
    monkeypatch.setattr(
        "msgflux.coding.tui.app.copy_system_clipboard", AsyncMock(return_value=False)
    )
    app = CodingApp(FakeSession())
    async with app.run_test() as pilot:
        await app._append_transcript(
            "Assistant: selectable content", classes="assistant-message"
        )
        await pilot.click("#transcript", button=3)
        assert app.clipboard == "Assistant: selectable content"
        message = await app.copy_transcript("")
        assert "OSC52" in message
        assert "--file PATH" in message
