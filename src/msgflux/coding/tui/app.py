"""Textual user interface for a coding session."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from msgflux.coding.session import TERMINAL_RUN_STATUSES
from msgflux.coding.tui.clipboard import (
    copy_system_clipboard,
    export_text,
    parse_copy_arguments,
)
from msgflux.coding.tui.commands import builtin_commands
from msgflux.coding.tui.pickers import CodingPicker
from msgflux.coding.tui.tool_cards import ToolCard
from msgflux.coding.tui.tool_cards import plain_content as _plain_content
from msgflux.exceptions import TaskPauseRequestedError


class CodingApp(App[None]):
    """A small, extensible shell around a ``CodingSession``-like object."""

    TITLE = "Vulcano"
    SUB_TITLE = "msgFlux coding"
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #left-sidebar {
      width: 26; min-width: 18; height: 1fr;
      border: round $primary; padding: 0 1;
    }
    #center { width: 1fr; height: 1fr; }
    #right-sidebar {
      width: 28; min-width: 18; border: round $primary;
      padding: 0 1; display: none;
    }
    #transcript { height: 1fr; border: round $surface; padding: 0 1; }
    .user-message, .assistant-message { padding: 0 1; margin: 1 0; }
    #composer-row { height: 3; }
    #composer { width: 1fr; height: 3; border: round $accent; }
    #command-hints { height: 8; border: none; display: none; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    .tool-card { border: round $warning; margin: 1 0; padding: 0 1; }
    .approval-card { border: round $warning; margin: 1 0; padding: 1; }
    .error { color: $error; }
    """
    BINDINGS = [
        Binding("enter", "send_prompt", "Send", priority=True),
        Binding("ctrl+enter", "send_prompt", show=False, priority=True),
        Binding("shift+enter,alt+enter", "newline", "Newline", priority=True),
        Binding("up", "previous_command", show=False, priority=True),
        Binding("down", "next_command", show=False, priority=True),
        Binding("escape", "cancel_run", "Cancel", priority=True),
        ("f2", "toggle_left", "Sidebar"),
        ("f3", "toggle_right", "Panels"),
        Binding("f4,ctrl+shift+c", "copy_transcript", "Copy", priority=True),
        ("ctrl+k", "command_picker", "Commands"),
        Binding("tab", "complete_command", "Complete", priority=True),
    ]

    def __init__(
        self,
        session: Any,
        *,
        workspace: str = ".",
        workspace_root: str | None = None,
        extensions: Any = None,
        approval_controller: Any = None,
        host: Any = None,
        open_session_picker: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.coding_session = session
        self.host = host
        self._open_session_picker = open_session_picker
        self._builtin_commands = builtin_commands(self)
        if extensions is not None:
            collisions = {item.id for item in self._builtin_commands} & {
                item.id for item in extensions.commands()
            }
            if collisions:
                raise ValueError(
                    f"Reserved coding commands: {', '.join(sorted(collisions))}"
                )
        self.workspace = workspace_root or workspace
        self.extensions = extensions
        self.approval_controller = approval_controller
        self._approval_actions: dict[int, tuple[str, str, str]] = {}
        self._approval_keys: set[tuple[str, str]] = set()
        self._unreviewable_paused_runs: set[str] = set()
        self._next_approval_action = 0
        self._stream_worker: asyncio.Task[None] | None = None
        self._run_active = False
        self._waiting_for_approval = False
        self._assistant_widget: Static | None = None
        self._assistant_text = ""
        self._commentary_widget: Static | None = None
        self._commentary_text = ""
        self._left_visible = True
        self._right_visible = False
        self._tool_cards = {}
        self._task_cards = {}
        self._selected_transcript_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="left-sidebar"):
                yield Static("Workspace", classes="sidebar-title")
                yield Static(self.workspace, id="workspace-path")
                yield Static("\nSessions", classes="sidebar-title")
                thread_id = getattr(self.coding_session, "thread_id", None)
                yield Static(str(thread_id or "Current session"), id="session-list")
                yield Vertical(id="left-extension-panels")
            with Vertical(id="center"):
                yield VerticalScroll(id="transcript")
                yield OptionList(id="command-hints")
                with Horizontal(id="composer-row"):
                    yield TextArea(
                        id="composer", soft_wrap=True, show_line_numbers=False
                    )
            with Vertical(id="right-sidebar"):
                yield Static("Panels", classes="sidebar-title")
                yield Static("Reserved for files, tasks, and extensions.")
                yield Vertical(id="right-extension-panels")
        yield Static("Ready", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#composer", TextArea).focus()
        self._apply_responsive_layout(self.size.width)
        await self._load_history()
        await self._mount_extension_panels()
        await self._refresh_sessions()
        if self._open_session_picker:
            await self._show_session_picker()

    async def _load_history(self) -> None:  # noqa: C901
        snapshot_method = getattr(self.coding_session, "snapshot", None)
        if not callable(snapshot_method):
            return
        try:
            snapshot = await snapshot_method()
        except Exception as exc:
            await self._append_transcript(
                f"Unable to restore history: {exc}", classes="error"
            )
            self.query_one("#status", Static).update("History restoration failed")
            return
        messages = getattr(snapshot, "messages", None)
        if messages is None:
            return
        to_chatml = getattr(messages, "to_chatml", None)
        items = to_chatml() if callable(to_chatml) else messages
        pending_tools = {}
        for item in items:
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            if role == "tool":
                call_id = item.get("tool_call_id")
                card = pending_tools.pop(call_id, None)
                if card is None:
                    card = ToolCard(str(item.get("name") or call_id or "unknown"))
                    await self.query_one("#transcript", VerticalScroll).mount(card)
                content = item.get("content", "")
                card.apply("tool.end", {"result": _history_value(content)})
                continue
            if role == "assistant" and item.get("tool_calls"):
                for call in item["tool_calls"]:
                    function = call.get("function", call)
                    card = ToolCard(str(function.get("name", "unknown")))
                    card.apply(
                        "tool.start",
                        {"arguments": _history_value(function.get("arguments", {}))},
                    )
                    pending_tools[call.get("id")] = card
                    await self.query_one("#transcript", VerticalScroll).mount(card)
            if role not in {"user", "assistant"}:
                continue
            content = item.get("content")
            if content is None:
                continue
            await self._append_transcript(
                f"{role.title()}: {_plain_content(content)}",
                classes=f"{role}-message",
            )

        latest_method = getattr(self.coding_session, "latest_run", None)
        latest = latest_method() if callable(latest_method) else None
        if latest and latest["status"] == "paused":
            await self._handle_pause_event(None, {"run_id": latest["run_id"]})
        elif latest and latest["status"] not in TERMINAL_RUN_STATUSES:
            self.query_one("#status", Static).update(
                "Unfinished run · /continue resumes after runtime checks"
            )

    async def _mount_extension_panels(self) -> None:
        if self.extensions is None:
            return
        for spec in self.extensions.panels():
            target = f"#{spec.side}-extension-panels"
            container = self.query_one(target, Vertical)
            await container.mount(Static(spec.title, classes="sidebar-title"))
            widget = spec.factory()
            if not isinstance(widget, Widget):
                raise TypeError(
                    f"Panel factory {spec.id!r} must return a Textual widget"
                )
            await container.mount(widget)

    async def on_unmount(self) -> None:
        worker = self._stream_worker
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    def on_resize(self, event: App.Resize) -> None:
        self._apply_responsive_layout(event.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        self.query_one("#right-sidebar").display = self._right_visible and width > 90
        self.query_one("#left-sidebar").display = self._left_visible and width > 68

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("approval-"):
            self._handle_approval_button(event.button.id)

    async def action_send_prompt(self) -> None:
        composer = self.query_one("#composer", TextArea)
        hints = self.query_one("#command-hints", OptionList)
        if hints.display and hints.highlighted is not None:
            selected = hints.get_option_at_index(hints.highlighted).id
            composer.load_text(f"/{selected}")
        await self._send_prompt()

    async def _send_prompt(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text.strip()
        if not prompt or self._run_active:
            return
        if self._waiting_for_approval and not prompt.startswith("/"):
            self.query_one("#status", Static).update(
                "Approval pending · review it before sending a prompt"
            )
            return
        composer.clear()
        if not prompt.startswith("/"):
            self._waiting_for_approval = False
            await self._append_transcript(f"You: {prompt}", classes="user-message")
        self._run_active = True
        self.query_one("#status", Static).update("Working…  (Esc to cancel)")
        if prompt.startswith("/"):
            self._stream_worker = asyncio.create_task(self._dispatch_command(prompt))
            return
        self._stream_worker = asyncio.create_task(self._consume_prompt(prompt))

    async def _dispatch_command(self, prompt: str) -> None:
        parts = prompt[1:].split(maxsplit=1)
        command_id = parts[0] if parts else ""
        argument_text = parts[1] if len(parts) > 1 else ""
        command = next(
            (item for item in self.command_specs() if item.id == command_id), None
        )
        if command is None:
            self.query_one("#status", Static).update("Unknown command")
            await self._append_transcript(
                f"Unknown coding command: /{command_id}", classes="error"
            )
            self._run_active = False
            self._stream_worker = None
            self.query_one("#composer", TextArea).focus()
            return
        self.query_one("#status", Static).update(f"Running /{command_id}…")
        if argument_text and not command.accepts_arguments:

            async def invalid_arguments(_args):
                raise ValueError(f"/{command.id} does not take arguments")

            await self._run_command(invalid_arguments, argument_text)
            return
        await self._run_command(
            command.handler, argument_text, preserve_status=command.preserve_status
        )

    def command_specs(self):
        extensions = self.extensions.commands() if self.extensions is not None else ()
        return (*self._builtin_commands, *extensions)

    def _commands(self):
        return {item.id: item.description for item in self.command_specs()}

    def on_text_area_changed(self, event: TextArea.Changed):
        if event.text_area.id != "composer":
            return
        text = event.text_area.text
        prefix = (
            text[1:]
            if text.startswith("/") and not any(c.isspace() for c in text)
            else None
        )
        options = self.query_one("#command-hints", OptionList)
        options.clear_options()
        options.add_options(
            Option(Text(f"/{spec.id}  {spec.description}"), id=spec.id)
            for spec in self.command_specs()
            if prefix is not None and spec.id.startswith(prefix)
        )
        options.display = bool(options.option_count)
        if options.option_count:
            options.highlighted = 0

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        composer_focused = (
            isinstance(self.focused, TextArea) and self.focused.id == "composer"
        )
        if action in {"send_prompt", "newline"}:
            return composer_focused
        if action in {"complete_command", "previous_command", "next_command"}:
            return (
                composer_focused
                and self.query_one("#command-hints", OptionList).display
            )
        return super().check_action(action, parameters)

    def action_newline(self):
        self.query_one("#composer", TextArea).insert("\n")

    def action_previous_command(self):
        self.query_one("#command-hints", OptionList).action_cursor_up()

    def action_next_command(self):
        self.query_one("#command-hints", OptionList).action_cursor_down()

    def action_complete_command(self):
        composer = self.query_one("#composer", TextArea)
        options = self.query_one("#command-hints", OptionList)
        if options.highlighted is not None:
            name = options.get_option_at_index(options.highlighted).id
            composer.load_text(f"/{name} ")
            composer.move_cursor((0, len(composer.text)))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        if event.option_list.id == "command-hints":
            self._command_selected(event.option.id)

    async def action_command_picker(self):
        await self.push_screen(
            CodingPicker(
                "Commands",
                tuple(
                    (name, f"/{name}  {description}")
                    for name, description in self._commands().items()
                ),
            ),
            self._command_selected,
        )

    def _command_selected(self, command):
        if command is not None:
            composer = self.query_one("#composer", TextArea)
            composer.load_text(f"/{command} ")
            composer.move_cursor((0, len(composer.text)))
            composer.focus()

    async def _refresh_sessions(self):
        thread_id = str(getattr(self.coding_session, "thread_id", "Current session"))
        count = len(self.host.threads()) if self.host is not None else 1
        self.query_one("#session-list", Static).update(
            f"{thread_id}\n{count} saved · /resume"
        )

    async def _show_session_picker(self):
        if self.host is None:
            raise ValueError("This app has no session host")
        threads = self.host.threads()
        if not threads:
            return "No saved sessions in this workspace"
        await self.push_screen(
            CodingPicker(
                "Resume session",
                tuple(
                    (
                        item.thread_id,
                        f"{item.title or item.thread_id}  ·  {item.updated_at}",
                    )
                    for item in threads
                ),
            ),
            self._session_selected,
        )

    async def _session_selected(self, thread_id):
        if thread_id is None or self._run_active:
            return
        self._run_active = True
        try:
            await self._switch_session(thread_id)
        except Exception as exc:
            await self._append_transcript(f"Unable to resume: {exc}", classes="error")
        finally:
            self._run_active = False

    async def _switch_session(self, thread_id):
        if self.host is None:
            raise ValueError("This app has no session host")
        session, controller = await self.host.select(thread_id)
        self.coding_session, self.approval_controller = session, controller
        await self.query_one("#transcript", VerticalScroll).remove_children()
        self._assistant_widget = self._commentary_widget = None
        self._assistant_text = self._commentary_text = ""
        self._waiting_for_approval = False
        self._approval_actions.clear()
        self._approval_keys.clear()
        self._tool_cards.clear()
        self._task_cards.clear()
        self._selected_transcript_text = ""
        self._unreviewable_paused_runs.clear()
        self.query_one("#status", Static).update("Ready")
        await self._load_history()
        await self._refresh_sessions()
        if self.host.cleanup_error is not None:
            await self._append_transcript(
                f"Previous session cleanup failed: {self.host.cleanup_error}",
                classes="error",
            )

    async def _run_command(
        self, handler: Any, argument_text: str, *, preserve_status=False
    ) -> None:
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(argument_text)
            else:
                result = await asyncio.to_thread(handler, argument_text)
                if inspect.isawaitable(result):
                    result = await result
            if result is not None:
                await self._append_transcript(
                    _plain_content(result), classes="command-result"
                )
            if not preserve_status and not self._waiting_for_approval:
                self.query_one("#status", Static).update("Ready")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._append_transcript(f"Command failed: {exc}", classes="error")
            self.query_one("#status", Static).update("Command failed")
        finally:
            self._run_active = False
            self._stream_worker = None
            self.query_one("#composer", TextArea).focus()

    def _handle_approval_button(self, button_id: str) -> None:
        prefix, separator, action_index = button_id.rpartition("-")
        if not separator or prefix not in {"approval-approve", "approval-deny"}:
            return
        try:
            index = int(action_index)
        except ValueError:
            return
        if self._run_active:
            return
        approval = self._approval_actions.pop(index, None)
        if approval is None:
            return
        run_id, request_id, thread_id = approval
        approved = prefix == "approval-approve"
        self._waiting_for_approval = False
        self._run_active = True
        self.query_one("#status", Static).update("Recording approval and resuming…")
        self._stream_worker = asyncio.create_task(
            self._decide_and_resume(thread_id, run_id, request_id, approved=approved)
        )

    async def _decide_and_resume(
        self, thread_id: str, run_id: str, request_id: str, *, approved: bool
    ) -> None:
        try:
            await asyncio.to_thread(
                self.approval_controller.decide,
                thread_id,
                run_id,
                request_id,
                approved=approved,
            )
            resume = getattr(self.coding_session, "resume", None)
            if not callable(resume):
                raise TypeError("The coding session cannot resume a paused run")
            await self._consume_events(resume(run_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._append_transcript(f"Approval failed: {exc}", classes="error")
            self.query_one("#status", Static).update("Approval failed")
            self._run_active = False
            self._stream_worker = None

    async def action_cancel_run(self) -> None:
        hints = self.query_one("#command-hints", OptionList)
        if not self._run_active and hints.display:
            hints.display = False
            return
        if not self._run_active:
            return
        cancel = getattr(self.coding_session, "cancel", None)
        if cancel is not None:
            result = cancel()
            if asyncio.iscoroutine(result):
                await result
        worker = self._stream_worker
        if worker is not None:
            worker.cancel()
        self._run_active = False
        self._waiting_for_approval = False
        self.query_one("#status", Static).update("Cancelled")

    def action_toggle_left(self) -> None:
        self._left_visible = not self._left_visible
        self._apply_responsive_layout(self.size.width)

    def action_toggle_right(self) -> None:
        self._right_visible = not self._right_visible
        self._apply_responsive_layout(self.size.width)

    def on_text_selected(self):
        selected = self.screen.get_selected_text()
        if selected:
            self._selected_transcript_text = selected

    async def on_mouse_down(self, event):
        if event.button == 3:
            event.stop()
            await self.action_copy_transcript()

    async def action_copy_transcript(self):
        message = await self.copy_transcript("")
        self.notify(message)

    async def copy_transcript(self, arguments):
        mode, destination = parse_copy_arguments(arguments)
        selected = self.screen.get_selected_text() or self._selected_transcript_text
        rows = self.query_one("#transcript").query(Static)
        if mode == "last":
            answers = [
                str(row.content) for row in rows if "assistant-message" in row.classes
            ]
            text = answers[-1] if answers else ""
        else:
            text = (
                selected
                if mode == "selection" and selected
                else "\n\n".join(str(row.content) for row in rows)
            )
        if not text:
            return "Nothing to copy"
        if destination is not None:
            await asyncio.to_thread(export_text, destination, text)
            return f"Saved copied text to {destination}"
        if await copy_system_clipboard(text):
            return "Copied to system clipboard"
        self.copy_to_clipboard(text)
        return (
            "Copy requested from terminal (OSC52). If blocked, use /copy --file PATH."
        )

    async def _consume_prompt(self, prompt: str) -> None:
        await self._consume_events(self.coding_session.stream(prompt))
        if (
            self.host is not None
            and self.host.storage.thread_dir(self.coding_session.thread_id).exists()
        ):
            metadata = self.host.storage.read_metadata(self.coding_session.thread_id)
            self.host.storage.touch(
                self.coding_session.thread_id, title=metadata.title or prompt
            )
            await self._refresh_sessions()

    async def _consume_events(self, events: AsyncIterator[Any]) -> None:
        self._assistant_widget = None
        self._assistant_text = ""
        assistant_started = False
        try:
            if hasattr(events, "__aiter__"):
                async for event in events:
                    assistant_started = await self._render_event(
                        event, assistant_started=assistant_started
                    )
            else:
                raise TypeError("Session stream must return an async iterator")
            if not self._run_active:
                return
            if not self._waiting_for_approval:
                self.query_one("#status", Static).update("Ready")
        except asyncio.CancelledError:
            raise
        except TaskPauseRequestedError as exc:
            self._waiting_for_approval = True
            await self._append_transcript(f"Run paused: {exc}", classes="error")
            if not str(self.query_one("#status", Static).content).startswith(
                "Approval pending"
            ):
                self.query_one("#status", Static).update("Run paused")
        except Exception as exc:  # The UI must keep working after a failed run.
            await self._append_transcript(f"Error: {exc}", classes="error")
            self.query_one("#status", Static).update("Run failed")
        finally:
            self._run_active = False
            self._stream_worker = None
            self.query_one("#composer", TextArea).focus()

    async def _append_transcript(self, content: str, *, classes: str = "") -> Static:
        row = Static(content, markup=False, classes=classes)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(row)
        transcript.scroll_end(animate=False)
        return row

    async def _handle_pause_event(self, event: Any, data: Mapping[str, Any]) -> None:
        self._waiting_for_approval = True
        run_id = _event_run_id(event, data)
        if self.host is not None:
            self.approval_controller = self.host.approval_controller
        if self.approval_controller is None:
            await self._show_unreviewable_pause(run_id)
            return
        if not run_id:
            self.query_one("#status", Static).update(
                "Approval pending · event has no run id"
            )
            return

        thread_id = getattr(self.coding_session, "thread_id", "")
        pending = await asyncio.to_thread(
            self.approval_controller.pending, thread_id, run_id
        )
        if not pending:
            self.query_one("#status", Static).update(
                f"Run paused · no pending approval found ({run_id})"
            )
            return
        self.query_one("#status", Static).update(f"Approval pending · run {run_id}")
        for review in pending:
            await self._append_approval_review(thread_id, run_id, review)

    async def _show_unreviewable_pause(self, run_id: str | None) -> None:
        if run_id and run_id in self._unreviewable_paused_runs:
            return
        if run_id:
            self._unreviewable_paused_runs.add(run_id)
        self.query_one("#status", Static).update(
            "Approval pending · no approval controller configured"
        )
        await self._append_transcript(
            f"Approval required for run {run_id or '(unknown run)'}. "
            "Configure an approval controller to review this request.",
            classes="approval-card",
        )

    async def _append_approval_review(
        self, thread_id: str, run_id: str, review: Any
    ) -> None:
        request_id = _review_field(review, "request_id", "")
        approval_key = (run_id, str(request_id))
        if approval_key in self._approval_keys:
            return
        self._approval_keys.add(approval_key)
        index = self._next_approval_action
        self._next_approval_action += 1
        self._approval_actions[index] = (run_id, str(request_id), thread_id)

        tool_name = _review_field(review, "tool_name", "Unknown tool")
        expires_at = _format_expiry(_review_field(review, "expires_at", None))
        diff = _review_field(review, "diff", None)
        card = Vertical(classes="approval-card")
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(card)
        await card.mount(Static(f"Approval required · {tool_name}", markup=False))
        await card.mount(Static(f"Expires: {expires_at}", markup=False))
        if diff:
            await card.mount(
                Static(f"Prepared diff:\n{_plain_content(diff)}", markup=False)
            )
        button_row = Horizontal()
        await card.mount(button_row)
        await button_row.mount(
            Button("Approve", id=f"approval-approve-{index}", variant="success")
        )
        await button_row.mount(
            Button("Deny", id=f"approval-deny-{index}", variant="error")
        )
        transcript.scroll_end(animate=False)

    async def _render_tool_event(self, event, kind, data):
        call_id = data.get("tool_call_id")
        run_id = _event_run_id(event, data)
        source = (
            event.get("source_path", ())
            if isinstance(event, Mapping)
            else getattr(event, "source_path", ())
        )
        key = (run_id, tuple(source), call_id) if call_id else None
        task_id = data.get("task_id")
        card = self._task_cards.get(task_id) if task_id else None
        card = card or (self._tool_cards.get(key) if key is not None else None)
        if card is None:
            name = data.get("tool_name") or data.get("tool") or data.get("name") or kind
            card = ToolCard(str(name))
            await self.query_one("#transcript", VerticalScroll).mount(card)
        if key is not None:
            self._tool_cards[key] = card
        if task_id:
            self._task_cards[task_id] = card
        card.apply(kind, data)
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    async def _render_event(self, event: Any, *, assistant_started: bool) -> bool:  # noqa: C901
        """Render known runtime events and ignore unsupported event payloads."""
        kind, data = _event_parts(event)
        if kind in {"run.paused", "tool.approval_required"}:
            await self._handle_pause_event(event, data)
            return assistant_started
        if kind == "commentary.delta":
            delta = str(data.get("delta", ""))
            if delta:
                self._commentary_text += delta
                text = f"Progress: {self._commentary_text}"
                if self._commentary_widget is None:
                    self._commentary_widget = await self._append_transcript(
                        text, classes="commentary-message"
                    )
                else:
                    self._commentary_widget.update(text)
            return assistant_started
        if kind.startswith(("tool.", "task.")) or kind in {
            "message.delta",
            "message.end",
            "run.started",
        }:
            self._commentary_widget = None
            self._commentary_text = ""
        if kind == "message.delta":
            delta = data.get("delta", "")
            if delta:
                self._assistant_text += str(delta)
                content = f"Assistant: {self._assistant_text}"
                if self._assistant_widget is None:
                    self._assistant_widget = await self._append_transcript(
                        content, classes="assistant-message"
                    )
                else:
                    self._assistant_widget.update(content)
            return assistant_started or bool(delta)
        if kind == "message.end":
            content = data.get("content")
            if content and not assistant_started:
                await self._append_transcript(
                    f"Assistant: {_plain_content(content)}",
                    classes="assistant-message",
                )
            self._assistant_widget = None
            self._assistant_text = ""
            return False
        if kind.startswith(("tool.", "tool_", "task.")):
            await self._render_tool_event(event, kind, data)
            return assistant_started
        if "error" in kind:
            await self._append_transcript(
                _plain_content(data.get("error") or data.get("content") or kind),
                classes="error",
            )
            return assistant_started
        if kind in {"run.started", "run.start"}:
            self.query_one("#status", Static).update("Running")
        return assistant_started


def _event_run_id(event: Any, data: Mapping[str, Any]) -> str | None:
    if isinstance(event, Mapping):
        value = event.get("run_id")
    else:
        value = getattr(event, "run_id", None)
    value = value or data.get("run_id")
    return value if isinstance(value, str) and value else None


def _review_field(review: Any, name: str, default: Any) -> Any:
    if isinstance(review, Mapping):
        return review.get(name, default)
    return getattr(review, name, default)


def _format_expiry(value: Any) -> str:
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    return str(value if value is not None else "unknown")


def _event_parts(event: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(event, Mapping):
        kind = event.get("type", "")
        data = event.get("data", event)
    else:
        kind = getattr(event, "type", "")
        data = getattr(event, "data", {})
    return str(kind), data if isinstance(data, Mapping) else {}


def _history_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value
