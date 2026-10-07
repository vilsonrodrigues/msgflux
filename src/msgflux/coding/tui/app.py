"""Textual user interface for a coding session."""

from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import Mapping
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Footer, Header, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from msgflux.coding.tui.clipboard import (
    copy_system_clipboard,
    export_text,
    parse_copy_arguments,
)
from msgflux.coding.tui.commands import builtin_commands
from msgflux.coding.tui.pickers import CodingPicker
from msgflux.coding.tui.tool_cards import ToolCard
from msgflux.coding.tui.tool_cards import plain_content as _plain_content
from msgflux.runtime.service.http.records import SnapshotRecord


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
        host: Any = None,
        open_session_picker: bool = False,
        observe_immediately: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.coding_session = session
        self.host = host
        self._open_session_picker = open_session_picker
        self._observe_immediately = observe_immediately
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
        self._observer_task: asyncio.Task[None] | None = None
        self._observer_ready = asyncio.Event()
        self._command_tasks: set[asyncio.Task[None]] = set()
        self._admission_task: asyncio.Task[None] | None = None
        self._active_run_id: str | None = None
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
        await self._mount_extension_panels()
        await self._refresh_sessions()
        if self._observe_immediately:
            self._start_observer()
        if self._open_session_picker:
            await self._show_session_picker()

    def _start_observer(self) -> None:
        self._observer_ready.clear()
        if self._observer_task is not None and not self._observer_task.done():
            self._observer_task.cancel()
        self._observer_task = asyncio.create_task(self._observe_session())

    async def _ensure_observer(self) -> None:
        if self._observer_task is None or self._observer_task.done():
            self._start_observer()
        try:
            await asyncio.wait_for(self._observer_ready.wait(), timeout=10)
        except TimeoutError as exc:
            raise RuntimeError("Could not attach to the session watcher") from exc

    async def _observe_session(self) -> None:
        delay = 0.25
        while True:
            try:
                async with self.coding_session.watch() as watcher:
                    await self._replace_from_snapshot(watcher.snapshot)
                    self._observer_ready.set()
                    delay = 0.25
                    async for event in watcher:
                        await self._render_remote_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.query_one("#status", Static).update(f"Watch disconnected: {exc}")
            self._observer_ready.clear()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
            reconnect = getattr(self.host, "reconnect", None)
            if callable(reconnect):
                try:
                    replacement = await reconnect()
                    if replacement is not None:
                        self.coding_session = replacement
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.query_one("#status", Static).update(
                        f"Backend unavailable: {exc}"
                    )

    async def _replace_from_snapshot(self, snapshot: Any) -> None:  # noqa: C901
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.remove_children()
        self._assistant_widget = self._commentary_widget = None
        self._assistant_text = self._commentary_text = ""
        self._tool_cards.clear()
        self._task_cards.clear()
        self._active_run_id = None
        self._run_active = False
        for item in getattr(snapshot, "messages", None) or ():
            if not isinstance(item, Mapping):
                continue
            role = item.get("role")
            channel = item.get("channel")
            if role == "assistant":
                if channel not in {None, "commentary", "final"} or item.get(
                    "tool_calls"
                ):
                    continue
            elif role != "user":
                continue
            content = item.get("content")
            if content is None or not _plain_content(content).strip():
                continue
            if role == "assistant" and channel == "commentary":
                await self._append_transcript(
                    f"Progress: {_plain_content(content)}",
                    classes="commentary-message",
                )
            else:
                await self._append_transcript(
                    f"{role.title()}: {_plain_content(content)}",
                    classes=f"{role}-message",
                )
        for run in getattr(snapshot, "active_runs", ()):
            if not isinstance(run, Mapping):
                continue
            source_path = run.get("source_path", ())
            if run.get("parent_run_id") or len(source_path) > 1:
                continue
            run_id = run.get("run_id")
            if run_id:
                self._active_run_id = str(run_id)
                self._run_active = True
            partial = run.get("streaming_message")
            if partial:
                self._assistant_text = _plain_content(partial)
                self._assistant_widget = await self._append_transcript(
                    f"Assistant: {self._assistant_text}", classes="assistant-message"
                )
        self.query_one("#status", Static).update(
            f"Working…  (Esc to cancel) · {self._active_run_id}"
            if self._active_run_id
            else "Ready"
        )
        transcript.scroll_end(animate=False)

    async def _render_remote_event(self, event: Any) -> None:
        kind, data = _event_parts(event)
        run_id = _event_run_id(event, data)
        source_path = getattr(event, "source_path", ())
        if isinstance(event, Mapping):
            source_path = event.get("source_path", source_path)
        is_root_event = len(source_path) <= 1 and not data.get("parent_run_id")
        if (
            kind in {"run.started", "run.start", "run.resume"}
            and run_id
            and is_root_event
        ):
            self._active_run_id, self._run_active = run_id, True
        # Child tool/task events remain useful; child messages must not appear as
        # the foreground assistant response.
        if (
            kind.startswith(("message.", "commentary."))
            and self._active_run_id
            and run_id != self._active_run_id
        ):
            return
        assistant_started = self._assistant_widget is not None or bool(
            self._assistant_text
        )
        await self._render_event(event, assistant_started=assistant_started)
        terminal = {"run.end", "run.error", "run.interrupted", "run.paused"}
        if kind in terminal and is_root_event and run_id == self._active_run_id:
            self._active_run_id, self._run_active = None, False
            outcome = str(data.get("outcome", ""))
            failed = "error" in kind or outcome in {"failed", "error"}
            self.query_one("#status", Static).update(
                "Run paused · approval requires service-side review"
                if kind == "run.paused"
                else ("Run failed" if failed else "Ready")
            )
            self._assistant_widget = self._commentary_widget = None
            self._assistant_text = self._commentary_text = ""

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
        tasks = [
            t
            for t in (self._observer_task, self._admission_task, *self._command_tasks)
            if t is not None and not t.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def on_resize(self, event: App.Resize) -> None:
        self._apply_responsive_layout(event.size.width)

    def _apply_responsive_layout(self, width: int) -> None:
        self.query_one("#right-sidebar").display = self._right_visible and width > 90
        self.query_one("#left-sidebar").display = self._left_visible and width > 68

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
        if not prompt:
            return
        composer.clear()
        if prompt.startswith("/"):
            task = asyncio.create_task(self._dispatch_command(prompt))
            self._command_tasks.add(task)
            task.add_done_callback(self._command_tasks.discard)
            return
        try:
            await self._ensure_observer()
        except Exception as exc:
            await self._append_transcript(
                f"Unable to attach to session: {exc}", classes="error"
            )
            self.query_one("#status", Static).update(
                "Session unavailable · prompt not sent"
            )
            composer.load_text(prompt)
            return
        if self._admission_task is not None and not self._admission_task.done():
            self.query_one("#status", Static).update(
                "Prompt admission is still pending"
            )
            composer.load_text(prompt)
            return
        if self._active_run_id:
            task = asyncio.create_task(self._steer_prompt(self._active_run_id, prompt))
            self._command_tasks.add(task)
            task.add_done_callback(self._command_tasks.discard)
            return
        await self._append_transcript(f"You: {prompt}", classes="user-message")
        request_id = uuid.uuid4().hex
        self._run_active = True
        self.query_one("#status", Static).update("Sending…")
        self._admission_task = asyncio.create_task(
            self._admit_prompt(prompt, request_id)
        )

    async def _admit_prompt(self, prompt: str, request_id: str) -> None:
        try:
            receipt = await self.coding_session.prompt(prompt, request_id=request_id)
            if self._active_run_id == receipt.run_id:
                self.query_one("#status", Static).update(
                    f"Working…  (Esc to cancel) · {receipt.run_id}"
                )
            elif self._run_active and self._active_run_id is None:
                self.query_one("#status", Static).update(
                    "Accepted · awaiting execution event"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._append_transcript(
                f"Unable to send prompt (request {request_id}): {exc}. "
                "Reconnecting to current session state.",
                classes="error",
            )
            self._run_active = False
            self.query_one("#status", Static).update(
                "Prompt admission failed · reconciling"
            )
            self._start_observer()

    async def _steer_prompt(self, run_id: str, prompt: str) -> None:
        try:
            await self.coding_session.steer(run_id, prompt)
            await self._append_transcript(f"You: {prompt}", classes="user-message")
        except Exception as exc:
            await self._append_transcript(
                f"Unable to steer run: {exc}", classes="error"
            )

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
        threads = await self.host.threads() if self.host is not None else ()
        count = len(threads) if self.host is not None else 1
        self.query_one("#session-list", Static).update(
            f"{thread_id}\n{count} saved · /resume"
        )

    async def _show_session_picker(self):
        if self.host is None:
            raise ValueError("This app has no session host")
        threads = await self.host.threads()
        if not threads:
            return "No saved sessions in this workspace"
        await self.push_screen(
            CodingPicker(
                "Resume session",
                tuple(
                    (
                        item.thread_id,
                        f"{item.thread_id}  ·  {getattr(item, 'cwd', 'workspace')}",
                    )
                    for item in threads
                ),
            ),
            self._session_selected,
        )

    async def _session_selected(self, thread_id):
        if thread_id is None:
            return
        try:
            await self._switch_session(thread_id)
        except Exception as exc:
            await self._append_transcript(f"Unable to resume: {exc}", classes="error")

    async def _switch_session(self, thread_id):
        if self.host is None:
            raise ValueError("This app has no session host")
        old = self._observer_task
        if old is not None:
            old.cancel()
            await asyncio.gather(old, return_exceptions=True)
        try:
            session, _controller = await self.host.select(thread_id)
        except Exception:
            if self._observe_immediately:
                self._start_observer()
            raise
        self.coding_session = session
        self.workspace = session.workspace_root
        self.query_one("#workspace-path", Static).update(self.workspace)
        self.query_one("#session-list", Static).update(str(session.thread_id))
        self._waiting_for_approval = False
        self._tool_cards.clear()
        self._task_cards.clear()
        await self._refresh_sessions()
        self._observe_immediately = not getattr(self.host, "is_new", False)
        if self._observe_immediately:
            self._start_observer()
        else:
            await self._replace_from_snapshot(
                SnapshotRecord(thread_id=session.thread_id)
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
                self.query_one("#status", Static).update(
                    "Working…" if self._active_run_id else "Ready"
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._append_transcript(f"Command failed: {exc}", classes="error")
            self.query_one("#status", Static).update(
                "Working…" if self._active_run_id else "Command failed"
            )
        finally:
            self.query_one("#composer", TextArea).focus()

    async def action_cancel_run(self) -> None:
        hints = self.query_one("#command-hints", OptionList)
        if not self._active_run_id:
            if hints.display:
                hints.display = False
            return
        try:
            cancelled = await self.coding_session.cancel(self._active_run_id)
            self.query_one("#status", Static).update(
                "Cancellation requested" if cancelled else "Run is no longer active"
            )
        except Exception as exc:
            await self._append_transcript(
                f"Unable to cancel run: {exc}", classes="error"
            )

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

    async def _append_transcript(self, content: str, *, classes: str = "") -> Static:
        row = Static(content, markup=False, classes=classes)
        transcript = self.query_one("#transcript", VerticalScroll)
        await transcript.mount(row)
        transcript.scroll_end(animate=False)
        return row

    async def _handle_pause_event(self, event: Any, data: Mapping[str, Any]) -> None:
        run_id = _event_run_id(event, data)
        await self._append_transcript(
            f"Run {run_id or '(unknown)'} is paused pending service-side approval.",
            classes="approval-card",
        )

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
            if content:
                rendered = f"Assistant: {_plain_content(content)}"
                if assistant_started and self._assistant_widget is not None:
                    self._assistant_widget.update(rendered)
                else:
                    self._assistant_widget = await self._append_transcript(
                        rendered, classes="assistant-message"
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
