"""Review cards for server owned remote tool approvals."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from msgflux.coding.tui.tool_cards import plain_content


def field(review: Any, name: str, default: Any = None) -> Any:
    if isinstance(review, dict):
        return review.get(name, default)
    return getattr(review, name, default)


def identity(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


class ApprovalCard(Vertical):
    """A safe review presentation with explicit decision controls."""

    def __init__(self, run_id: str, review: Any):
        self.run_id = run_id
        self.request_id = str(field(review, "request_id", ""))
        self.tool_call_id = str(field(review, "tool_call_id", ""))
        self.tool_name = str(field(review, "tool_name", "tool"))
        self.status = str(field(review, "status", "unknown"))
        self.revision = int(field(review, "revision", 0))
        self.diff = field(review, "diff")
        self.busy = False
        rid = identity(self.request_id)
        super().__init__(id=f"approval-{rid}", classes="approval-card")

    def compose(self):
        yield Static(
            self._content(),
            id=f"approval-content-{identity(self.request_id)}",
            markup=False,
        )
        with Horizontal(
            id=f"approval-controls-{identity(self.request_id)}",
        ):
            yield Button(
                "Approve", id=f"approve-{identity(self.request_id)}", disabled=self.busy
            )
            yield Button(
                "Deny", id=f"deny-{identity(self.request_id)}", disabled=self.busy
            )

    def update_review(self, review: Any) -> None:
        self.tool_call_id = str(field(review, "tool_call_id", self.tool_call_id))
        self.tool_name = str(field(review, "tool_name", self.tool_name))
        self.status = str(field(review, "status", self.status))
        self.revision = int(field(review, "revision", self.revision))
        self.diff = field(review, "diff", self.diff)
        if self.is_mounted:
            self.call_after_refresh(self._refresh_card)

    def on_mount(self) -> None:
        self.call_after_refresh(self._refresh_card)

    def _refresh_card(self) -> None:
        if not self.is_mounted:
            return
        status = self.status.lower()
        pending = status == "pending"
        try:
            content = self.query_one(
                f"#approval-content-{identity(self.request_id)}", Static
            )
            controls = self.query_one(f"#approval-controls-{identity(self.request_id)}")
        except NoMatches:
            # A queued refresh may run while the transcript removes our children.
            return
        content.update(self._content())
        controls.display = pending
        for button in self.query(Button):
            button.disabled = self.busy

    def _content(self) -> str:
        safe_diff = (
            plain_content(self.diff) if self.diff else "No verified diff was supplied."
        )
        return (
            f"Tool: {self.tool_name} · {self.status}\n"
            f"Tool call: {self.tool_call_id}\n"
            f"Request: {self.request_id} · revision {self.revision}\n\n"
            f"{safe_diff}"
        )

    def set_busy(self, *, busy: bool) -> None:
        self.busy = busy
        if self.is_mounted:
            self.call_after_refresh(self._refresh_card)
