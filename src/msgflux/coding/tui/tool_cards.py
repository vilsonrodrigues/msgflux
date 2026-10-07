"""Readable tool lifecycle cards, shared by live and restored transcripts."""

import json
from collections.abc import Mapping

import msgspec
from textual.widgets import Static

_MAX_TEXT = 4000


def plain_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(plain_content(item) for item in value)
    if isinstance(value, Mapping):
        if "results" in value and isinstance(value["results"], (list, tuple)):
            return shell_text(value)
        text = value.get("text", value.get("content"))
        if text is not None:
            return plain_content(text)
        return json.dumps(dict(value), ensure_ascii=False, indent=2, default=str)
    try:
        builtins = msgspec.to_builtins(value)
    except (TypeError, ValueError, RecursionError):
        return str(value)
    return (
        plain_content(builtins)
        if isinstance(builtins, (dict, list, tuple))
        else str(builtins)
    )


def bounded_text(value, limit=_MAX_TEXT):
    text = plain_content(value)
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def shell_text(value):
    parts = []
    for result in value["results"]:
        if not isinstance(result, Mapping):
            parts.append(str(result))
            continue
        status = result.get("status", "")
        code = result.get("returncode")
        parts.append(f"{status} · exit {code}" if code is not None else str(status))
        if result.get("stdout"):
            parts.append(str(result["stdout"]).rstrip())
        if result.get("stderr"):
            parts.append("stderr:\n" + str(result["stderr"]).rstrip())
    if value.get("output_reference") is not None:
        parts.append("Full output: " + plain_content(value["output_reference"]))
    return "\n".join(parts)


class ToolCard(Static):
    """Update one row without losing its launch arguments or falsey result."""

    def __init__(self, name: str):
        super().__init__("", markup=False, classes="tool-card")
        self.tool_name = name
        self.arguments = None
        self.result = None
        self.has_result = False
        self.error = None
        self.status = "running"
        self.task_id = None
        self.background = False

    def apply(self, kind, data):
        if "arguments" in data:
            self.arguments = data["arguments"]
        if "result" in data:
            self.result, self.has_result = data["result"], True
        elif "output" in data:
            self.result, self.has_result = data["output"], True
        if data.get("error") is not None:
            self.error = data["error"]
        if data.get("task_id"):
            self.task_id = data["task_id"]
            self.background = True
        if kind.endswith(".end"):
            self.status = data.get("status") or (
                "failed" if self.error else "completed"
            )
        elif "blocked" in kind or "permission_denied" in kind:
            self.status = "blocked"
        elif data.get("status"):
            self.status = data["status"]
        self._refresh_content()

    def _refresh_content(self):
        label = self.tool_name + (" (background)" if self.background else "")
        sections = [f"Tool: {label} · {self.status}"]
        if self.arguments is not None:
            if isinstance(self.arguments, Mapping) and "command" in self.arguments:
                sections.append("$ " + plain_content(self.arguments["command"]))
            else:
                sections.append(bounded_text(self.arguments))
        if self.has_result:
            sections.append(bounded_text(self.result))
        if self.error is not None:
            sections.append("Error: " + bounded_text(self.error))
        if self.task_id:
            sections.append(f"Task: {self.task_id}")
        self.update("\n".join(sections))
