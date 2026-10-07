"""One model-facing tool for controlling background tasks."""

from __future__ import annotations

from typing import Any, Union

import msgspec

from msgflux.tools.types import Hidden, ToolBucket, ToolLibraryOperator


class _TaskStatusRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="status",
    tag_field="action",
):
    task_id: str


class _TaskListRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="list",
    tag_field="action",
):
    status: str | None


class _TaskOutputRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="output",
    tag_field="action",
):
    task_id: str


class _TaskWaitRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="wait",
    tag_field="action",
):
    task_id: str
    timeout: float | None


class _TaskInterruptRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="interrupt",
    tag_field="action",
):
    task_id: str


class _TaskActivityRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="activity",
    tag_field="action",
):
    task_id: str
    limit: int | None


class _TaskMessageRequest(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="message",
    tag_field="action",
):
    task_id: str
    message: str


_REQUEST_TYPES = {
    "task_status": _TaskStatusRequest,
    "task_list": _TaskListRequest,
    "task_output": _TaskOutputRequest,
    "task_wait": _TaskWaitRequest,
    "task_interrupt": _TaskInterruptRequest,
    "task_activity": _TaskActivityRequest,
    "task_message": _TaskMessageRequest,
}
_TaskRequest = Union[tuple(_REQUEST_TYPES.values())]
_ACTION_TO_TOOL = {
    "status": "task_status",
    "list": "task_list",
    "output": "task_output",
    "wait": "task_wait",
    "interrupt": "task_interrupt",
    "activity": "task_activity",
    "message": "task_message",
}


class TaskTool(ToolBucket, ToolLibraryOperator):
    """Inspect, wait for, interrupt, or message a background task.

    Use `action` to select the operation. `request` carries the arguments for
    that operation: status/output/interrupt take `task_id`; list accepts an
    nullable `status`; wait accepts an nullable `timeout`; activity accepts an
    nullable `limit`; and message requires `task_id` and `message`.
    """

    name = "task"
    display_name = "Tasks"
    usage_guidance = (
        "All task controls use this tool's request.action. Background launch "
        "results may mention internal names such as task_wait or task_message; "
        "invoke task with action wait or message instead of those hidden tools. "
        "Use a bounded timeout when waiting. Nullable fields may be null."
    )
    capture = {
        "tool_kind": "background|background_activity|background_message",
        "defer_loading": False,
    }
    description = (
        "Control background tasks. Available actions: status, list, output, wait, "
        "interrupt, activity, message."
    )
    annotations = {"request": _TaskRequest, "handle": Hidden, "return": Any}

    def __call__(self, request: _TaskRequest, *, handle: Hidden) -> Any:
        normalized = msgspec.convert(request, type=_TaskRequest, strict=True)
        payload = msgspec.to_builtins(normalized)
        action = payload.pop("action")
        return self._dispatch(action, payload, handle)

    async def acall(self, request: _TaskRequest, *, handle: Hidden) -> Any:
        normalized = msgspec.convert(request, type=_TaskRequest, strict=True)
        payload = msgspec.to_builtins(normalized)
        action = payload.pop("action")
        return await self._adispatch(action, payload, handle)

    def patch_schema_annotations(self, annotations):
        available = [
            request_type
            for tool_name, request_type in _REQUEST_TYPES.items()
            if tool_name in self.tools
        ]
        if available:
            annotations = dict(annotations)
            annotations["request"] = Union[tuple(available)]
        return annotations

    def _dispatch(self, action: str, payload: dict[str, Any], handle: Any) -> Any:
        if action == "activity" and payload.get("limit") is None:
            payload["limit"] = 10
        tool_name = _ACTION_TO_TOOL[action]
        if tool_name not in self.tools:
            return {
                "status": "unsupported",
                "error": f"The `{action}` task action is unavailable.",
            }
        return handle(tool_name, **payload)

    async def _adispatch(
        self,
        action: str,
        payload: dict[str, Any],
        handle: Any,
    ) -> Any:
        if action == "activity" and payload.get("limit") is None:
            payload["limit"] = 10
        tool_name = _ACTION_TO_TOOL[action]
        if tool_name not in self.tools:
            return {
                "status": "unsupported",
                "error": f"The `{action}` task action is unavailable.",
            }
        return await handle.acall(tool_name, **payload)

    def refresh(self, entries=()) -> None:
        actions = {entry.name: entry.description for entry in entries}
        ordered = [
            ("status", "task_status"),
            ("list", "task_list"),
            ("output", "task_output"),
            ("wait", "task_wait"),
            ("interrupt", "task_interrupt"),
            ("activity", "task_activity"),
            ("message", "task_message"),
        ]
        available = [
            f"{action} ({actions[name]})" for action, name in ordered if name in actions
        ]
        if available:
            self.description = "Control background tasks. Actions: " + "; ".join(
                available
            )
        else:
            self.description = "Control background tasks."


__all__ = ["TaskTool"]
