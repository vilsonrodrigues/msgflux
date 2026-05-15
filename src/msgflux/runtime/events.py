from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from msgtrace.sdk import EventStream, StreamEvent, add_event

from msgflux.context import get_execution_scope

TOOL_CALL_METADATA_KEY = "_tool_call_metadata"

__all__ = [
    "EventStream",
    "EventType",
    "StreamEvent",
    "TOOL_CALL_METADATA_KEY",
    "ToolCallMetadata",
    "emit_agent_complete",
    "emit_agent_error",
    "emit_agent_resumed",
    "emit_agent_start",
    "emit_checkpoint_saved",
    "emit_checkpoint_loaded",
    "emit_compaction_post",
    "emit_compaction_pre",
    "emit_event",
    "emit_file_read",
    "emit_file_edit_applied",
    "emit_file_edit_failed",
    "emit_file_edit_proposed",
    "emit_file_edit_rejected",
    "emit_inbox_notification",
    "emit_model_request",
    "emit_model_response",
    "emit_permission_denied",
    "emit_permission_granted",
    "emit_permission_requested",
    "emit_subagent_complete",
    "emit_subagent_error",
    "emit_subagent_start",
    "emit_task_event",
    "emit_todo_updated",
    "emit_tool_call",
    "emit_tool_error",
    "emit_tool_result",
    "emit_tool_started",
    "emit_tool_update",
    "emit_turn_complete",
    "emit_turn_error",
    "emit_turn_start",
    "emit_user_interaction_answered",
    "emit_user_interaction_cancelled",
    "emit_user_interaction_requested",
    "emit_user_message_injected",
    "emit_user_message_received",
    "emit_user_message_sent",
]


class EventType:
    AGENT_START = "gen_ai.agent.start"
    AGENT_RESUMED = "gen_ai.agent.resumed"
    AGENT_COMPLETE = "gen_ai.agent.complete"
    AGENT_ERROR = "gen_ai.agent.error"
    AGENT_STEP = "gen_ai.agent.step"

    TURN_START = "gen_ai.turn.start"
    TURN_COMPLETE = "gen_ai.turn.complete"
    TURN_ERROR = "gen_ai.turn.error"

    USER_MESSAGE_RECEIVED = "gen_ai.user_message.received"
    USER_MESSAGE_INJECTED = "gen_ai.user_message.injected"
    USER_MESSAGE_SENT = "gen_ai.user_message.sent"

    FILE_READ = "gen_ai.file.read"
    FILE_EDIT_PROPOSED = "gen_ai.file.edit.proposed"
    FILE_EDIT_APPLIED = "gen_ai.file.edit.applied"
    FILE_EDIT_REJECTED = "gen_ai.file.edit.rejected"
    FILE_EDIT_FAILED = "gen_ai.file.edit.failed"

    MODEL_REQUEST = "gen_ai.model.request"
    MODEL_RESPONSE = "gen_ai.model.response"
    MODEL_RESPONSE_CHUNK = "gen_ai.model.response.chunk"
    MODEL_REASONING = "gen_ai.model.reasoning"
    MODEL_REASONING_CHUNK = "gen_ai.model.reasoning.chunk"

    PERMISSION_REQUESTED = "gen_ai.permission.requested"
    PERMISSION_GRANTED = "gen_ai.permission.granted"
    PERMISSION_DENIED = "gen_ai.permission.denied"

    USER_INTERACTION_REQUESTED = "gen_ai.user_interaction.requested"
    USER_INTERACTION_ANSWERED = "gen_ai.user_interaction.answered"
    USER_INTERACTION_CANCELLED = "gen_ai.user_interaction.cancelled"

    TOOL_CALL = "gen_ai.tool.call"
    TOOL_STARTED = "gen_ai.tool.started"
    TOOL_RESULT = "gen_ai.tool.result"
    TOOL_ERROR = "gen_ai.tool.error"
    TOOL_UPDATE = "gen_ai.tool.update"

    TASK_CREATED = "gen_ai.task.created"
    TASK_RUNNING = "gen_ai.task.running"
    TASK_PROGRESS = "gen_ai.task.progress"
    TASK_PAUSED = "gen_ai.task.paused"
    TASK_STOPPED = "gen_ai.task.stopped"
    TASK_COMPLETED = "gen_ai.task.completed"
    TASK_FAILED = "gen_ai.task.failed"
    TASK_STOP_REQUESTED = "gen_ai.task.stop_requested"
    TASK_REQUEUED = "gen_ai.task.requeued"

    TODO_UPDATED = "gen_ai.todo.updated"

    INBOX_NOTIFICATION = "gen_ai.inbox.notification"
    CONTROL_RECEIVED = "gen_ai.control.received"
    CHECKPOINT_SAVED = "gen_ai.checkpoint.saved"
    CHECKPOINT_LOADED = "gen_ai.checkpoint.loaded"
    COMPACTION_PRE = "gen_ai.compaction.pre"
    COMPACTION_POST = "gen_ai.compaction.post"

    SUBAGENT_START = "gen_ai.subagent.start"
    SUBAGENT_COMPLETE = "gen_ai.subagent.complete"
    SUBAGENT_ERROR = "gen_ai.subagent.error"

    MODULE_START = "gen_ai.module.start"
    MODULE_COMPLETE = "gen_ai.module.complete"
    MODULE_ERROR = "gen_ai.module.error"

    FLOW_STEP = "gen_ai.flow.step"
    FLOW_REASONING = "gen_ai.flow.reasoning"
    FLOW_COMPLETE = "gen_ai.flow.complete"


@dataclass(frozen=True)
class ToolCallMetadata:
    tool_call_id: str | None = None
    tool_name: str | None = None
    caller_name: str | None = None
    caller_namespace: str | None = None
    caller_session_id: str | None = None
    caller_run_id: str | None = None
    caller_root_run_id: str | None = None
    step: int | None = None
    task_id: str | None = None
    task_kind: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != {}
        }


def _scope_attributes() -> dict[str, Any]:
    scope = get_execution_scope()
    return {
        "session_id": scope.session_id,
        "namespace": scope.namespace,
        "run_id": scope.run_id,
        "parent_run_id": scope.parent_run_id,
        "root_run_id": scope.root_run_id,
    }


def _clean_attributes(attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {key: _json_safe(value) for key, value in dict(attributes or {}).items()}
    payload["scope"] = {
        key: value for key, value in _scope_attributes().items() if value is not None
    }
    return {key: value for key, value in payload.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)


def emit_event(name: str, attributes: Mapping[str, Any] | None = None) -> None:
    add_event(name, _clean_attributes(attributes))


def emit_agent_start(agent_name: str, *, resumed: bool = False) -> None:
    emit_event(EventType.AGENT_START, {"agent_name": agent_name, "resumed": resumed})


def emit_agent_resumed(agent_name: str, *, status: str | None = None) -> None:
    emit_event(EventType.AGENT_RESUMED, {"agent_name": agent_name, "status": status})


def emit_agent_complete(agent_name: str, *, response_type: str | None = None) -> None:
    emit_event(
        EventType.AGENT_COMPLETE,
        {"agent_name": agent_name, "response_type": response_type},
    )


def emit_agent_error(agent_name: str, error: BaseException | str) -> None:
    emit_event(EventType.AGENT_ERROR, {"agent_name": agent_name, "error": str(error)})


def emit_model_request(agent_name: str, *, message_count: int | None = None) -> None:
    emit_event(
        EventType.MODEL_REQUEST,
        {"agent_name": agent_name, "message_count": message_count},
    )


def emit_model_response(agent_name: str, *, response_type: str | None = None) -> None:
    emit_event(
        EventType.MODEL_RESPONSE,
        {"agent_name": agent_name, "response_type": response_type},
    )


def emit_turn_start(agent_name: str, *, resumed: bool = False) -> None:
    emit_event(EventType.TURN_START, {"agent_name": agent_name, "resumed": resumed})


def emit_turn_complete(agent_name: str, *, response_type: str | None = None) -> None:
    emit_event(
        EventType.TURN_COMPLETE,
        {"agent_name": agent_name, "response_type": response_type},
    )


def emit_turn_error(agent_name: str, error: BaseException | str) -> None:
    emit_event(EventType.TURN_ERROR, {"agent_name": agent_name, "error": str(error)})


def emit_user_message_received(
    *,
    notification_id: str | None = None,
    source: str = "incoming_user_message",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {"notification_id": notification_id, "source": source}
    if metadata:
        payload["metadata"] = dict(metadata)
    emit_event(EventType.USER_MESSAGE_RECEIVED, payload)


def emit_user_message_injected(
    *,
    message_count: int,
    notification_ids: list[str] | None = None,
) -> None:
    emit_event(
        EventType.USER_MESSAGE_INJECTED,
        {"message_count": message_count, "notification_ids": notification_ids or []},
    )


def emit_user_message_sent(
    message: str,
    *,
    status: str = "info",
    attachments: list[str] | None = None,
) -> None:
    payload = {"message": message, "status": status}
    if attachments:
        payload["attachments"] = list(attachments)
    emit_event(EventType.USER_MESSAGE_SENT, payload)


def emit_file_read(
    *,
    path: str,
    line_start: int,
    line_end: int,
    lines_returned: int,
    chars_returned: int,
    truncated: bool,
    reason: str | None = None,
) -> None:
    emit_event(
        EventType.FILE_READ,
        {
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
            "lines_returned": lines_returned,
            "chars_returned": chars_returned,
            "truncated": truncated,
            "reason": reason,
        },
    )


def _emit_file_edit_event(event_type: str, attributes: Mapping[str, Any]) -> None:
    emit_event(event_type, attributes)


def emit_file_edit_proposed(attributes: Mapping[str, Any]) -> None:
    _emit_file_edit_event(EventType.FILE_EDIT_PROPOSED, attributes)


def emit_file_edit_applied(attributes: Mapping[str, Any]) -> None:
    _emit_file_edit_event(EventType.FILE_EDIT_APPLIED, attributes)


def emit_file_edit_rejected(attributes: Mapping[str, Any]) -> None:
    _emit_file_edit_event(EventType.FILE_EDIT_REJECTED, attributes)


def emit_file_edit_failed(attributes: Mapping[str, Any]) -> None:
    _emit_file_edit_event(EventType.FILE_EDIT_FAILED, attributes)


def emit_permission_requested(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.PERMISSION_REQUESTED, attributes)


def emit_permission_granted(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.PERMISSION_GRANTED, attributes)


def emit_permission_denied(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.PERMISSION_DENIED, attributes)


def emit_user_interaction_requested(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.USER_INTERACTION_REQUESTED, attributes)


def emit_user_interaction_answered(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.USER_INTERACTION_ANSWERED, attributes)


def emit_user_interaction_cancelled(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.USER_INTERACTION_CANCELLED, attributes)


def emit_tool_call(metadata: ToolCallMetadata) -> None:
    emit_event(EventType.TOOL_CALL, metadata.to_dict())


def emit_tool_started(metadata: ToolCallMetadata) -> None:
    emit_event(EventType.TOOL_STARTED, metadata.to_dict())


def emit_tool_result(metadata: ToolCallMetadata, result: Any) -> None:
    payload = metadata.to_dict()
    payload["result"] = result
    emit_event(EventType.TOOL_RESULT, payload)


def emit_tool_error(metadata: ToolCallMetadata, error: BaseException | str) -> None:
    payload = metadata.to_dict()
    payload["error"] = str(error)
    emit_event(EventType.TOOL_ERROR, payload)


def emit_tool_update(
    metadata: ToolCallMetadata | None,
    *,
    status: str,
    hint: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> None:
    payload = metadata.to_dict() if metadata is not None else {}
    payload.update({"status": status, "hint": hint})
    if data:
        payload["data"] = dict(data)
    emit_event(EventType.TOOL_UPDATE, payload)


def _subagent_payload(metadata: ToolCallMetadata) -> dict[str, Any]:
    payload = metadata.to_dict()
    if metadata.tool_name:
        payload["subagent_name"] = metadata.tool_name
    return payload


def emit_subagent_start(metadata: ToolCallMetadata) -> None:
    emit_event(EventType.SUBAGENT_START, _subagent_payload(metadata))


def emit_subagent_complete(metadata: ToolCallMetadata, result: Any = None) -> None:
    payload = _subagent_payload(metadata)
    if result is not None:
        payload["result"] = result
    emit_event(EventType.SUBAGENT_COMPLETE, payload)


def emit_subagent_error(metadata: ToolCallMetadata, error: BaseException | str) -> None:
    payload = _subagent_payload(metadata)
    payload["error"] = str(error)
    emit_event(EventType.SUBAGENT_ERROR, payload)


def emit_task_event(
    event_type: str,
    *,
    task_id: str,
    tool_name: str | None = None,
    status: str | None = None,
    data: Mapping[str, Any] | None = None,
) -> None:
    payload = {"task_id": task_id, "tool_name": tool_name, "status": status}
    if data:
        payload["data"] = dict(data)
    emit_event(event_type, payload)


def emit_todo_updated(attributes: Mapping[str, Any]) -> None:
    emit_event(EventType.TODO_UPDATED, attributes)


def emit_inbox_notification(
    *,
    source: str,
    status: str | None = None,
    ref: str | None = None,
    notification_id: str | None = None,
) -> None:
    emit_event(
        EventType.INBOX_NOTIFICATION,
        {
            "source": source,
            "status": status,
            "ref": ref,
            "notification_id": notification_id,
        },
    )


def emit_checkpoint_saved(
    *,
    namespace: str,
    session_id: str,
    run_id: str,
    status: str,
) -> None:
    emit_event(
        EventType.CHECKPOINT_SAVED,
        {
            "checkpoint_namespace": namespace,
            "checkpoint_session_id": session_id,
            "checkpoint_run_id": run_id,
            "status": status,
        },
    )


def emit_checkpoint_loaded(
    *,
    namespace: str,
    session_id: str,
    run_id: str,
    status: str | None = None,
) -> None:
    emit_event(
        EventType.CHECKPOINT_LOADED,
        {
            "checkpoint_namespace": namespace,
            "checkpoint_session_id": session_id,
            "checkpoint_run_id": run_id,
            "status": status,
        },
    )


def emit_compaction_pre(
    *,
    target: str | None = None,
    strategy: str | None = None,
    message_count: int | None = None,
    token_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "target": target,
        "strategy": strategy,
        "message_count": message_count,
        "token_count": token_count,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    emit_event(EventType.COMPACTION_PRE, payload)


def emit_compaction_post(
    *,
    target: str | None = None,
    strategy: str | None = None,
    message_count_before: int | None = None,
    message_count_after: int | None = None,
    token_count_before: int | None = None,
    token_count_after: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "target": target,
        "strategy": strategy,
        "message_count_before": message_count_before,
        "message_count_after": message_count_after,
        "token_count_before": token_count_before,
        "token_count_after": token_count_after,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    emit_event(EventType.COMPACTION_POST, payload)
