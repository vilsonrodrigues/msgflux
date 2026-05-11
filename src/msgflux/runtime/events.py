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
    "emit_event",
    "emit_inbox_notification",
    "emit_model_request",
    "emit_model_response",
    "emit_task_event",
    "emit_tool_call",
    "emit_tool_error",
    "emit_tool_result",
    "emit_tool_started",
    "emit_tool_update",
]


class EventType:
    AGENT_START = "gen_ai.agent.start"
    AGENT_RESUMED = "gen_ai.agent.resumed"
    AGENT_COMPLETE = "gen_ai.agent.complete"
    AGENT_ERROR = "gen_ai.agent.error"
    AGENT_STEP = "gen_ai.agent.step"

    MODEL_REQUEST = "gen_ai.model.request"
    MODEL_RESPONSE = "gen_ai.model.response"
    MODEL_RESPONSE_CHUNK = "gen_ai.model.response.chunk"
    MODEL_REASONING = "gen_ai.model.reasoning"
    MODEL_REASONING_CHUNK = "gen_ai.model.reasoning.chunk"

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

    INBOX_NOTIFICATION = "gen_ai.inbox.notification"
    CONTROL_RECEIVED = "gen_ai.control.received"
    CHECKPOINT_SAVED = "gen_ai.checkpoint.saved"

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
