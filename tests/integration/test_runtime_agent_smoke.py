import asyncio
from pathlib import Path
from typing import Any

import pytest

import msgflux as mf
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AskUserManager,
    EventStream,
    EventType,
    PermissionManager,
    TodoManager,
)
from msgflux.sandbox import Sandbox
from msgflux.tools.builtin import (
    ApplyPatch,
    AskUser,
    FileEdit,
    FileRead,
    SendUserMessage,
    TodoWrite,
)
from msgflux.utils.msgspec import msgspec_dumps


def _tool_call_response(
    tool_name: str,
    parameters: dict[str, Any],
    *,
    call_id: str,
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    aggregator = ToolCallAggregator()
    aggregator.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(aggregator)
    response.reasoning = None
    response.metadata = {}
    return response


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _ScriptedModel:
    model_type = "chat_completion"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ModelResponse:
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        self.requests.append(kwargs)
        return self._responses.pop(0)

    async def acall(self, **kwargs: Any) -> ModelResponse:
        return self(**kwargs)


async def _resolve_external_runtime_requests(
    *,
    permission_manager: PermissionManager,
    ask_user_manager: AskUserManager,
    agent_task: asyncio.Task[Any],
) -> None:
    approved_permissions: set[str] = set()
    answered_questions: set[str] = set()

    while not agent_task.done():
        for request in permission_manager.list_pending():
            if request.request_id in approved_permissions:
                continue
            permission_manager.approve(
                request.request_id,
                reason="approved by integration smoke test",
            )
            approved_permissions.add(request.request_id)

        for request in ask_user_manager.list_pending():
            if request.request_id in answered_questions:
                continue
            ask_user_manager.answer(
                request.request_id,
                {
                    question.question: question.options[0].label
                    for question in request.questions
                },
                annotations={"source": "integration-smoke"},
            )
            answered_questions.add(request.request_id)

        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_agent_runtime_tools_permissions_events_and_sandboxes_work_together(
    tmp_path: Path,
):
    pytest.importorskip("just_bash")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    notes_path = workspace / "notes.txt"
    notes_path.write_text("alpha\nbeta\n", encoding="utf-8")
    artifact_path = tmp_path / "brief.txt"
    artifact_path.write_text("MSGFLUX-42 needs runtime validation.", encoding="utf-8")

    async def llm_query(task: str, context: str) -> str:
        """Query a secondary model with bounded context."""
        return f"{task}:{context[:10]}"

    permission_manager = PermissionManager(default_mode="ask_user", timeout=2)
    ask_user_manager = AskUserManager(timeout=2)
    todo_manager = TodoManager()
    inbox = mf.AgentInbox()
    inbox.user_message("Prefer the mounted artifact when validating context.")

    model = _ScriptedModel(
        [
            _tool_call_response(
                "todo_write",
                {
                    "todos": [
                        {
                            "content": "Validate runtime integration",
                            "active_form": "Validating runtime integration",
                            "status": "in_progress",
                        }
                    ]
                },
                call_id="call_todo",
            ),
            _tool_call_response(
                "SendUserMessage",
                {"message": "Starting runtime validation.", "status": "progress"},
                call_id="call_user_message",
            ),
            _tool_call_response(
                "ask_user",
                {
                    "questions": [
                        {
                            "question": "Proceed with file edits?",
                            "header": "Approval",
                            "options": [
                                {
                                    "label": "Proceed",
                                    "description": "Continue the smoke test.",
                                },
                                {
                                    "label": "Stop",
                                    "description": "Do not edit files.",
                                },
                            ],
                        }
                    ]
                },
                call_id="call_ask_user",
            ),
            _tool_call_response(
                "Read",
                {"file_path": str(notes_path), "limit": 20},
                call_id="call_read",
            ),
            _tool_call_response(
                "edit",
                {
                    "file_path": str(notes_path),
                    "old_string": "beta",
                    "new_string": "gamma",
                },
                call_id="call_edit",
            ),
            _tool_call_response(
                "apply_patch",
                {
                    "patch": (
                        "*** Begin Patch\n"
                        f"*** Add File: {workspace / 'summary.txt'}\n"
                        "+runtime smoke completed\n"
                        "*** End Patch"
                    )
                },
                call_id="call_apply_patch",
            ),
            _tool_call_response(
                "shell",
                {"command": "grep gamma notes.txt && cat summary.txt"},
                call_id="call_shell",
            ),
            _tool_call_response(
                "python_interpreter",
                {
                    "code": (
                        "context = await artifacts['read']('brief', offset=0, limit=32)\n"
                        "answer = await tools['llm_query']("
                        "task='summarize', context=context"
                        ")\n"
                        "print(answer)\n"
                        "result = answer"
                    )
                },
                call_id="call_python",
            ),
            _text_response("runtime integration ok"),
        ]
    )
    agent = Agent(
        name="RuntimeAgent",
        model=model,
        tools=[
            TodoWrite(todo_manager),
            SendUserMessage(),
            AskUser(ask_user_manager),
            FileRead(),
            FileEdit(),
            ApplyPatch(),
            Sandbox.shell("just-bash", fs_mode="overlay", workspace=workspace),
            llm_query,
        ],
        code_interpreter=Sandbox.python("local"),
        agent_inbox=inbox,
        permission_manager=permission_manager,
        checkpointer=mf.InMemoryCheckpointStore(),
        config={
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": ["llm_query"]},
            }
        },
    )

    scope = mf.ExecutionScope(
        session_id="session_runtime_smoke",
        run_id="run_runtime_smoke",
        permission_mode="ask_user",
    )

    with EventStream() as stream:
        agent_task = asyncio.create_task(
            agent.acall(
                "Validate runtime integration.",
                scope=scope,
                artifacts={"brief": artifact_path},
                vars={"context_hint": "runtime-smoke"},
            )
        )
        resolver_task = asyncio.create_task(
            _resolve_external_runtime_requests(
                permission_manager=permission_manager,
                ask_user_manager=ask_user_manager,
                agent_task=agent_task,
            )
        )
        result = await asyncio.wait_for(agent_task, timeout=5)
        await resolver_task
        stream.close()
        events = stream.events

    names = [event.name for event in events]

    assert result == "runtime integration ok"
    assert notes_path.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert (workspace / "summary.txt").read_text(
        encoding="utf-8"
    ) == "runtime smoke completed\n"
    assert todo_manager.get(
        session_id="session_runtime_smoke",
        namespace="RuntimeAgent",
    )[0].content == "Validate runtime integration"

    expected_events = {
        EventType.AGENT_START,
        EventType.TURN_START,
        EventType.MODEL_REQUEST,
        EventType.MODEL_RESPONSE,
        EventType.USER_MESSAGE_INJECTED,
        EventType.TODO_UPDATED,
        EventType.USER_MESSAGE_SENT,
        EventType.USER_INTERACTION_REQUESTED,
        EventType.USER_INTERACTION_ANSWERED,
        EventType.FILE_READ,
        EventType.FILE_EDIT_PROPOSED,
        EventType.FILE_EDIT_APPLIED,
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_GRANTED,
        EventType.TOOL_STARTED,
        EventType.TOOL_RESULT,
        EventType.CHECKPOINT_SAVED,
        EventType.AGENT_COMPLETE,
    }
    missing = expected_events - set(names)
    assert not missing

    assert "llm_query" not in agent.tool_library.get_tool_names()
    assert "llm_query" in agent.code_interpreter.get_tool_names()
    assert len(permission_manager.list_pending()) == 0
    assert len(ask_user_manager.list_pending()) == 0

    shell_result = next(
        event
        for event in events
        if event.name == EventType.TOOL_RESULT
        and event.attributes["tool_name"] == "shell"
    )
    assert "gamma" in shell_result.attributes["result"]
    assert "runtime smoke completed" in shell_result.attributes["result"]

    python_result = next(
        event
        for event in events
        if event.name == EventType.TOOL_RESULT
        and event.attributes["tool_name"] == "python_interpreter"
    )
    assert "summarize:MSGFLUX-42" in python_result.attributes["result"]
