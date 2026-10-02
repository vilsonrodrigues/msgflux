"""Exercise the coding host across workspace, checkpoint, and approval APIs."""

import os
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.coding.approval import CodingApprovalController
from msgflux.coding.workspace import open_coding_workspace
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    InMemoryApprovalStore,
)
from msgflux.tools.builtin import EditTool
from msgflux.utils.msgspec import msgspec_dumps


def _tool_response():
    calls = ToolCallAggregator()
    calls.process(
        0,
        "edit-call",
        "edit",
        msgspec_dumps({"path": "note.txt", "old": "old", "new": "new"}),
    )
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("updated")
    return response


@pytest.mark.asyncio
async def test_local_edit_requires_review_and_resumes_same_run(tmp_path):
    if os.name != "posix":
        pytest.skip("LocalWorkspaceBackend requires POSIX")
    file = tmp_path / "note.txt"
    file.write_text("old")
    workspace, registry = await open_coding_workspace(
        tmp_path, tmp_path / "workspace.sqlite3"
    )
    checkpoints = InMemoryCheckpointStore()
    journal = InMemoryApprovalStore()
    try:
        policy = AgentApprovals(journal, {"edit": "v1"}, "coding-v1")
        model = Mock(model_type="chat_completion")
        agent = Agent(
            name="coding_agent",
            model=model,
            tools=[EditTool()],
            checkpoint_store=checkpoints,
            approvals=policy,
        )
        agent.generator.aforward = AsyncMock(
            side_effect=[_tool_response(), _text_response()]
        )
        session = CodingSession(
            agent,
            checkpoint_store=checkpoints,
            scope_factory=lambda scope: replace(
                scope,
                workspace=workspace,
                permissions=workspace.permissions,
                principal="local_user",
            ),
        )
        controller = CodingApprovalController(
            agent, policy, principal=lambda: "local_user"
        )

        events = []
        with pytest.raises(TaskPauseRequestedError):
            async for event in session.stream("Update note.txt"):
                events.append(event)
        run_id = next(event.run_id for event in events if event.type == "run.paused")
        assert file.read_text() == "old"

        reviews = controller.pending(session.thread_id, run_id)
        assert len(reviews) == 1
        assert "-old" in reviews[0].diff
        assert "+new" in reviews[0].diff
        controller.decide(
            session.thread_id, run_id, reviews[0].request_id, approved=True
        )
        resumed = [event async for event in session.resume(run_id)]

        assert file.read_text() == "new"
        assert any(event.type == "run.end" for event in resumed)
        assert agent.generator.aforward.call_count == 2
    finally:
        await workspace.aclose()
        registry.close()
