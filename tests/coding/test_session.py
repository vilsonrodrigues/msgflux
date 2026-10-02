import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    ExecutionEnvironment,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
)
from msgflux.runtime.context import ExecutionScope, get_execution_context
from msgflux.runtime.events import ExecutionEvent


class _FakeAgent:
    async def stream_events(self, prompt, *, scope):
        self.prompt = prompt
        self.scope = scope
        self.context = get_execution_context()
        yield ExecutionEvent("message.delta", "now", {"text": prompt})


@pytest.mark.asyncio
async def test_session_stream_keeps_thread_and_injects_durable_store():
    agent = _FakeAgent()
    store = object()
    session = CodingSession(agent, namespace="coding", checkpoint_store=store)

    events = [event async for event in session.stream("hello")]

    assert [event.data["text"] for event in events] == ["hello"]
    assert agent.prompt == "hello"
    assert agent.scope.thread_id == session.thread_id
    assert agent.scope.namespace == "coding"
    assert agent.context["checkpoint_store"] is store
    assert session.thread_id.startswith("thd_")


@pytest.mark.asyncio
async def test_session_scope_factory_can_add_execution_environment():
    agent = _FakeAgent()
    initial = ExecutionScope(thread_id="fixed-thread", namespace="coding")
    session = CodingSession(
        agent,
        thread_id="fixed-thread",
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            abort_signal=scope.abort_signal,
            principal="ui",
        ),
    )

    [event async for event in session.stream("hello")]

    assert agent.scope.principal == "ui"
    assert agent.scope.abort_signal is not None
    assert initial.thread_id == session.thread_id


@pytest.mark.asyncio
async def test_cancel_aborts_current_stream():
    started = asyncio.Event()

    class BlockingAgent:
        async def stream_events(self, _prompt, *, scope):
            started.set()
            await scope.abort_signal.wait()
            scope.abort_signal.raise_if_aborted()
            yield ExecutionEvent("never", "now")

    session = CodingSession(BlockingAgent())
    stream = session.stream("wait")
    task = asyncio.create_task(anext(stream))
    await started.wait()
    session.cancel()

    with pytest.raises(Exception, match="cancelled"):
        await task
    await stream.aclose()


def _text_response(text):
    response = Mock(spec=ModelResponse)
    response.response_type = "text_generation"
    response.data = text
    response.reasoning = None
    response.metadata = {"model": "scripted"}
    response.consume.return_value = text
    return response


def _coding_agent(store, scripted_forward):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="coding", model=model, checkpoint_store=store)
    agent.generator.aforward = AsyncMock(side_effect=scripted_forward)
    return agent


@pytest.mark.asyncio
async def test_new_session_resumes_durable_thread_and_snapshot_history():
    store = InMemoryCheckpointStore()
    first_agent = _coding_agent(store, lambda **_kwargs: _text_response("first reply"))
    first = CodingSession(first_agent, checkpoint_store=store)
    thread_id = first.thread_id

    [event async for event in first.stream("remember the word: cobalt")]

    seen_on_second_call = {}

    async def answer_from_history(**kwargs):
        messages = kwargs["messages"]
        assert isinstance(messages, ChatMessages)
        seen_on_second_call["chat"] = messages.to_chatml()
        return _text_response("cobalt")

    second_agent = _coding_agent(store, answer_from_history)
    reconnected = CodingSession(
        second_agent,
        thread_id=thread_id,
        checkpoint_store=store,
    )

    events = [event async for event in reconnected.stream("what word?")]
    snapshot = await reconnected.snapshot()

    assert reconnected.thread_id == thread_id
    assert reconnected.namespace == "coding"
    assert any(
        item.get("role") == "user" and "cobalt" in item.get("content", "")
        for item in seen_on_second_call["chat"]
    )
    assert any(
        item.get("role") == "assistant" and item.get("content") == "first reply"
        for item in seen_on_second_call["chat"]
    )
    assert snapshot.thread_id == thread_id
    assert isinstance(snapshot.messages, ChatMessages)
    assert snapshot.messages.to_chatml()[-1]["content"] == "cobalt"
    assert any(event.type == "run.end" for event in events)


def _tool_response():
    calls = ToolCallAggregator()
    calls.process(0, "call_lookup", "lookup", '{"query":"secret"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


@pytest.mark.asyncio
async def test_resume_approval_pause_replays_tool_batch_once():
    checkpoint_store = InMemoryCheckpointStore()
    approval_store = InMemoryApprovalStore()
    calls = []

    def lookup(query):
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    policy = AgentApprovals(approval_store, {"lookup": "v1"}, "policy-v1")
    agent = Agent(
        name="coding",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoint_store,
        approvals=policy,
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_tool_response(), _text_response("done")]
    )
    session = CodingSession(
        agent,
        checkpoint_store=checkpoint_store,
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            abort_signal=scope.abort_signal,
            principal="human",
        ),
    )

    paused_events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in session.stream("lookup the secret"):
            paused_events.append(event)
    assert any(event.type == "run.paused" for event in paused_events)
    assert calls == []
    paused_run = next(
        event.run_id for event in paused_events if event.type == "run.paused"
    )
    record = approval_store.pending("coding", session.thread_id, paused_run)[0]
    paused_state = checkpoint_store.load_state("coding", session.thread_id, paused_run)
    assert paused_state["status"] == "paused"
    assert paused_state["runtime"]["extensions"]["pending_approvals"]
    await agent.adecide_approval(
        record.request_id,
        approved=True,
        decided_by="human",
    )

    resumed_events = [event async for event in session.resume(paused_run)]

    assert any(event.type == "run.end" for event in resumed_events)
    assert calls == ["secret"], [
        (event.type, event.run_id, dict(event.data)) for event in resumed_events
    ]
    assert agent.generator.aforward.call_count == 2
    assert (
        checkpoint_store.load_state("coding", session.thread_id, paused_run)["status"]
        == "completed"
    )


@pytest.mark.asyncio
async def test_workspace_files_uses_scoped_virtual_filesystem_and_live_grants():
    workspace = InMemoryWorkspace(
        "coding-workspace",
        {
            "/public.py": b"print('ok')",
            "/.env": b"TOKEN=private",
            "/src/main.py": b"pass",
            "/private/secret.py": b"secret",
        },
    )
    environment = ExecutionEnvironment(workspace)
    permissions = PermissionSet(
        resources=[
            workspace.permission("/", "filesystem.list"),
            workspace.permission("/public.py", "filesystem.read"),
            workspace.permission("/src", "filesystem.list"),
            workspace.permission("/src/main.py", "filesystem.read"),
        ]
    )
    driver = AgentWorkspace(environment, permissions=permissions)
    session = CodingSession(
        _FakeAgent(),
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            workspace=driver,
            permissions=permissions,
        ),
    )

    paths = await session.workspace_files()

    assert paths == ("/public.py", "/src/main.py")
    assert "/private/secret.py" not in paths
    assert "/.env" not in paths
    assert await session.workspace_files(limit=1) == ("/public.py",)


@pytest.mark.asyncio
async def test_early_close_releases_agent_stream_and_execution_context():
    from msgflux.runtime.context import get_execution_scope

    closed = []

    class ClosingAgent:
        async def stream_events(self, prompt, *, scope):
            try:
                yield ExecutionEvent("run.error", "now", {"error": "failed"})
            finally:
                closed.append(get_execution_scope().thread_id)

    session = CodingSession(ClosingAgent(), thread_id="close-thread")
    previous = get_execution_scope()
    stream = session.stream("run")
    await anext(stream)
    await stream.aclose()
    assert closed == ["close-thread"]
    assert get_execution_scope() == previous
    assert session._abort_signal is None
