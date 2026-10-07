"""Textual Pilot consumes the actual authenticated AgentService SSE stream."""

import asyncio
import socket
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
import uvicorn
from textual.widgets import Static, TextArea

from msgflux.coding import CodingCheckpointExtension
from msgflux.runtime.service.http import AgentSessionClient
from msgflux.coding.tui import CodingApp
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import AgentServiceClient, create_service_app
from msgflux.tools.builtin import BashTool, ReadFileTool


def _response(value, kind="text_generation"):
    result = ModelResponse()
    result.set_response_type(kind)
    result.add(value)
    return result


@asynccontextmanager
async def _server(service):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_service_app(service, token="pilot-test"),
            log_level="error",
            access_log=False,
            ws="none",
            lifespan="off",
            timeout_graceful_shutdown=1,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    client = AgentServiceClient(f"http://127.0.0.1:{port}", token="pilot-test")
    try:
        await _until(lambda: server.started)
        yield client
    finally:
        await client.aclose()
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            listener.close()


async def _until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.01)


def _text(app):
    return "\n".join(
        str(row.content) for row in app.query_one("#transcript").query(Static)
    )


@pytest.mark.asyncio
async def test_pilot_closes_while_service_runs_then_restores_history(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()
    model = Mock(model_type="chat_completion")
    agent = Agent(name="main", model=model, checkpoint_store=InMemoryCheckpointStore())
    agent.register_extension("coding_checkpoints", CodingCheckpointExtension())

    async def respond(**_kwargs):
        entered.set()
        await release.wait()
        return _response("survived detached frontend")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            app = CodingApp(session, observe_immediately=False)
            async with app.run_test() as pilot:
                app.query_one("#composer", TextArea).load_text("keep working")
                await pilot.press("enter")
                await asyncio.wait_for(entered.wait(), 10)
                await _until(lambda: app._active_run_id is not None)
                run_id = app._active_run_id
            assert (
                service.receipt_for_run(session.thread_id, run_id).status == "running"
            )
            release.set()
            receipt = service.receipt_for_run(session.thread_id, run_id)
            assert (
                await service.wait(session.thread_id, receipt.request_id)
            ).status == "completed"
            reopened = CodingApp(session)
            async with reopened.run_test():
                await _until(lambda: "survived detached frontend" in _text(reopened))
                assert _text(reopened).count("survived detached frontend") == 1
                assert "keep working" in _text(reopened)
                assert not reopened._run_active
            assert agent.generator.aforward.await_count == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_pilot_renders_real_bash_and_read_results(tmp_path):
    (tmp_path / "note.txt").write_text("socket fixture")
    workspace = AgentWorkspace.local(tmp_path)
    calls = ToolCallAggregator()
    calls.process(
        0, "bash-call", "bash", '{"command":"pwd; cat note.txt","timeout_ms":1000}'
    )
    calls.process(
        1, "read-call", "read", '{"path":"note.txt","offset":null,"limit":null}'
    )
    replies = iter([_response(calls, "tool_call"), _response("tools finished")])
    agent = Agent(
        name="main",
        model=Mock(model_type="chat_completion"),
        workspace=workspace,
        tools=[BashTool(), ReadFileTool()],
        checkpoint_store=InMemoryCheckpointStore(),
    )
    agent.generator.aforward = AsyncMock(side_effect=lambda **_kwargs: next(replies))
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    try:
        async with _server(service) as client:
            session = await AgentSessionClient.open(client, cwd=tmp_path)
            app = CodingApp(session, observe_immediately=False)
            async with app.run_test() as pilot:
                app.query_one("#composer", TextArea).load_text("inspect")
                await pilot.press("enter")
                await _until(lambda: "tools finished" in _text(app))
                assert "socket fixture" in _text(app)
                assert str(tmp_path) in _text(app)
                assert _text(app).count("tools finished") == 1
                await _until(lambda: not app._run_active)
    finally:
        await service.aclose()
        service.store.close()
        await workspace.aclose()
