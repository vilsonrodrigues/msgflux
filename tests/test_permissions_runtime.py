import asyncio
from threading import Thread

import pytest

import msgflux as mf
from msgflux.context import ExecutionScope, execution_context, get_execution_context
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolLibrary
from msgflux.nn.modules import tool as tool_module
from msgflux.runtime import EventStream, EventType
from msgflux.runtime.permissions import (
    PermissionDeniedError,
    PermissionManager,
    PermissionTimeoutError,
)
from msgflux.utils.msgspec import msgspec_dumps


def _tool_call_response(
    tool_name: str,
    parameters: dict,
    *,
    call_id: str = "call_1",
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(agg)
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
    def __init__(self, responses):
        self.model_type = "chat_completion"
        self._responses = list(responses)

    def __call__(self, **kwargs):
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        return self._responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


@pytest.mark.asyncio
async def test_permission_manager_bypass_emits_request_and_grant_events():
    manager = PermissionManager(policy="bypass")

    with EventStream() as stream:
        decision = await manager.request(
            "file.write",
            resource="workspace/report.txt",
            tool_name="write_file",
            caller_name="Assistant",
            risk="high",
        )
        stream.close()
        events = stream.events

    assert decision.approved is True
    assert [event.name for event in events] == [
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_GRANTED,
    ]
    requested = events[0].attributes
    assert requested["action"] == "file.write"
    assert requested["policy"] == "bypass"
    assert requested["tool_name"] == "write_file"
    assert requested["caller_name"] == "Assistant"


@pytest.mark.asyncio
async def test_permission_manager_deny_policy_can_be_enforced():
    manager = PermissionManager(policy="deny")

    with pytest.raises(PermissionDeniedError, match="denied by policy"):
        await manager.enforce("file.write", resource="workspace/report.txt")


@pytest.mark.asyncio
async def test_permission_manager_ask_user_can_be_approved_externally():
    manager = PermissionManager(policy="ask_user")

    with EventStream() as stream:
        task = asyncio.create_task(manager.request("file.write"))
        await asyncio.sleep(0)

        [pending] = manager.list_pending()
        decision = manager.approve(pending.request_id, reason="approved by test")
        result = await task

        stream.close()
        events = stream.events

    assert decision.approved is True
    assert result == decision
    assert [event.name for event in events] == [
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_GRANTED,
    ]


@pytest.mark.asyncio
async def test_permission_manager_approval_is_thread_safe():
    manager = PermissionManager(policy="ask_user")
    task = asyncio.create_task(manager.request("file.write"))
    await asyncio.sleep(0)

    [pending] = manager.list_pending()
    thread = Thread(target=lambda: manager.approve(pending.request_id))
    thread.start()
    thread.join()

    decision = await asyncio.wait_for(task, timeout=1)

    assert decision.approved is True


@pytest.mark.asyncio
async def test_permission_manager_ask_user_timeout_raises_and_denies():
    manager = PermissionManager(policy="ask_user", timeout=0.001)

    with EventStream() as stream:
        with pytest.raises(PermissionTimeoutError):
            await manager.request("file.write")
        stream.close()
        events = stream.events

    assert manager.list_pending() == []
    assert [event.name for event in events] == [
        EventType.PERMISSION_REQUESTED,
        EventType.PERMISSION_DENIED,
    ]


@pytest.mark.asyncio
async def test_permission_policy_can_change_per_session_context():
    manager = PermissionManager(policy="ask_user")

    with execution_context(permission_manager=manager):
        ctx_manager = get_execution_context()["permission_manager"]
        assert ctx_manager is manager

        manager.set_policy("bypass")
        decision = await ctx_manager.request("file.write", tool_name="write_file")

    assert decision.approved is True


@pytest.mark.asyncio
async def test_permission_policy_context_manager_restores_previous_policy():
    manager = PermissionManager(policy="ask_user")

    with manager.use_policy("bypass"):
        assert manager.policy == "bypass"
        decision = await manager.request("file.write")

    assert decision.approved is True
    assert manager.policy == "ask_user"


def test_agent_exposes_default_permission_manager_to_tools():
    seen_policies = []

    def read_permission_policy() -> str:
        """Read the active permission policy."""
        manager = get_execution_context()["permission_manager"]
        seen_policies.append(manager.policy)
        return manager.policy

    manager = PermissionManager(policy="ask_user")
    model = _ScriptedModel(
        [
            _tool_call_response("read_permission_policy", {}),
            _text_response("ok"),
        ]
    )
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[read_permission_policy],
        permission_manager=manager,
    )
    agent.set_permission_policy("bypass")

    agent("Read the active permission policy.")

    assert seen_policies == ["bypass"]


def test_tool_permission_uses_scope_mode_over_manager_default():
    @mf.tool_config(
        permission={
            "action": "file.write",
            "risk": "high",
            "resource_arg": "path",
        }
    )
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return f"{path}:{content}"

    manager = PermissionManager(default_mode="deny")
    model = _ScriptedModel(
        [
            _tool_call_response(
                "write_file",
                {"path": "workspace/report.txt", "content": "ok"},
            ),
            _text_response("done"),
        ]
    )
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[write_file],
        permission_manager=manager,
    )

    with EventStream() as stream:
        with execution_context(
            scope=ExecutionScope(
                session_id="user_1",
                run_id="run_1",
                permission_mode="bypass",
            )
        ):
            agent("Write the report.")
        stream.close()
        events = stream.events

    requested = next(
        event for event in events if event.name == EventType.PERMISSION_REQUESTED
    )
    assert requested.attributes["policy"] == "bypass"
    assert requested.attributes["action"] == "file.write"
    assert requested.attributes["resource"] == "workspace/report.txt"


def test_tool_permission_config_rejects_permission_mode():
    @mf.tool_config(
        permission={
            "action": "file.write",
            "mode": "bypass",
        }
    )
    def write_file(path: str) -> str:
        """Write a file."""
        return path

    library = ToolLibrary("tools", [write_file])

    with execution_context(
        scope=ExecutionScope(permission_mode="deny"),
        permission_manager=PermissionManager(default_mode="bypass"),
    ):
        with pytest.raises(ValueError, match="cannot define permission mode"):
            library([("call_1", "write_file", {"path": "workspace/report.txt"})])


def test_sync_tool_execution_rejects_ask_user_permission_mode():
    @mf.tool_config(permission={"action": "file.write"})
    def write_file(path: str) -> str:
        """Write a file."""
        return path

    library = ToolLibrary("tools", [write_file])

    with execution_context(
        scope=ExecutionScope(permission_mode="ask_user"),
        permission_manager=PermissionManager(default_mode="bypass"),
    ):
        response = library([("call_1", "write_file", {"path": "workspace/report.txt"})])

    assert response.tool_calls[0].error is not None
    assert "only supported by async tool execution" in response.tool_calls[0].error


def test_invalid_scope_permission_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown permission mode"):
        with execution_context(permission_mode="askuser"):
            pass


def test_spawn_tool_is_not_dispatched_when_permission_is_denied(monkeypatch):
    dispatched = []

    @mf.tool_config(spawn=True, permission={"action": "file.write"})
    def write_file(path: str) -> str:
        """Write a file."""
        return path

    def fake_spawn(*args, **kwargs):
        dispatched.append((args, kwargs))

    monkeypatch.setattr(tool_module.F, "spawn", fake_spawn)
    library = ToolLibrary("tools", [write_file])

    with execution_context(
        scope=ExecutionScope(permission_mode="deny"),
        permission_manager=PermissionManager(default_mode="bypass"),
    ):
        response = library([("call_1", "write_file", {"path": "workspace/report.txt"})])

    assert dispatched == []
    assert response.tool_calls[0].error is not None


def test_background_tool_is_not_started_when_permission_is_denied():
    @mf.tool_config(background=True, permission={"action": "file.write"})
    def write_file(path: str) -> str:
        """Write a file."""
        return path

    library = ToolLibrary("tools", [write_file])

    with execution_context(
        scope=ExecutionScope(permission_mode="deny"),
        permission_manager=PermissionManager(default_mode="bypass"),
    ):
        response = library([("call_1", "write_file", {"path": "workspace/report.txt"})])

    assert response.tool_calls[0].error is not None
    assert library.task_store.list() == []


def test_subagent_tool_permissions_inherit_parent_scope_mode():
    @mf.tool_config(
        permission={
            "action": "file.write",
            "resource_arg": "path",
        }
    )
    def write_file(path: str) -> str:
        """Write a file."""
        return path

    child_model = _ScriptedModel(
        [
            _tool_call_response("write_file", {"path": "workspace/child.txt"}),
            _text_response("child done"),
        ]
    )
    child_agent = Agent(name="FileWorker", model=child_model, tools=[write_file])
    parent_model = _ScriptedModel(
        [
            _tool_call_response("FileWorker", {"task": "Write child file."}),
            _text_response("parent done"),
        ]
    )
    parent_agent = Agent(
        name="MainAgent",
        model=parent_model,
        tools=[child_agent],
        permission_manager=PermissionManager(default_mode="deny"),
    )

    with EventStream() as stream:
        with execution_context(
            scope=ExecutionScope(
                session_id="user_1",
                run_id="run_parent",
                permission_mode="bypass",
            )
        ):
            parent_agent("Delegate file writing.")
        stream.close()
        events = stream.events

    requested = next(
        event for event in events if event.name == EventType.PERMISSION_REQUESTED
    )
    assert requested.attributes["policy"] == "bypass"
    assert requested.attributes["tool_name"] == "write_file"
    assert requested.attributes["caller_name"] == "FileWorker"
