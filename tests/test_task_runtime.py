"""Focused tests for background tasks, task progress, notifications, and
library-aware tools."""

import threading
import time
from unittest.mock import MagicMock, Mock

import msgflux as mf
import pytest
from msgflux.chat_messages import ChatMessages
from msgflux.context import execution_context
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Timed out waiting for condition.")


def _mock_model(text: str = "ok") -> MagicMock:
    model = MagicMock()
    model.model_type = "chat_completion"
    resp = Mock(spec=ModelResponse)
    resp.response_type = "text_generation"
    resp.consume.return_value = text
    resp.data = text
    resp.reasoning = None
    resp.metadata = {}
    model.return_value = resp
    return model


def test_background_tool_schema_excludes_task_handle():
    @mf.tool_config(background=True, inject_task=True)
    def background_tool(query: str, task) -> str:
        """Run a query in the background."""
        return query

    library = ToolLibrary(name="lib", tools=[background_tool])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "background_tool"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "query" in props
    assert "task" not in props


def test_inject_library_schema_excludes_tool_library_handle():
    @mf.tool_config(inject_library=True)
    def register_tool(tool_library, name: str) -> str:
        """Register a tool by name."""
        return name

    library = ToolLibrary(name="lib", tools=[register_tool])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "register_tool"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "name" in props
    assert "tool_library" not in props


def test_inject_library_can_add_and_remove_tools():
    def multiply(x: int) -> int:
        """Multiply a number by two."""
        return x * 2

    @mf.tool_config(inject_library=True)
    def add_multiplier(tool_library) -> list[str]:
        """Register the multiply tool."""
        tool_library.add(multiply)
        return tool_library.list_tools()

    @mf.tool_config(inject_library=True)
    def remove_tool(tool_library, name: str) -> list[str]:
        """Remove a tool by name."""
        tool_library.remove(name)
        return tool_library.list_tools()

    library = ToolLibrary(name="lib", tools=[add_multiplier, remove_tool])

    add_result = library([("call_1", "add_multiplier", {})])
    assert "multiply" in add_result.tool_calls[0].result

    run_result = library([("call_2", "multiply", {"x": 4})])
    assert run_result.tool_calls[0].result == 8

    remove_result = library([("call_3", "remove_tool", {"name": "multiply"})])
    assert "multiply" not in remove_result.tool_calls[0].result
    assert "multiply" not in library.get_tool_names()


def test_inject_library_can_add_background_tool_with_task_tools():
    @mf.tool_config(background=True, inject_task=True)
    def background_multiplier(value: int, task) -> int:
        """Multiply a number by two in the background."""
        task.update_progress(stage="work", message="Running", current=1, total=1)
        return value * 2

    @mf.tool_config(inject_library=True)
    def add_background_multiplier(tool_library) -> list[str]:
        """Register a background tool."""
        tool_library.add(background_multiplier)
        return tool_library.list_tools()

    library = ToolLibrary(name="lib", tools=[add_background_multiplier])

    add_result = library([("call_1", "add_background_multiplier", {})])
    assert "background_multiplier" in add_result.tool_calls[0].result
    assert "task_get" in add_result.tool_calls[0].result
    assert "task_output" in add_result.tool_calls[0].result

    dispatch = library([("call_2", "background_multiplier", {"value": 4})])
    assert "task_id='" in dispatch.tool_calls[0].result

    _wait_until(
        lambda: library([("call_3", "task_list", {})]).tool_calls[0].result[0]["status"]
        == "completed"
    )

    task_id = library([("call_4", "task_list", {})]).tool_calls[0].result[0]["task_id"]
    output_result = library([("call_5", "task_output", {"task_id": task_id})])
    assert output_result.tool_calls[0].result == 8


def test_background_task_reports_progress_and_output():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True, inject_task=True)
    def long_job(value: int, task) -> int:
        """Run a long job in the background."""
        task.set_running(stage="prepare", message="Preparing")
        task.update_progress(stage="work", message="Halfway", current=1, total=2)
        started.set()
        release.wait(timeout=2.0)
        task.update_progress(stage="work", message="Finishing", current=2, total=2)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    dispatch = library([("call_1", "long_job", {"value": 21})])
    assert "task_get" in library.get_tool_names()
    assert "task_output" in library.get_tool_names()
    assert started.wait(timeout=1.0)
    assert "task_id='" in dispatch.tool_calls[0].result

    list_result = library([("call_2", "task_list", {})])
    task_id = list_result.tool_calls[0].result[0]["task_id"]

    get_result = library([("call_3", "task_get", {"task_id": task_id})])
    task_state = get_result.tool_calls[0].result
    assert task_state["status"] == "running"
    assert task_state["progress"]["stage"] == "work"
    assert task_state["progress"]["percent"] == 50.0

    release.set()
    _wait_until(
        lambda: library([("call_4", "task_get", {"task_id": task_id})]).tool_calls[0]
        .result["status"]
        == "completed"
    )

    output_result = library([("call_5", "task_output", {"task_id": task_id})])
    assert output_result.tool_calls[0].result == 42


def test_agent_injects_pending_task_notifications_as_system_note_messages():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    agent = Agent(
        name="Assistant",
        model=_mock_model(),
        tools=[long_job],
    )

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = agent.tool_library([("call_2", "task_list", {})]).tool_calls[0].result[0][
        "task_id"
    ]

    release.set()
    _wait_until(
        lambda: agent.tool_library(
            [("call_3", "task_get", {"task_id": task_id})]
        ).tool_calls[0].result["status"]
        == "completed"
    )

    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]

    assert len(notification_messages) == 1
    assert "<system_note>" in notification_messages[0]["content"]
    assert f"task_output(task_id='{task_id}')" in notification_messages[0]["content"]

    params = agent._prepare_inputs("Continue again.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]

    assert len(notification_messages) == 1

    params = agent._prepare_inputs("Continue once more.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]

    assert notification_messages == []


def test_inspect_model_execution_params_does_not_consume_notifications():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    agent = Agent(
        name="Assistant",
        model=_mock_model(),
        tools=[long_job],
    )

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = agent.tool_library([("call_2", "task_list", {})]).tool_calls[0].result[0][
        "task_id"
    ]

    release.set()
    _wait_until(
        lambda: agent.tool_library(
            [("call_3", "task_get", {"task_id": task_id})]
        ).tool_calls[0].result["status"]
        == "completed"
    )

    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]
    assert len(notification_messages) == 1

    params = agent.inspect_model_execution_params("Continue again.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]
    assert len(notification_messages) == 1

    params = agent._prepare_inputs("Continue now.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]
    assert len(notification_messages) == 1

    params = agent._prepare_inputs("Continue once more.")
    notification_messages = [
        message
        for message in params["messages"]
        if isinstance(message.get("content"), str)
        and "<task_notification>" in message["content"]
    ]
    assert notification_messages == []


def test_background_agent_inherits_context_and_checkpoint_run_id():
    store = InMemoryCheckpointStore()
    worker = Agent(name="worker", model=_mock_model("worker-done"))
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])

    with execution_context(
        session_id="user_42",
        namespace="root_agent",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library([("call_1", "worker", {"task": "Solve this"})])

    assert "task_id='" in dispatch.tool_calls[0].result
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    _wait_until(
        lambda: library([("call_3", "task_get", {"task_id": task_id})]).tool_calls[0]
        .result["status"]
        == "completed"
    )

    task_state = library([("call_4", "task_get", {"task_id": task_id})]).tool_calls[0].result
    assert task_state["metadata"]["session_id"] == "user_42"
    assert task_state["metadata"]["parent_run_id"] == "run_root"
    assert task_state["metadata"]["root_run_id"] == "run_root"
    assert task_state["metadata"]["checkpoint_session_id"] == "user_42"
    assert task_state["metadata"]["checkpoint_run_id"] == task_id

    state = store.load_state("worker", "user_42", task_id)
    assert state is not None
    assert state["status"] == "completed"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    history = restored.to_chatml()
    assert any(
        message["role"] == "user" and "Solve this" in str(message["content"])
        for message in history
    )
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "worker-done"
