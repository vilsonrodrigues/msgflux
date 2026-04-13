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


def _notification_messages(
    messages,
    *,
    source: str | None = None,
    status: str | None = None,
):
    result = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or "<notifications>" not in content:
            continue
        if source is not None and f'source="{source}"' not in content:
            continue
        if status is not None and f'status="{status}"' not in content:
            continue
        result.append(message)
    return result


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
    assert "task_status" in add_result.tool_calls[0].result
    assert "task_wait" in add_result.tool_calls[0].result
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
    assert "task_status" in library.get_tool_names()
    assert "task_wait" in library.get_tool_names()
    assert "task_output" in library.get_tool_names()
    assert started.wait(timeout=1.0)
    assert "task_id='" in dispatch.tool_calls[0].result

    list_result = library([("call_2", "task_list", {})])
    task_id = list_result.tool_calls[0].result[0]["task_id"]

    get_result = library([("call_3", "task_status", {"task_id": task_id})])
    task_state = get_result.tool_calls[0].result
    assert task_state["status"] == "running"
    assert task_state["progress"]["stage"] == "work"
    assert task_state["progress"]["percent"] == 50.0

    release.set()
    _wait_until(
        lambda: library([("call_4", "task_status", {"task_id": task_id})]).tool_calls[0]
        .result["status"]
        == "completed"
    )

    output_result = library([("call_5", "task_output", {"task_id": task_id})])
    assert output_result.tool_calls[0].result == 42


def test_task_wait_returns_final_output():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    dispatch = library([("call_1", "long_job", {"value": 21})])
    assert "task_wait" in library.get_tool_names()
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]
    assert f"task_wait(task_id='{task_id}')" in dispatch.tool_calls[0].result

    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        wait_result = library([("call_3", "task_wait", {"task_id": task_id, "timeout": 1.0})])
    finally:
        timer.cancel()

    assert wait_result.tool_calls[0].result == 42


def test_task_wait_returns_timeout_payload_with_progress():
    release = threading.Event()

    @mf.tool_config(background=True, inject_task=True)
    def long_job(value: int, task) -> int:
        """Run a long job in the background."""
        task.update_progress(stage="work", message="Halfway", current=1, total=2)
        release.wait(timeout=2.0)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    library([("call_1", "long_job", {"value": 21})])
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    wait_result = library([("call_3", "task_wait", {"task_id": task_id, "timeout": 0.05})])
    payload = wait_result.tool_calls[0].result

    assert payload["task_id"] == task_id
    assert payload["status"] == "timeout"
    assert payload["task_status"] == "running"
    assert payload["progress"]["stage"] == "work"
    assert payload["progress"]["percent"] == 50.0

    release.set()
    _wait_until(
        lambda: library([("call_4", "task_status", {"task_id": task_id})]).tool_calls[0]
        .result["status"]
        == "completed"
    )


def test_task_wait_returns_failed_payload():
    @mf.tool_config(background=True)
    def failing_job() -> int:
        """Always fail."""
        raise RuntimeError("boom")

    library = ToolLibrary(name="lib", tools=[failing_job])

    library([("call_1", "failing_job", {})])
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    wait_result = library([("call_3", "task_wait", {"task_id": task_id, "timeout": 1.0})])
    payload = wait_result.tool_calls[0].result

    assert payload["task_id"] == task_id
    assert payload["status"] == "failed"
    assert "boom" in payload["error"]


def test_task_wait_falls_back_to_task_store_polling_without_future():
    @mf.tool_config(background=True)
    def placeholder() -> None:
        """Enable task runtime tools for the library."""
        return None

    library = ToolLibrary(name="lib", tools=[placeholder])
    task = library.task_store.create(tool_name="external_job")

    def complete_task():
        time.sleep(0.1)
        library.task_store.complete(task.task_id, 99)

    timer = threading.Thread(target=complete_task)
    timer.start()
    try:
        wait_result = library([("call_1", "task_wait", {"task_id": task.task_id, "timeout": 1.0})])
    finally:
        timer.join(timeout=1.0)

    assert wait_result.tool_calls[0].result == 99


def test_agent_injects_pending_task_notifications_as_system_note_messages():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    agent = Agent(name="Assistant", model=_mock_model(), tools=[long_job])

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = agent.tool_library([("call_2", "task_list", {})]).tool_calls[0].result[0][
        "task_id"
    ]

    release.set()
    _wait_until(
        lambda: agent.tool_library(
            [("call_3", "task_status", {"task_id": task_id})]
        ).tool_calls[0].result["status"]
        == "completed"
    )

    _wait_until(
        lambda: bool(
            _notification_messages(
                agent.inspect_model_execution_params("Continue.")["messages"],
                source="task",
                status="completed",
            )
        )
    )
    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )

    assert len(notification_messages) == 1
    content = notification_messages[0]["content"]
    assert "<system_note>" in content
    assert "<notifications>" in content
    assert f'ref="{task_id}"' in content
    assert "tool=long_job" in content
    assert f"task_output(task_id='{task_id}')" in content


def test_inspect_model_execution_params_does_not_consume_notifications():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    model = _mock_model()
    agent = Agent(name="Assistant", model=model, tools=[long_job])

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = agent.tool_library([("call_2", "task_list", {})]).tool_calls[0].result[0][
        "task_id"
    ]

    release.set()
    _wait_until(
        lambda: agent.tool_library(
            [("call_3", "task_status", {"task_id": task_id})]
        ).tool_calls[0].result["status"]
        == "completed"
    )

    _wait_until(
        lambda: bool(
            _notification_messages(
                agent.inspect_model_execution_params("Continue.")["messages"],
                source="task",
                status="completed",
            )
        )
    )
    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1

    params = agent.inspect_model_execution_params("Continue again.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1

    messages = ChatMessages()
    agent("Continue now.", messages=messages)

    model_messages = model.call_args.kwargs["messages"]
    notification_messages = _notification_messages(
        model_messages,
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1

    history_messages = messages.to_chatml()
    persisted_notifications = _notification_messages(
        history_messages,
        source="task",
        status="completed",
    )
    assert len(persisted_notifications) == 1
    notification_index = history_messages.index(persisted_notifications[0])
    user_index = next(
        index
        for index, message in enumerate(history_messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and "Continue now." in message["content"]
    )
    assert notification_index < user_index

    params = agent.inspect_model_execution_params("Continue once more.")
    notification_messages = _notification_messages(params["messages"])
    assert notification_messages == []


def test_task_progress_notifications_are_persisted():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True, inject_task=True)
    def long_job(value: int, task) -> int:
        """Emit progress updates while running in the background."""
        task.notify(
            source="task_progress",
            status="update",
            hint="Wait for the final completion notification before consuming output.",
            metadata={"tool_stage": "prepare"},
            dedupe_key=f"progress:{task.task_id}",
        )
        started.set()
        release.wait(timeout=2.0)
        return value * 2

    model = _mock_model()
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[long_job],
    )

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = agent.tool_library([("call_2", "task_list", {})]).tool_calls[0].result[0][
        "task_id"
    ]
    assert started.wait(timeout=1.0)

    messages = ChatMessages()
    agent("Continue.", messages=messages)

    model_messages = model.call_args.kwargs["messages"]
    progress_notifications = _notification_messages(
        model_messages,
        source="task_progress",
        status="update",
    )
    assert len(progress_notifications) == 1
    assert f'ref="{task_id}"' in progress_notifications[0]["content"]
    assert "tool_stage=prepare" in progress_notifications[0]["content"]

    persisted_notifications = _notification_messages(
        messages.to_chatml(),
        source="task_progress",
        status="update",
    )
    assert len(persisted_notifications) == 1

    release.set()
    _wait_until(
        lambda: agent.tool_library(
            [("call_3", "task_status", {"task_id": task_id})]
        ).tool_calls[0].result["status"]
        == "completed"
    )


def test_nested_agent_uses_inherited_inbox_from_execution_context():
    parent_inbox = mf.AgentInbox()
    child = Agent(name="child", model=_mock_model())

    with execution_context(agent_inbox=parent_inbox):
        effective_inbox = child._get_effective_agent_inbox()

    assert effective_inbox is parent_inbox


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
        lambda: library([("call_3", "task_status", {"task_id": task_id})]).tool_calls[0]
        .result["status"]
        == "completed"
    )

    task_state = library([("call_4", "task_status", {"task_id": task_id})]).tool_calls[0].result
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
