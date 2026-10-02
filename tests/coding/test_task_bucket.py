"""Tests for the coding host's consolidated background task tool."""

import msgflux as mf
from msgflux.nn import ToolLibrary
from msgflux.tools.builtin.task_bucket import TaskTool


def test_task_bucket_captures_auto_installed_background_controls():
    @mf.tool_config(background=True)
    def background_job() -> str:
        """Return a small background result."""
        return "finished"

    task_tool = TaskTool()
    library = ToolLibrary("coding", [task_tool, background_job])

    assert set(task_tool.tools) == {
        "task_status",
        "task_list",
        "task_output",
        "task_wait",
        "task_interrupt",
    }
    assert library.get_tool_names() == ["task", "background_job"]

    listed = library(
        [("call_1", "task", {"request": {"action": "list", "status": None}})]
    )
    assert listed.tool_calls[0].result == []

    library.remove("background_job")
    assert task_tool.tools == {}
    assert library.get_tool_names() == ["task"]


def test_task_bucket_schema_is_a_tagged_union_of_available_actions():
    @mf.tool_config(background=True)
    def background_job() -> str:
        """Return a small background result."""
        return "finished"

    library = ToolLibrary("coding", [TaskTool(), background_job])
    schema = library.get_tool_definition("task").input_schema
    actions = {
        tuple(branch["properties"]["action"]["enum"])
        for branch in schema["properties"]["request"]["anyOf"]
    }
    for branch in schema["properties"]["request"]["anyOf"]:
        assert set(branch["properties"]) == set(branch["required"])
        assert all("default" not in prop for prop in branch["properties"].values())

    assert actions == {
        ("status",),
        ("list",),
        ("output",),
        ("wait",),
        ("interrupt",),
    }
    description = library.get_tool_definition("task").description
    assert "activity" not in description
    assert "message" not in description


def test_task_bucket_wait_returns_real_background_result():
    @mf.tool_config(background=True)
    def background_job() -> str:
        """Return a small background result."""
        return "finished"

    library = ToolLibrary("coding", [TaskTool(), background_job])
    launched = library([("launch", "background_job", {})])
    task_id = launched.tool_calls[0].result.split("task_id='")[1].split("'")[0]
    waited = library(
        [
            (
                "wait",
                "task",
                {"request": {"action": "wait", "task_id": task_id, "timeout": 5.0}},
            )
        ]
    )
    assert waited.tool_calls[0].result == "finished"


def test_task_bucket_advertises_agent_capabilities_and_removes_them(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from msgflux.nn import Agent

    worker = Agent(name="worker", model="openai/gpt-5")
    worker.tool_config = {"background": True}
    bucket = TaskTool()
    library = ToolLibrary("coding", [bucket, worker])
    assert {"task_message", "task_activity"} <= set(bucket.tools)
    schema = library.get_tool_definition("task").input_schema
    assert len(schema["properties"]["request"]["anyOf"]) == 7
    library.remove("worker")
    assert not bucket.tools
