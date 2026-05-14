import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType, TodoItem, TodoManager
from msgflux.tools.builtin import TodoWrite


def _todos():
    return [
        TodoItem(
            content="Implement todo runtime",
            active_form="Implementing todo runtime",
            status="in_progress",
        ),
        TodoItem(
            content="Write todo runtime tests",
            active_form="Writing todo runtime tests",
            status="pending",
        ),
    ]


def test_todo_manager_updates_session_scoped_state_and_emits_event():
    manager = TodoManager()

    with execution_context(session_id="session_a", namespace="agent"):
        with EventStream() as stream:
            result = manager.update(_todos())
            stream.close()
            events = stream.events

        assert manager.get() == _todos()

    assert result["old_todos"] == []
    assert result["new_todos"] == [
        {
            "content": "Implement todo runtime",
            "active_form": "Implementing todo runtime",
            "status": "in_progress",
        },
        {
            "content": "Write todo runtime tests",
            "active_form": "Writing todo runtime tests",
            "status": "pending",
        },
    ]
    assert result["stats"] == {
        "pending": 1,
        "in_progress": 1,
        "completed": 0,
        "total": 2,
    }
    assert [event.name for event in events] == [EventType.TODO_UPDATED]
    assert events[0].attributes["session_id"] == "session_a"
    assert events[0].attributes["namespace"] == "agent"
    assert events[0].attributes["stats"]["in_progress"] == 1


def test_todo_manager_keeps_sessions_isolated():
    manager = TodoManager()

    manager.update(_todos(), session_id="session_a", namespace="agent")
    manager.update([], session_id="session_b", namespace="agent")

    assert len(manager.get(session_id="session_a", namespace="agent")) == 2
    assert manager.get(session_id="session_b", namespace="agent") == []


def test_todo_manager_rejects_multiple_in_progress_items():
    manager = TodoManager()

    with pytest.raises(ValueError, match="Only one todo"):
        manager.update(
            [
                TodoItem(
                    content="First task",
                    active_form="Working first task",
                    status="in_progress",
                ),
                TodoItem(
                    content="Second task",
                    active_form="Working second task",
                    status="in_progress",
                ),
            ]
        )


def test_todo_write_tool_restores_struct_items_and_updates_manager():
    manager = TodoManager()
    library = ToolLibrary("agent", [TodoWrite(manager)])

    responses = library(
        [
            (
                "call_1",
                "todo_write",
                {
                    "todos": [
                        {
                            "content": "Implement todo runtime",
                            "active_form": "Implementing todo runtime",
                            "status": "in_progress",
                        }
                    ]
                },
            )
        ]
    )

    response = responses.get_by_name("todo_write")
    assert response is not None
    assert response.error is None
    assert response.result["stats"]["in_progress"] == 1
    assert response.result["message"].startswith("Todo list updated.")
    assert manager.get() == [
        TodoItem(
            content="Implement todo runtime",
            active_form="Implementing todo runtime",
            status="in_progress",
        )
    ]
    assert library.library["todo_write"].display_name == "Todo"
    assert library.library["todo_write"].description == TodoWrite.description


def test_todo_write_tool_schema_uses_msgspec_struct_items():
    tool = ToolLibrary("agent", [TodoWrite()]).library["todo_write"]
    schema = tool.get_json_schema()

    todos_schema = schema["function"]["parameters"]["properties"]["todos"]
    item_schema = todos_schema["items"]

    assert todos_schema["type"] == "array"
    assert item_schema["type"] == "object"
    assert item_schema["additionalProperties"] is False
    assert set(item_schema["properties"]["status"]["enum"]) == {
        "pending",
        "in_progress",
        "completed",
    }
