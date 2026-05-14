from typing import Any

from msgflux.runtime.todos import TodoItem, TodoManager


class TodoWrite:
    """Update the todo list for the current session."""

    name = "todo_write"
    display_name = "Todo"
    description = (
        "Update the todo list for the current session. Use it proactively to "
        "track progress on multi-step tasks. Keep at most one todo in_progress."
    )
    read_only = False
    concurrency_safe = True

    def __init__(self, manager: TodoManager | None = None):
        self.manager = manager or TodoManager()

    def __call__(self, todos: list[TodoItem]) -> dict[str, Any]:
        """Update the todo list for the current session.

        Args:
            todos: Complete todo list. Each item requires content, active_form,
                and status.
        """
        result = self.manager.update(todos)
        return {
            "old_todos": result["old_todos"],
            "new_todos": result["new_todos"],
            "stats": result["stats"],
            "changed": result["changed"],
            "message": self._format_tool_result(result["stats"]),
        }

    async def acall(self, todos: list[TodoItem]) -> dict[str, Any]:
        return self(todos)

    def _format_tool_result(self, stats: dict[str, int]) -> str:
        return (
            "Todo list updated. Continue using it to track progress. "
            f"Current status: {stats['in_progress']} in progress, "
            f"{stats['pending']} pending, {stats['completed']} completed."
        )
