from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Mapping

import msgspec

from msgflux.context import get_execution_scope
from msgflux.runtime.events import emit_todo_updated

TodoStatus = Literal["pending", "in_progress", "completed"]


class TodoItem(msgspec.Struct):
    content: str
    active_form: str
    status: TodoStatus


class TodoManager:
    """In-memory todo state scoped by execution session and namespace."""

    def __init__(self) -> None:
        self._todos: dict[tuple[str, str], list[TodoItem]] = {}

    def get(
        self,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> list[TodoItem]:
        key = self._scope_key(session_id=session_id, namespace=namespace)
        return list(self._todos.get(key, []))

    def update(
        self,
        todos: list[TodoItem | Mapping[str, Any]],
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        key = self._scope_key(session_id=session_id, namespace=namespace)
        old_todos = self._todos.get(key, [])
        new_todos = self._normalize_todos(todos)
        self._todos[key] = new_todos

        payload = {
            "session_id": key[0],
            "namespace": key[1],
            "old_todos": self.to_builtins(old_todos),
            "new_todos": self.to_builtins(new_todos),
            "stats": self.stats(new_todos),
            "changed": self.to_builtins(old_todos) != self.to_builtins(new_todos),
        }
        emit_todo_updated(payload)
        return payload

    def clear(
        self,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        return self.update([], session_id=session_id, namespace=namespace)

    def _scope_key(
        self,
        *,
        session_id: str | None = None,
        namespace: str | None = None,
    ) -> tuple[str, str]:
        scope = get_execution_scope()
        return (
            session_id or scope.session_id,
            namespace or scope.namespace,
        )

    def _normalize_todos(
        self,
        todos: list[TodoItem | Mapping[str, Any]],
    ) -> list[TodoItem]:
        if not isinstance(todos, list):
            raise TypeError("`todos` must be a list.")
        normalized = []
        for todo in todos:
            if isinstance(todo, TodoItem):
                item = todo
            elif isinstance(todo, Mapping):
                item = TodoItem(
                    content=str(todo.get("content", "")).strip(),
                    active_form=str(todo.get("active_form", "")).strip(),
                    status=todo.get("status"),
                )
            else:
                raise TypeError("Each todo must be a TodoItem or mapping.")
            self._validate_todo(item)
            normalized.append(item)

        in_progress_count = sum(
            1 for todo in normalized if todo.status == "in_progress"
        )
        if in_progress_count > 1:
            raise ValueError("Only one todo can be `in_progress` at a time.")
        return normalized

    def _validate_todo(self, todo: TodoItem) -> None:
        if not todo.content.strip():
            raise ValueError("Todo `content` cannot be empty.")
        if not todo.active_form.strip():
            raise ValueError("Todo `active_form` cannot be empty.")
        if todo.status not in {"pending", "in_progress", "completed"}:
            raise ValueError(f"Invalid todo status: {todo.status}")

    @staticmethod
    def to_builtins(todos: list[TodoItem]) -> list[dict[str, Any]]:
        return [msgspec.to_builtins(todo) for todo in todos]

    @staticmethod
    def stats(todos: list[TodoItem]) -> dict[str, int]:
        counts = Counter(todo.status for todo in todos)
        return {
            "pending": counts["pending"],
            "in_progress": counts["in_progress"],
            "completed": counts["completed"],
            "total": len(todos),
        }
