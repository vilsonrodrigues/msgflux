from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskProgress:
    stage: Optional[str] = None
    message: Optional[str] = None
    current: Optional[int] = None
    total: Optional[int] = None
    percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    tool_name: str
    status: str
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskStore:
    """Thread-safe in-memory store for background task state."""

    def __init__(self):
        self._lock = RLock()
        self._tasks: Dict[str, TaskRecord] = {}

    def create(
        self,
        tool_name: str,
        *,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskRecord:
        now = _utc_now()
        task = TaskRecord(
            task_id=task_id or uuid4().hex[:8],
            tool_name=tool_name,
            status="queued",
            created_at=now,
            updated_at=now,
            metadata=deepcopy(metadata or {}),
        )
        with self._lock:
            self._tasks[task.task_id] = task
        return self.get(task.task_id)  # type: ignore[return-value]

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task is not None else None

    def list(self, *, status: Optional[str] = None) -> List[TaskRecord]:
        with self._lock:
            tasks = list(self._tasks.values())
            if status is not None:
                tasks = [task for task in tasks if task.status == status]
            return deepcopy(tasks)

    def set_running(
        self,
        task_id: str,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "running"
            task.updated_at = _utc_now()
            if stage is not None:
                task.progress.stage = stage
            if message is not None:
                task.progress.message = message
            return deepcopy(task)

    def update_progress(
        self,
        task_id: str,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        percent: Optional[float] = None,
    ) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status == "queued":
                task.status = "running"
            task.updated_at = _utc_now()
            if stage is not None:
                task.progress.stage = stage
            if message is not None:
                task.progress.message = message
            if current is not None:
                task.progress.current = current
            if total is not None:
                task.progress.total = total
            if percent is None and task.progress.current is not None and task.progress.total:
                percent = (task.progress.current / task.progress.total) * 100
            if percent is not None:
                task.progress.percent = round(percent, 2)
            return deepcopy(task)

    def complete(self, task_id: str, result: Any) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = _utc_now()
            task.status = "completed"
            task.updated_at = now
            task.completed_at = now
            task.result = result
            task.error = None
            if task.progress.percent is None and task.progress.total:
                task.progress.percent = 100.0
            return deepcopy(task)

    def fail(self, task_id: str, error: Any) -> Optional[TaskRecord]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = _utc_now()
            task.status = "failed"
            task.updated_at = now
            task.completed_at = now
            task.error = str(error)
            return deepcopy(task)

class TaskHandle:
    """Small mutable handle injected into background tools."""

    def __init__(self, task_id: str, store: TaskStore):
        self.task_id = task_id
        self._store = store

    def set_running(
        self,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        return self._store.set_running(task_id=self.task_id, stage=stage, message=message)

    def update_progress(
        self,
        *,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        current: Optional[int] = None,
        total: Optional[int] = None,
        percent: Optional[float] = None,
    ) -> Optional[TaskRecord]:
        return self._store.update_progress(
            task_id=self.task_id,
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )

    def complete(self, result: Any) -> Optional[TaskRecord]:
        return self._store.complete(self.task_id, result)

    def fail(self, error: Any) -> Optional[TaskRecord]:
        return self._store.fail(self.task_id, error)
