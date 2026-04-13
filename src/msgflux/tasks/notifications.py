from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Notification:
    notification_id: str
    task_id: str
    tool_name: str
    kind: str
    status: str
    message: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_message(self) -> Dict[str, str]:
        return {
            "role": "user",
            "content": (
                "<system_note>\n"
                "<task_notification>\n"
                f"{self.message}\n"
                "</task_notification>\n"
                "</system_note>"
            ),
        }


class NotificationBus:
    """Small in-memory queue for passive task notifications."""

    def __init__(self):
        self._lock = Lock()
        self._notifications: List[Notification] = []

    def publish_task_completed(self, task_id: str, tool_name: str) -> Notification:
        notification = Notification(
            notification_id=uuid4().hex[:8],
            task_id=task_id,
            tool_name=tool_name,
            kind="completed",
            status="completed",
            message=(
                f"Background task '{task_id}' from tool '{tool_name}' completed. "
                f"Call task_output(task_id='{task_id}') to inspect the final result."
            ),
            created_at=_utc_now(),
        )
        with self._lock:
            self._notifications.append(notification)
        return notification

    def publish_task_failed(self, task_id: str, tool_name: str) -> Notification:
        notification = Notification(
            notification_id=uuid4().hex[:8],
            task_id=task_id,
            tool_name=tool_name,
            kind="failed",
            status="failed",
            message=(
                f"Background task '{task_id}' from tool '{tool_name}' failed. "
                f"Call task_get(task_id='{task_id}') to inspect the error."
            ),
            created_at=_utc_now(),
        )
        with self._lock:
            self._notifications.append(notification)
        return notification

    def list(self) -> List[Notification]:
        with self._lock:
            return deepcopy(self._notifications)

    def drain(self) -> List[Notification]:
        with self._lock:
            notifications = deepcopy(self._notifications)
            self._notifications.clear()
        return notifications

    def drain_messages(self) -> List[Dict[str, str]]:
        return [notification.to_message() for notification in self.drain()]
