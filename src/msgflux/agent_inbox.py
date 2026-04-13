from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import uuid4
from xml.sax.saxutils import escape

from msgflux.logger import logger


# --- Module Utilities ---

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Notification Model ---

@dataclass
class AgentNotification:
    notification_id: str
    source: str
    ref: Optional[str] = None
    status: Optional[str] = None
    hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: Optional[str] = None
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentInbox:
    """Small inbox for notifications delivered to an agent runtime."""

    def __init__(self, *, verbose: bool = False, owner: Optional[str] = None):
        self._lock = Lock()
        self._notifications: List[AgentNotification] = []
        self.verbose = verbose
        self.owner = owner

    # --- Configuration ---

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = bool(verbose)

    # --- Queue Operations ---

    def publish(
        self,
        notification: AgentNotification | Mapping[str, Any],
    ) -> AgentNotification:
        normalized = self._normalize(notification)
        with self._lock:
            if normalized.dedupe_key:
                for index, existing in enumerate(self._notifications):
                    if existing.dedupe_key == normalized.dedupe_key:
                        self._notifications[index] = normalized
                        if self.verbose:
                            logger.debug(
                                "AgentInbox[%s] replaced notification source=%s ref=%s status=%s",
                                self.owner or "unknown",
                                normalized.source,
                                normalized.ref,
                                normalized.status,
                            )
                        return deepcopy(normalized)
            self._notifications.append(normalized)
        if self.verbose:
            logger.debug(
                "AgentInbox[%s] published notification source=%s ref=%s status=%s",
                self.owner or "unknown",
                normalized.source,
                normalized.ref,
                normalized.status,
            )
        return deepcopy(normalized)

    def publish_many(
        self,
        notifications: Iterable[AgentNotification | Mapping[str, Any]],
    ) -> List[AgentNotification]:
        published: List[AgentNotification] = []
        for notification in notifications:
            published.append(self.publish(notification))
        return published

    def peek(self) -> List[AgentNotification]:
        with self._lock:
            return deepcopy(self._notifications)

    def drain(self) -> List[AgentNotification]:
        with self._lock:
            notifications = deepcopy(self._notifications)
            self._notifications.clear()
        if self.verbose and notifications:
            logger.debug(
                "AgentInbox[%s] drained %d notification(s)",
                self.owner or "unknown",
                len(notifications),
            )
        return notifications

    def ack(self, notification_ids: Iterable[str]) -> None:
        ids = {notification_id for notification_id in notification_ids}
        if not ids:
            return
        with self._lock:
            self._notifications = [
                notification
                for notification in self._notifications
                if notification.notification_id not in ids
            ]

    # --- Rendering ---

    def render(
        self,
        notifications: Iterable[AgentNotification | Mapping[str, Any]],
    ) -> Optional[Dict[str, str]]:
        normalized = [self._normalize(notification) for notification in notifications]
        if not normalized:
            return None

        lines = ["<system_note>", "<notifications>"]
        for notification in normalized:
            attrs = [f'source="{self._escape_attr(notification.source)}"']
            if notification.ref:
                attrs.append(f'ref="{self._escape_attr(notification.ref)}"')
            if notification.status:
                attrs.append(f'status="{self._escape_attr(notification.status)}"')

            body_lines: List[str] = []
            for key, value in sorted(notification.metadata.items()):
                body_lines.append(
                    f"{self._escape_text(key)}={self._escape_text(self._stringify(value))}"
                )
            if notification.hint:
                body_lines.append(f"hint={self._escape_text(notification.hint)}")

            attrs_repr = " ".join(attrs)
            if body_lines:
                lines.append(f"<notification {attrs_repr}>")
                lines.extend(body_lines)
                lines.append("</notification>")
            else:
                lines.append(f"<notification {attrs_repr} />")
        lines.extend(["</notifications>", "</system_note>"])
        return {"role": "user", "content": "\n".join(lines)}

    # --- Normalization Helpers ---

    def _normalize(
        self,
        notification: AgentNotification | Mapping[str, Any],
    ) -> AgentNotification:
        if isinstance(notification, AgentNotification):
            return deepcopy(notification)

        payload = dict(notification)
        source = payload.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("`AgentNotification.source` must be a non-empty string.")

        notification_id = payload.get("notification_id")
        if not isinstance(notification_id, str) or not notification_id:
            notification_id = uuid4().hex[:8]

        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "`AgentNotification.metadata` must be a mapping, "
                f"given `{type(metadata)}`"
            )

        created_at = payload.get("created_at")
        if not isinstance(created_at, str) or not created_at:
            created_at = _utc_now()

        return AgentNotification(
            notification_id=notification_id,
            source=source,
            ref=payload.get("ref"),
            status=payload.get("status"),
            hint=payload.get("hint"),
            metadata=deepcopy(dict(metadata)),
            dedupe_key=payload.get("dedupe_key"),
            created_at=created_at,
        )

    # --- Escaping Helpers ---

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _escape_text(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})

    @staticmethod
    def _escape_attr(value: Any) -> str:
        return escape(str(value), {'"': "&quot;"})
