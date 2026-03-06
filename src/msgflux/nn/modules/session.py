"""Agent session management for checkpoint/resume."""

import time
from typing import Any, Dict, List, Optional

import msgspec

from msgflux.data.stores.base import CheckpointStore
from msgflux.data.stores.schemas import SessionEvent, SessionStatus


class AgentSessionManager:
    """Manages checkpoint/resume logic for Agent sessions.

    Uses a CheckpointStore to persist agent session state, enabling
    resume after crash, replay of pending tool calls, and subagent
    continuity.

    The store key is composed as ``{session_id}:{agent_name}``.
    """

    def __init__(self, store: CheckpointStore, agent_name: str):
        self.store = store
        self.agent_name = agent_name

    def _store_key(self, session_id: str) -> str:
        return f"{session_id}:{self.agent_name}"

    def save_checkpoint(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        vars: Dict[str, Any],  # noqa: A002
        *,
        status: SessionStatus = SessionStatus.ACTIVE,
        pending_tool_calls: Optional[List] = None,
        error: Optional[str] = None,
    ) -> None:
        """Save a session checkpoint event."""
        store_key = self._store_key(session_id)

        event = SessionEvent(
            agent_name=self.agent_name,
            status=status.value,
            timestamp=time.time(),
            messages_snapshot=msgspec.json.encode(messages),
            vars_snapshot=msgspec.json.encode(vars),
            error=error,
            pending_tool_calls=(
                msgspec.json.encode(pending_tool_calls)
                if pending_tool_calls is not None
                else None
            ),
        )

        # Reuse CheckpointStore by wrapping SessionEvent as StepEvent-compatible
        from msgflux.data.stores.schemas import StepEvent  # noqa: PLC0415

        step_event = StepEvent(
            step_name=self.agent_name,
            status=status.value,
            timestamp=event.timestamp,
            message_snapshot=msgspec.json.encode(event),
            error=error,
        )
        self.store.save_event(store_key, step_event)

    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the last checkpoint for a session.

        Returns:
            Dict with keys: messages, vars, pending_tool_calls, status, error.
            None if no checkpoint exists.
        """
        store_key = self._store_key(session_id)
        run_state = self.store.load_run(store_key)
        if run_state is None or not run_state.events:
            return None

        last_step_event = run_state.events[-1]
        if last_step_event.message_snapshot is None:
            return None

        session_event = msgspec.json.decode(
            last_step_event.message_snapshot, type=SessionEvent
        )

        result: Dict[str, Any] = {
            "status": session_event.status,
            "error": session_event.error,
            "messages": (
                msgspec.json.decode(session_event.messages_snapshot)
                if session_event.messages_snapshot
                else []
            ),
            "vars": (
                msgspec.json.decode(session_event.vars_snapshot)
                if session_event.vars_snapshot
                else {}
            ),
            "pending_tool_calls": (
                msgspec.json.decode(session_event.pending_tool_calls)
                if session_event.pending_tool_calls
                else None
            ),
        }
        return result

    def is_completed(self, session_id: str) -> bool:
        """Check if a session is already completed."""
        checkpoint = self.load_checkpoint(session_id)
        if checkpoint is None:
            return False
        return checkpoint["status"] == SessionStatus.COMPLETED.value
