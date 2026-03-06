"""Schemas for checkpoint store events and run state."""

import enum
from typing import List, Optional

import msgspec


class StepStatus(str, enum.Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class StepEvent(msgspec.Struct):
    """A checkpoint event for a single pipeline step."""

    step_name: str
    status: str
    timestamp: float
    message_snapshot: Optional[bytes] = None
    error: Optional[str] = None
    retry_count: int = 0


class RunState(msgspec.Struct):
    """Aggregated state for a pipeline run."""

    run_id: str
    expression: str
    events: List[StepEvent]
    created_at: float
    updated_at: float


# --- Agent Session Schemas ---


class SessionStatus(str, enum.Enum):
    """Status of an agent session."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionEvent(msgspec.Struct):
    """A checkpoint event for an agent session."""

    agent_name: str
    status: str
    timestamp: float
    messages_snapshot: Optional[bytes] = None
    vars_snapshot: Optional[bytes] = None
    error: Optional[str] = None
    pending_tool_calls: Optional[bytes] = None


class SessionState(msgspec.Struct):
    """Aggregated state for an agent session."""

    session_id: str
    agent_name: str
    events: List[SessionEvent]
    created_at: float
    updated_at: float
