"""Context variables for durable execution.

The run_id propagates via contextvars.ContextVar, naturally reaching
subagents called as tools without modifying ToolLibrary.
"""

import contextvars
from typing import Optional

_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_run_id", default=None
)

_current_while_scope: contextvars.ContextVar[str] = contextvars.ContextVar(
    "msgflux_while_scope", default=""
)


def get_run_id() -> Optional[str]:
    """Get the current run_id from context."""
    return _current_run_id.get()


def set_run_id(run_id: Optional[str]) -> contextvars.Token:
    """Set the run_id in the current context.

    Returns:
        A token that can be used to reset the context variable.
    """
    return _current_run_id.set(run_id)


_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_session_id", default=None
)


def get_session_id() -> Optional[str]:
    """Get the current session_id from context."""
    return _current_session_id.get()


def set_session_id(session_id: Optional[str]) -> contextvars.Token:
    """Set the session_id in the current context.

    Returns:
        A token that can be used to reset the context variable.
    """
    return _current_session_id.set(session_id)
