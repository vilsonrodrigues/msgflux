"""Runtime context — session and namespace propagation via ContextVar.

This module owns the canonical ``session_id`` and ``namespace``
context variables used throughout msgflux.  All components that need
session identity (ChatMessages, CheckpointStore, Inline, telemetry)
consume these primitives rather than defining their own.

Usage::

    from msgflux.context import session_context, get_session_context

    with session_context(session_id="user_42", namespace="pipeline"):
        # Everything created here inherits the session.
        chat = ChatMessages()          # chat.session_id == "user_42"
        inline(msg, store=store)       # checkpoint keyed by "user_42"
"""

import contextvars
from contextlib import contextmanager
from typing import Mapping, Optional
from uuid import uuid4

_CURRENT_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_session_id", default=None,
)
_CURRENT_NAMESPACE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_namespace", default=None,
)


@contextmanager
def session_context(
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
):
    """Set ``session_id`` and ``namespace`` for the enclosed scope.

    If *session_id* is ``None`` and no parent context exists, a fresh
    ``sess_<uuid>`` is generated automatically.

    This is a synchronous context manager but works correctly in async
    code because :mod:`contextvars` are task-local.
    """
    current_session_id = _CURRENT_SESSION_ID.get()
    resolved_session_id = (
        session_id if session_id is not None else current_session_id
    )
    if resolved_session_id is None:
        resolved_session_id = f"sess_{uuid4().hex}"

    current_namespace = _CURRENT_NAMESPACE.get()
    resolved_namespace = namespace if namespace is not None else current_namespace

    session_token = _CURRENT_SESSION_ID.set(resolved_session_id)
    namespace_token = _CURRENT_NAMESPACE.set(resolved_namespace)
    try:
        yield
    finally:
        _CURRENT_SESSION_ID.reset(session_token)
        _CURRENT_NAMESPACE.reset(namespace_token)


def get_session_context() -> Mapping[str, Optional[str]]:
    """Return the current ``session_id`` and ``namespace``."""
    return {
        "session_id": _CURRENT_SESSION_ID.get(),
        "namespace": _CURRENT_NAMESPACE.get(),
    }
