"""Execution context propagated through msgFlux runtimes.

This module owns the shared ContextVars used by components that need execution
identity and inherited runtime configuration. The first durable features mainly
use `session_id`, `run_id`, and `namespace`, but the extra fields are already
present so nested runtimes can propagate lineage consistently later.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Mapping, Optional
from uuid import uuid4

_CURRENT_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_session_id",
    default=None,
)
_CURRENT_NAMESPACE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_namespace",
    default=None,
)
_CURRENT_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_run_id",
    default=None,
)
_CURRENT_PARENT_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_parent_run_id",
    default=None,
)
_CURRENT_ROOT_RUN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "msgflux_root_run_id",
    default=None,
)
_CURRENT_CHECKPOINT_STORE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_checkpoint_store",
    default=None,
)
_CURRENT_AGENT_INBOX: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_agent_inbox",
    default=None,
)
_CURRENT_TASK_HANDLE: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_task_handle",
    default=None,
)
_CURRENT_TASK_ACTIVITY_RECORDER: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "msgflux_task_activity_recorder",
    default=None,
)


@contextmanager
def execution_context(
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    root_run_id: Optional[str] = None,
    checkpoint_store: Any = None,
    agent_inbox: Any = None,
    task_handle: Any = None,
    task_activity_recorder: Any = None,
):
    """Set execution identity for the enclosed scope.

    Resolution rules intentionally favor explicit inputs and otherwise inherit
    from the current context. When no session exists yet, a fresh session id is
    generated automatically.
    """

    current_session_id = _CURRENT_SESSION_ID.get()
    resolved_session_id = (
        session_id if session_id is not None else current_session_id
    )
    if resolved_session_id is None:
        resolved_session_id = f"sess_{uuid4().hex}"

    current_namespace = _CURRENT_NAMESPACE.get()
    resolved_namespace = namespace if namespace is not None else current_namespace

    current_run_id = _CURRENT_RUN_ID.get()
    resolved_run_id = run_id if run_id is not None else current_run_id

    current_parent_run_id = _CURRENT_PARENT_RUN_ID.get()
    resolved_parent_run_id = (
        parent_run_id if parent_run_id is not None else current_parent_run_id
    )

    current_root_run_id = _CURRENT_ROOT_RUN_ID.get()
    if root_run_id is not None:
        resolved_root_run_id = root_run_id
    elif current_root_run_id is not None:
        resolved_root_run_id = current_root_run_id
    else:
        resolved_root_run_id = resolved_run_id

    current_checkpoint_store = _CURRENT_CHECKPOINT_STORE.get()
    resolved_checkpoint_store = (
        checkpoint_store if checkpoint_store is not None else current_checkpoint_store
    )
    current_agent_inbox = _CURRENT_AGENT_INBOX.get()
    resolved_agent_inbox = (
        agent_inbox if agent_inbox is not None else current_agent_inbox
    )
    current_task_handle = _CURRENT_TASK_HANDLE.get()
    resolved_task_handle = (
        task_handle if task_handle is not None else current_task_handle
    )
    current_task_activity_recorder = _CURRENT_TASK_ACTIVITY_RECORDER.get()
    resolved_task_activity_recorder = (
        task_activity_recorder
        if task_activity_recorder is not None
        else current_task_activity_recorder
    )

    session_token = _CURRENT_SESSION_ID.set(resolved_session_id)
    namespace_token = _CURRENT_NAMESPACE.set(resolved_namespace)
    run_token = _CURRENT_RUN_ID.set(resolved_run_id)
    parent_run_token = _CURRENT_PARENT_RUN_ID.set(resolved_parent_run_id)
    root_run_token = _CURRENT_ROOT_RUN_ID.set(resolved_root_run_id)
    checkpoint_token = _CURRENT_CHECKPOINT_STORE.set(resolved_checkpoint_store)
    inbox_token = _CURRENT_AGENT_INBOX.set(resolved_agent_inbox)
    task_handle_token = _CURRENT_TASK_HANDLE.set(resolved_task_handle)
    activity_token = _CURRENT_TASK_ACTIVITY_RECORDER.set(
        resolved_task_activity_recorder
    )
    try:
        yield
    finally:
        _CURRENT_SESSION_ID.reset(session_token)
        _CURRENT_NAMESPACE.reset(namespace_token)
        _CURRENT_RUN_ID.reset(run_token)
        _CURRENT_PARENT_RUN_ID.reset(parent_run_token)
        _CURRENT_ROOT_RUN_ID.reset(root_run_token)
        _CURRENT_CHECKPOINT_STORE.reset(checkpoint_token)
        _CURRENT_AGENT_INBOX.reset(inbox_token)
        _CURRENT_TASK_HANDLE.reset(task_handle_token)
        _CURRENT_TASK_ACTIVITY_RECORDER.reset(activity_token)


@contextmanager
def session_context(
    *,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
):
    """Compatibility helper for callers that only care about session scope."""

    with execution_context(session_id=session_id, namespace=namespace):
        yield


def get_execution_context() -> Mapping[str, Optional[Any]]:
    """Return the current execution context."""

    return {
        "session_id": _CURRENT_SESSION_ID.get(),
        "namespace": _CURRENT_NAMESPACE.get(),
        "run_id": _CURRENT_RUN_ID.get(),
        "parent_run_id": _CURRENT_PARENT_RUN_ID.get(),
        "root_run_id": _CURRENT_ROOT_RUN_ID.get(),
        "checkpoint_store": _CURRENT_CHECKPOINT_STORE.get(),
        "agent_inbox": _CURRENT_AGENT_INBOX.get(),
        "task_handle": _CURRENT_TASK_HANDLE.get(),
        "task_activity_recorder": _CURRENT_TASK_ACTIVITY_RECORDER.get(),
    }


def get_session_context() -> Mapping[str, Optional[str]]:
    """Return the current session-scoped context."""

    return {
        "session_id": _CURRENT_SESSION_ID.get(),
        "namespace": _CURRENT_NAMESPACE.get(),
    }
