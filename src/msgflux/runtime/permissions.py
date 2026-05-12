from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Literal, Mapping
from uuid import uuid4

from msgflux.runtime.events import (
    emit_permission_denied,
    emit_permission_granted,
    emit_permission_requested,
)

PermissionPolicy = Literal["bypass", "ask_user", "deny"]
PermissionRisk = Literal["low", "medium", "high"]


class PermissionRuntimeError(RuntimeError):
    """Base error for permission decisions."""


class PermissionDeniedError(PermissionRuntimeError):
    """Raised when a permission request is denied."""


class PermissionTimeoutError(PermissionRuntimeError):
    """Raised when a permission request is not answered in time."""


@dataclass(frozen=True)
class PermissionRequest:
    action: str
    request_id: str = field(default_factory=lambda: f"perm_{uuid4().hex[:12]}")
    resource: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    caller_name: str | None = None
    risk: PermissionRisk = "medium"
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != {}
        }


@dataclass(frozen=True)
class PermissionDecision:
    request_id: str
    approved: bool
    policy: PermissionPolicy
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    decided_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != {}
        }


class PermissionManager:
    """Async-first permission coordinator for runtime/tool approval flows."""

    def __init__(
        self,
        *,
        policy: PermissionPolicy = "bypass",
        timeout: float | None = None,
    ) -> None:
        self.policy = policy
        self.timeout = timeout
        self._pending: dict[str, asyncio.Future[PermissionDecision]] = {}
        self._requests: dict[str, PermissionRequest] = {}

    # --- Policy ---

    def set_policy(self, policy: PermissionPolicy) -> None:
        self._validate_policy(policy)
        self.policy = policy

    @contextmanager
    def use_policy(self, policy: PermissionPolicy) -> Iterator[None]:
        self._validate_policy(policy)
        previous_policy = self.policy
        self.policy = policy
        try:
            yield
        finally:
            self.policy = previous_policy

    # --- Request Lifecycle ---

    async def request(
        self,
        action: str,
        *,
        resource: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        caller_name: str | None = None,
        risk: PermissionRisk = "medium",
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        policy: PermissionPolicy | None = None,
        timeout: float | None = None,
    ) -> PermissionDecision:
        return await self.request_permission(
            PermissionRequest(
                action=action,
                resource=resource,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                caller_name=caller_name,
                risk=risk,
                reason=reason,
                metadata=deepcopy(dict(metadata or {})),
            ),
            policy=policy,
            timeout=timeout,
        )

    async def request_permission(
        self,
        request: PermissionRequest,
        *,
        policy: PermissionPolicy | None = None,
        timeout: float | None = None,
    ) -> PermissionDecision:
        resolved_policy = policy or self.policy
        self._requests[request.request_id] = request
        emit_permission_requested(
            {**request.to_dict(), "policy": resolved_policy},
        )

        if resolved_policy == "bypass":
            decision = PermissionDecision(
                request_id=request.request_id,
                approved=True,
                policy=resolved_policy,
                reason="bypass",
            )
            emit_permission_granted(decision.to_dict())
            return decision

        if resolved_policy == "deny":
            decision = PermissionDecision(
                request_id=request.request_id,
                approved=False,
                policy=resolved_policy,
                reason="denied by policy",
            )
            emit_permission_denied(decision.to_dict())
            return decision

        if resolved_policy != "ask_user":
            raise ValueError(f"Unknown permission policy: {resolved_policy}")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionDecision] = loop.create_future()
        self._pending[request.request_id] = future
        try:
            decision = await asyncio.wait_for(
                future,
                timeout=self.timeout if timeout is None else timeout,
            )
        except TimeoutError as exc:
            self._pending.pop(request.request_id, None)
            decision = PermissionDecision(
                request_id=request.request_id,
                approved=False,
                policy=resolved_policy,
                reason="permission request timed out",
            )
            emit_permission_denied(decision.to_dict())
            raise PermissionTimeoutError(decision.reason) from exc
        return decision

    async def enforce(self, *args: Any, **kwargs: Any) -> PermissionDecision:
        decision = await self.request(*args, **kwargs)
        if not decision.approved:
            raise PermissionDeniedError(decision.reason or "Permission denied.")
        return decision

    async def enforce_permission(
        self,
        request: PermissionRequest,
        *,
        policy: PermissionPolicy | None = None,
        timeout: float | None = None,
    ) -> PermissionDecision:
        decision = await self.request_permission(
            request,
            policy=policy,
            timeout=timeout,
        )
        if not decision.approved:
            raise PermissionDeniedError(decision.reason or "Permission denied.")
        return decision

    # --- External Decisions ---

    def approve(
        self,
        request_id: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionDecision:
        return self._resolve(
            PermissionDecision(
                request_id=request_id,
                approved=True,
                policy="ask_user",
                reason=reason,
                metadata=deepcopy(dict(metadata or {})),
            )
        )

    def deny(
        self,
        request_id: str,
        *,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PermissionDecision:
        return self._resolve(
            PermissionDecision(
                request_id=request_id,
                approved=False,
                policy="ask_user",
                reason=reason,
                metadata=deepcopy(dict(metadata or {})),
            )
        )

    def _resolve(self, decision: PermissionDecision) -> PermissionDecision:
        future = self._pending.pop(decision.request_id, None)
        if future is None:
            raise KeyError(
                f"Permission request `{decision.request_id}` is not pending."
            )
        if decision.approved:
            emit_permission_granted(decision.to_dict())
        else:
            emit_permission_denied(decision.to_dict())
        if not future.done():
            future.set_result(decision)
        return decision

    # --- Introspection ---

    def get_request(self, request_id: str) -> PermissionRequest | None:
        return self._requests.get(request_id)

    def list_pending(self) -> list[PermissionRequest]:
        return [
            request
            for request_id, request in self._requests.items()
            if request_id in self._pending
        ]

    @staticmethod
    def _validate_policy(policy: PermissionPolicy) -> None:
        if policy not in {"bypass", "ask_user", "deny"}:
            raise ValueError(f"Unknown permission policy: {policy}")
