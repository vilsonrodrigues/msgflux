"""Small host-side approval review adapter for coding interfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import msgspec

from msgflux.runtime.approvals.agent import AgentApprovals
from msgflux.runtime.approvals.records import ApprovalRecord, require_name


class ApprovalReview(msgspec.Struct, frozen=True):
    """Safe fields for a review row; tool arguments are deliberately omitted."""

    request_id: str
    tool_call_id: str
    tool_name: str
    expires_at: float
    diff: str | None = None


class CodingApprovalController:
    """Review and resolve approvals through an Agent's existing approval APIs.

    ``principal`` must read the host's current authenticated identity each time
    it is called. It is checked against the principal bound to the tool call.
    """

    def __init__(
        self,
        agent,
        approvals: AgentApprovals,
        *,
        principal: Callable[[], str],
    ) -> None:
        if not isinstance(approvals, AgentApprovals):
            raise TypeError("approvals must be AgentApprovals")
        if not callable(principal):
            raise TypeError("principal must be a callable returning the live identity")
        self.agent = agent
        self.approvals = approvals
        self._principal = principal

    def pending(self, thread_id: str, run_id: str) -> tuple[ApprovalReview, ...]:
        """Return pending requests in this checkpointed run for the live principal."""
        thread_id, run_id = require_name(thread_id), require_name(run_id)
        principal = require_name(self._principal())
        namespace = self.agent.get_module_name()
        checkpoint = self.agent.inspect_approval_batch(thread_id, run_id)
        pending_state = (
            checkpoint.get("runtime", {})
            .get("extensions", {})
            .get("pending_approvals", {})
        )
        if not isinstance(pending_state, Mapping):
            return ()
        request_ids = set(pending_state.get("requests", {}).values())
        reviews = []
        for record in self.approvals.store.pending(namespace, thread_id, run_id):
            binding = record.binding
            if record.request_id not in request_ids or binding.principal != principal:
                continue
            change = self.agent.inspect_approval_preview(
                thread_id,
                run_id,
                record.request_id,
                approvals=self.approvals,
            )
            reviews.append(
                ApprovalReview(
                    request_id=record.request_id,
                    tool_call_id=binding.tool_call_id,
                    tool_name=binding.tool_name,
                    expires_at=record.expires_at,
                    diff=change.diff if change is not None else None,
                )
            )
        return tuple(reviews)

    def decide(
        self,
        thread_id: str,
        run_id: str,
        request_id: str,
        *,
        approved: bool,
    ) -> ApprovalRecord:
        """Resolve a request only while it remains pending and principal-bound."""
        thread_id, run_id = require_name(thread_id), require_name(run_id)
        request_id = require_name(request_id)
        principal = require_name(self._principal())
        namespace = self.agent.get_module_name()
        checkpoint = self.agent.inspect_approval_batch(thread_id, run_id)
        pending_state = (
            checkpoint.get("runtime", {})
            .get("extensions", {})
            .get("pending_approvals", {})
        )
        request_ids = set(pending_state.get("requests", {}).values())
        record = self.approvals.store.get(namespace, request_id)
        if (
            record is None
            or request_id not in request_ids
            or record.status != "pending"
            or record.binding.thread_id != thread_id
            or record.binding.run_id != run_id
            or record.binding.principal != principal
        ):
            raise PermissionError("Approval is not pending for this live principal")
        return self.agent.decide_approval(
            request_id,
            approved=approved,
            decided_by=principal,
            approvals=self.approvals,
        )


__all__ = ["ApprovalReview", "CodingApprovalController"]
