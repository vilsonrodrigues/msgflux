from types import SimpleNamespace

import pytest

from msgflux.coding.approval import CodingApprovalController
from msgflux.runtime import AgentApprovals, ApprovalBinding, InMemoryApprovalStore


class FakeAgent:
    def __init__(self, store):
        self.approvals_store = store
        self.checkpoint = {
            "runtime": {
                "extensions": {
                    "pending_approvals": {"requests": {"call-1": "request-1"}}
                }
            }
        }
        self.previews = {}
        self.decisions = []

    def get_module_name(self):
        return "coding-agent"

    def inspect_approval_batch(self, thread_id, run_id):
        return self.checkpoint

    def inspect_approval_preview(self, thread_id, run_id, request_id, *, approvals):
        assert approvals.store is self.approvals_store
        return self.previews.get(request_id)

    def decide_approval(self, request_id, *, approved, decided_by, approvals):
        self.decisions.append((request_id, approved, decided_by))
        return approvals.store.decide(
            "coding-agent", request_id, approved=approved, decided_by=decided_by
        )


def _record(store, request_id="request-1", *, principal="alice", call_id="call-1"):
    binding = ApprovalBinding.from_call(
        namespace="coding-agent",
        thread_id="thread-1",
        run_id="run-1",
        principal=principal,
        tool_call_id=call_id,
        tool_name="write_file",
        tool_revision="v1",
        policy_version="p1",
        arguments={"path": "/secret", "content": "sensitive"},
        resources={},
    )
    return store.request(binding, request_id=request_id, expires_at=1000)


def _controller(principal="alice"):
    store = InMemoryApprovalStore(clock=lambda: 10)
    _record(store)
    agent = FakeAgent(store)
    policy = AgentApprovals(store, {"write_file": "v1"}, "p1")
    current_principal = [principal]
    controller = CodingApprovalController(
        agent, policy, principal=lambda: current_principal[0]
    )
    return controller, agent, store, current_principal


def test_pending_returns_checkpointed_review_and_prepared_diff_only():
    controller, agent, _, _ = _controller()
    agent.previews["request-1"] = SimpleNamespace(diff="--- a/file\n+++ b/file\n")

    reviews = controller.pending("thread-1", "run-1")

    assert len(reviews) == 1
    assert reviews[0].request_id == "request-1"
    assert reviews[0].tool_name == "write_file"
    assert reviews[0].diff == "--- a/file\n+++ b/file\n"
    assert "sensitive" not in repr(reviews)


def test_pending_filters_principal_and_non_checkpointed_requests():
    controller, agent, store, current = _controller("bob")
    _record(store, "other-request", principal="alice", call_id="other-call")
    assert controller.pending("thread-1", "run-1") == ()

    current[0] = "alice"
    reviews = controller.pending("thread-1", "run-1")
    assert [review.request_id for review in reviews] == ["request-1"]
    assert agent.previews == {}


def test_decide_uses_live_principal_and_existing_store():
    controller, agent, store, current = _controller()
    record = controller.decide("thread-1", "run-1", "request-1", approved=True)
    assert record.status == "approved"
    assert record.decided_by == "alice"
    assert agent.decisions == [("request-1", True, "alice")]

    current[0] = "bob"
    with pytest.raises(PermissionError, match="live principal"):
        controller.decide("thread-1", "run-1", "request-1", approved=False)
    assert store.get("coding-agent", "request-1").status == "approved"


@pytest.mark.parametrize(
    ("thread_id", "run_id", "request_id"),
    [
        ("other-thread", "run-1", "request-1"),
        ("thread-1", "other-run", "request-1"),
        ("thread-1", "run-1", "missing"),
    ],
)
def test_decide_rejects_request_outside_the_thread_run(thread_id, run_id, request_id):
    controller, agent, _, _ = _controller()
    agent.checkpoint = {
        "runtime": {"extensions": {"pending_approvals": {"requests": {}}}}
    }
    with pytest.raises(PermissionError, match="live principal"):
        controller.decide(thread_id, run_id, request_id, approved=True)
