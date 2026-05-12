# /// script
# dependencies = []
# ///
# ruff: noqa: T201

import argparse
import asyncio

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.msgspec import msgspec_dumps

EXECUTED_UPDATES: list[str] = []


def tool_call_response(
    tool_name: str, parameters: dict, *, call_id: str
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(agg)
    response.reasoning = None
    response.metadata = {}
    return response


def text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class ScriptedModel:
    """Small deterministic model used to make the example runnable offline."""

    model_type = "chat_completion"

    def __init__(self, responses):
        self._responses = list(responses)

    def __call__(self, **kwargs):  # noqa: ARG002
        if not self._responses:
            raise RuntimeError("Scripted model exhausted.")
        return self._responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


@mf.tool_config(
    approval={
        "action": "ticket.write",
        "risk": "high",
        "resource_arg": "ticket_id",
        "reason": "The tool changes ticket state.",
    }
)
def update_ticket(ticket_id: str, status: str) -> str:
    """Update a support ticket.

    Args:
        ticket_id: Ticket identifier.
        status: New ticket status.
    """
    EXECUTED_UPDATES.append(f"{ticket_id}:{status}")
    return f"{ticket_id} updated to {status}"


def build_agents() -> nn.Agent:
    worker_model = ScriptedModel(
        [
            tool_call_response(
                "update_ticket",
                {"ticket_id": "MSGFLUX-42", "status": "approved"},
                call_id="call_worker_update",
            ),
            text_response("Worker finished after updating MSGFLUX-42."),
        ]
    )
    worker = nn.Agent(
        name="ticket_worker",
        model=worker_model,
        tools=[update_ticket],
        instructions="Update tickets when the coordinator delegates ticket work.",
    )

    coordinator_model = ScriptedModel(
        [
            tool_call_response(
                "ticket_worker",
                {"task": "Set MSGFLUX-42 to approved."},
                call_id="call_delegate_worker",
            ),
            text_response("Coordinator received the worker result and completed."),
        ]
    )
    return nn.Agent(
        name="support_coordinator",
        model=coordinator_model,
        tools=[worker],
        permission_manager=mf.PermissionManager(default_mode="ask_user"),
        instructions="Delegate ticket state changes to ticket_worker.",
    )


async def wait_for_permission(manager: mf.PermissionManager):
    for _ in range(100):
        pending = manager.list_pending()
        if pending:
            return pending[0]
        await asyncio.sleep(0.01)
    raise TimeoutError("No pending permission request was created.")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["ask_user", "bypass", "deny"],
        default="ask_user",
        help="Permission mode carried by ExecutionScope.",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "deny"],
        default="approve",
        help="External decision used only when --mode ask_user.",
    )
    args = parser.parse_args()
    EXECUTED_UPDATES.clear()

    agent = build_agents()
    manager = agent.permission_manager
    scope = mf.ExecutionScope(
        session_id="approval-demo-session",
        run_id="approval-demo-run",
        permission_mode=args.mode,
    )

    task = asyncio.create_task(
        agent.acall("Please approve the ticket update.", scope=scope)
    )

    if args.mode == "ask_user":
        request = await wait_for_permission(manager)
        print("Permission requested:")
        print(f"- request_id: {request.request_id}")
        print(f"- action: {request.action}")
        print(f"- tool_name: {request.tool_name}")
        print(f"- caller_name: {request.caller_name}")
        print(f"- resource: {request.resource}")
        print(f"- risk: {request.risk}")

        if args.decision == "approve":
            manager.approve(request.request_id, reason="approved from demo")
        else:
            manager.deny(request.request_id, reason="denied from demo")

    result = await task
    print("\nAgent result:")
    print(result)
    print("\nExecuted ticket updates:")
    print(EXECUTED_UPDATES)


if __name__ == "__main__":
    asyncio.run(main())
