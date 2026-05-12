# /// script
# dependencies = []
# ///
# ruff: noqa: T201

import argparse
import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import msgflux as mf
from msgflux import nn

mf.load_dotenv()


def _compact(value: Any, *, limit: int = 180) -> str:
    if isinstance(value, Mapping):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_event(event) -> None:
    attrs = event.attributes
    detail = {
        key: attrs.get(key)
        for key in (
            "agent_name",
            "response_type",
            "tool_name",
            "tool_call_id",
            "caller_name",
            "caller_namespace",
            "caller_session_id",
            "caller_run_id",
            "status",
            "result",
        )
        if attrs.get(key) is not None
    }
    print(f"{event.name}: {_compact(detail or attrs)}")


@mf.tool_config(inject_notification=True)
def lookup_ticket(ticket_id: str, notification) -> str:
    """Lookup a support ticket and publish a progress notification."""
    notification.update(
        status="lookup_started",
        hint=f"Looking up ticket {ticket_id}.",
        metadata={"ticket_id": ticket_id},
    )
    time.sleep(0.2)
    return f"{ticket_id}: owner=runtime, priority=high, status=investigating"


def build_agent(model_name: str) -> nn.Agent:
    return nn.Agent(
        name="event_streaming_support_agent",
        model=mf.Model.chat_completion(model_name),
        tools=[lookup_ticket],
        instructions=(
            "Use lookup_ticket when the user asks about a ticket. "
            "After the tool returns, answer with a short operational summary."
        ),
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--ticket", default="MSGFLUX-42")
    args = parser.parse_args()

    agent = build_agent(args.model)
    scope = mf.ExecutionScope(
        session_id="event-streaming-demo",
        run_id=f"ticket-{args.ticket.lower()}",
    )

    async for event in agent.astream_events(
        f"Check ticket {args.ticket} and summarize the current state.",
        scope=scope,
    ):
        print_event(event)

    print("\nEvent stream completed.")


if __name__ == "__main__":
    asyncio.run(main())
