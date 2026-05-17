# /// script
# dependencies = []
# ///
# ruff: noqa: T201

import argparse

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.msgspec import msgspec_dumps


def tool_call_response(
    tool_name: str,
    parameters: dict[str, str],
    *,
    call_id: str,
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    aggregator = ToolCallAggregator()
    aggregator.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(aggregator)
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
    """Deterministic model used to make the example runnable offline."""

    model_type = "chat_completion"

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def __call__(self, **kwargs: object) -> ModelResponse:  # noqa: ARG002
        if not self._responses:
            raise RuntimeError("Scripted model exhausted.")
        return self._responses.pop(0)

    async def acall(self, **kwargs: object) -> ModelResponse:
        return self(**kwargs)


def lookup_ticket(ticket_id: str) -> str:
    """Look up a support ticket by id."""
    return f"{ticket_id}: owner=runtime, priority=high, status=investigating"


def build_agent() -> nn.Agent:
    code = """
ticket = await tools.lookup_ticket(ticket_id=vars["ticket_id"])
result = f"{ticket}; inspected_by=python_interpreter"
""".strip()
    model = ScriptedModel(
        [
            tool_call_response(
                "python_interpreter",
                {"code": code},
                call_id="call_python_interpreter",
            ),
            text_response("The ticket context was inspected in the code interpreter."),
        ]
    )
    return nn.Agent(
        name="support_agent",
        model=model,
        tools=[lookup_ticket],
        code_interpreter=mf.Sandbox.python("local"),
        config={
            "code_interpreter": {
                "ptc": True,
                "ptc_tools": {"allow": ["lookup_ticket"]},
                "inject_vars": True,
                "notify_vars": True,
            }
        },
        instructions=(
            "Use the Python interpreter for lightweight runtime analysis. "
            "Use programmatic tools through the tools namespace."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket-id", default="MSGFLUX-42")
    args = parser.parse_args()

    agent = build_agent()
    response = agent(
        "Inspect this ticket with the code interpreter.",
        vars={"ticket_id": args.ticket_id},
    )
    print(response)


if __name__ == "__main__":
    main()
