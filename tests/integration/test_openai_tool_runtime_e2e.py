"""Live OpenAI validation for canonical tool intents and outcomes.

Requires ``OPENAI_API_KEY`` in the environment or repository ``.env``. Each
parameter performs paid API calls.
"""

import os

import pytest

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope

mf.load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live OpenAI integration tests.",
)


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
def test_openai_agent_executes_canonical_tool_loop(api_mode: str):
    observed_codes = []

    def lookup_status(code: str) -> str:
        """Return the status associated with an exact incident code."""
        observed_codes.append(code)
        return "scanner_restarted"

    store = InMemoryCheckpointStore()
    model_kwargs = {
        "api_mode": api_mode,
        "max_tokens": 300,
        "reasoning_effort": "low" if api_mode == "responses" else "none",
    }
    if api_mode == "responses":
        model_kwargs["store"] = False
    agent = Agent(
        name=f"tool_runtime_{api_mode}",
        model=mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            **model_kwargs,
        ),
        tools=[lookup_status],
        checkpoint_store=store,
        instructions=(
            "Always call lookup_status exactly once with the incident code from "
            "the user. Then report the returned status verbatim."
        ),
    )
    scope = ExecutionScope(
        thread_id=f"tool_runtime_{api_mode}",
        namespace=agent.name,
        run_id="canonical_tool_loop",
    )

    result = agent("Check incident code SCANNER-42.", scope=scope)

    assert observed_codes == ["SCANNER-42"]
    assert "scanner_restarted" in str(result)
    state = store.load_state(agent.name, scope.thread_id, scope.run_id)
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    outputs = [item for item in restored if item.get("type") == "function_call_output"]
    assert len(outputs) == 1
    assert outputs[0]["output"] == "scanner_restarted"
