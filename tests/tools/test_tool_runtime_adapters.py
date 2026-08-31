import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.config import tool_config
from msgflux.tools.runtime import ToolIntent


def test_library_executes_canonical_intent_with_compiled_feedback():
    @tool_config(return_direct=True)
    def double(value: int) -> int:
        return value * 2

    library = ToolLibrary(name="test", tools=[double])
    intent = ToolIntent(id="call_1", name="double", arguments={"value": 4})

    outcomes = library.execute_intents([intent])

    assert len(outcomes) == 1
    assert outcomes[0].result == 8
    assert outcomes[0].status == "completed"
    assert outcomes[0].feedback.name == "direct"


@pytest.mark.asyncio
async def test_library_aexecutes_canonical_intent_and_normalizes_failure():
    async def explode() -> None:
        raise RuntimeError("boom")

    library = ToolLibrary(name="test", tools=[explode])
    intent = ToolIntent(id="call_1", name="explode")

    outcomes = await library.aexecute_intents((intent,))

    assert outcomes[0].status == "execution_failed"
    assert outcomes[0].error.code == "tool_execution_failed"
    assert "boom" in outcomes[0].error.message


def test_library_normalizes_unknown_tool_as_not_found():
    library = ToolLibrary(name="test", tools=[])
    intent = ToolIntent(id="call_1", name="missing")

    outcomes = library.execute_intents([intent])

    assert outcomes[0].status == "not_found"
    assert outcomes[0].error.code == "tool_not_found"


def test_library_preserves_custom_compiled_feedback_mode():
    @tool_config(feedback="approval")
    def deploy(environment: str) -> str:
        return environment

    library = ToolLibrary(name="test", tools=[deploy])
    intent = ToolIntent(
        id="call_1",
        name="deploy",
        arguments={"environment": "staging"},
    )

    outcome = library.execute_intents([intent])[0]

    assert outcome.result == "staging"
    assert outcome.feedback.name == "approval"
