import pytest

from msgflux.nn.modules.agent import Agent
from msgflux.sandbox import Sandbox, get_ptc_allowed_tool_names, ptc_context


class MockModel:
    model_type = "chat_completion"

    def __call__(self, **kwargs):
        return kwargs

    async def acall(self, **kwargs):
        return kwargs


def search(query: str) -> str:
    """Search the knowledge base."""
    return f"found:{query}"


def send_user_message(message: str) -> str:
    """Send a message to the user."""
    return f"sent:{message}"


def test_sandbox_factory_creates_python_interpreter():
    sandbox = Sandbox.python("local")

    assert sandbox.name == "python_interpreter"
    assert sandbox.capabilities.programmatic_tool_calls is True


def test_python_sandbox_persists_variables_between_calls():
    sandbox = Sandbox.python("local")

    sandbox("counter = 41")
    result = sandbox("result = counter + 1")

    assert result == "42"


def test_ptc_context_scopes_allowed_tool_names():
    assert get_ptc_allowed_tool_names() == frozenset()

    with ptc_context({"search"}):
        assert get_ptc_allowed_tool_names() == frozenset({"search"})

    assert get_ptc_allowed_tool_names() == frozenset()


def test_agent_registers_code_interpreter_when_ptc_enabled():
    agent = Agent(
        name="agent",
        model=MockModel(),
        tools=[search],
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True, "ptc_tools": {"allow": "*"}}},
    )

    assert "python_interpreter" in agent.tool_library.get_tool_names()
    assert "search" in agent.tool_library.get_tool_names()


def test_code_interpreter_description_uses_filtered_ptc_tools():
    agent = Agent(
        name="agent",
        model=MockModel(),
        tools=[search, send_user_message],
        code_interpreter=Sandbox.python("local"),
        config={
            "code_interpreter": {
                "ptc": True,
                "ptc_tools": {"allow": "*", "block": ["send_user_message"]},
            }
        },
    )

    params = agent.inspect_model_execution_params(
        "hello",
        tool_filter={"allow": ["python_interpreter", "search", "send_user_message"]},
    )
    schemas = params.tool_definitions.schemas
    interpreter_schema = next(
        schema
        for schema in schemas
        if schema["function"]["name"] == "python_interpreter"
    )
    description = interpreter_schema["function"]["description"]

    assert "tools.search" in description
    assert "tools.send_user_message" not in description


@pytest.mark.asyncio
async def test_code_interpreter_can_call_allowed_ptc_tool():
    agent = Agent(
        name="agent",
        model=MockModel(),
        tools=[search],
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True, "ptc_tools": {"allow": "*"}}},
    )

    with ptc_context({"search"}):
        responses = await agent.tool_library.acall(
            [
                (
                    "call_1",
                    "python_interpreter",
                    {"code": "result = tools.search(query='msgflux')"},
                )
            ]
        )

    response = responses.get_by_name("python_interpreter")
    assert response is not None
    assert response.error is None
    assert response.result == "found:msgflux"


@pytest.mark.asyncio
async def test_code_interpreter_rejects_blocked_ptc_tool():
    agent = Agent(
        name="agent",
        model=MockModel(),
        tools=[search],
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True, "ptc_tools": {"allow": "*"}}},
    )

    with ptc_context(set()):
        responses = await agent.tool_library.acall(
            [
                (
                    "call_1",
                    "python_interpreter",
                    {"code": "result = tools.search(query='msgflux')"},
                )
            ]
        )

    response = responses.get_by_name("python_interpreter")
    assert response is not None
    assert response.error is not None
    assert "Tool `search` is not available" in response.error
