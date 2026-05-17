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


def test_python_sandbox_does_not_return_stale_result():
    sandbox = Sandbox.python("local")

    assert sandbox("result = 42") == "42"
    assert sandbox("counter = 43") == ""
    assert sandbox("result = counter") == "43"


def test_python_sandbox_exposes_runtime_vars_without_persisting_namespace():
    sandbox = Sandbox.python("local")
    sandbox.set_vars({"ticket": {"id": "MSGFLUX-42"}})

    result = sandbox("result = vars['ticket']['id']")

    assert result == "MSGFLUX-42"
    assert "vars" not in sandbox._globals


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


def test_code_interpreter_vars_notice_is_added_to_model_messages():
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
    )

    params = agent.inspect_model_execution_params(
        "hello",
        vars={"tickets": [{"id": "MSGFLUX-42"}], "threshold": 0.7},
    )
    contents = [message["content"] for message in params.messages]

    assert any("<runtime_context>" in content for content in contents)
    assert any('name="tickets"' in content for content in contents)
    assert any('type="list"' in content for content in contents)
    assert any('name="threshold"' in content for content in contents)


def test_code_interpreter_vars_are_not_notified_when_disabled():
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"notify_vars": False}},
    )

    params = agent.inspect_model_execution_params("hello", vars={"ticket": "MSG-1"})

    assert all("<runtime_context>" not in message["content"] for message in params.messages)


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
async def test_code_interpreter_supports_top_level_await_for_ptc_tools():
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
                    {
                        "code": (
                            "ticket = await tools.search(query='msgflux')\n"
                            "result = f'ptc:{ticket}'"
                        )
                    },
                )
            ]
        )

    response = responses.get_by_name("python_interpreter")
    assert response is not None
    assert response.error is None
    assert response.result == "ptc:found:msgflux"


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
