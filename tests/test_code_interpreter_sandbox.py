import pytest

from msgflux.nn.modules.agent import Agent
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.sandbox import (
    ArtifactNamespace,
    Sandbox,
    get_ptc_allowed_tool_names,
    normalize_artifacts,
    ptc_context,
)


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


def test_python_sandbox_returns_stdout_and_result():
    sandbox = Sandbox.python("local")

    result = sandbox("print('chunk:0:120')\nresult = 'done'")

    assert result == "chunk:0:120\ndone"


def test_python_sandbox_exposes_runtime_vars_without_persisting_namespace():
    sandbox = Sandbox.python("local")
    sandbox.set_vars({"ticket": {"id": "MSGFLUX-42"}})

    result = sandbox("result = vars['ticket']['id']")

    assert result == "MSGFLUX-42"
    assert "vars" not in sandbox._globals


def test_artifact_namespace_reads_with_offset_and_limit(tmp_path):
    artifact_path = tmp_path / "commands.txt"
    artifact_path.write_text("alpha\nbeta\ngamma\ngamma\n", encoding="utf-8")
    artifacts = normalize_artifacts({"commands": artifact_path})
    namespace = ArtifactNamespace(artifacts)

    assert namespace.list() == ["commands"]
    assert namespace.info("commands")["size"] == artifact_path.stat().st_size
    assert namespace.info("commands")["unit"] == "bytes"
    assert namespace.read("commands", offset=6, limit=4) == "beta"
    matches = namespace.search("commands", "gamma")
    assert [match["offset"] for match in matches] == [11, 17]


@pytest.mark.asyncio
async def test_artifact_namespace_async_reads_with_offset_and_limit(tmp_path):
    artifact_path = tmp_path / "commands.txt"
    artifact_path.write_text("alpha\nbeta\ngamma\ngamma\n", encoding="utf-8")
    artifacts = normalize_artifacts({"commands": artifact_path})
    namespace = ArtifactNamespace(artifacts)

    assert await namespace["list"]() == ["commands"]
    assert (await namespace["info"]("commands"))["unit"] == "bytes"
    assert await namespace.aread("commands", offset=6, limit=4) == "beta"
    assert await namespace["read"]("commands", offset=6, limit=4) == "beta"
    matches = await namespace.asearch("commands", "gamma")
    assert [match["offset"] for match in matches] == [11, 17]


def test_python_sandbox_sync_artifacts_namespace_requires_await(tmp_path):
    artifact_path = tmp_path / "README.md"
    artifact_path.write_text("msgFlux runtime artifacts", encoding="utf-8")
    sandbox = Sandbox.python("local")
    sandbox.set_artifacts(normalize_artifacts({"readme": artifact_path}))

    with pytest.raises(RuntimeError, match="Artifacts are async-first"):
        sandbox("result = artifacts['read']('readme', offset=8, limit=7)")
    assert "artifacts" not in sandbox._globals


def test_python_sandbox_sync_ptc_tools_require_async_execution():
    sandbox = Sandbox.python("local")
    sandbox.set_tools({"search": search})

    with pytest.raises(RuntimeError, match="async-first"):
        sandbox("result = tools['search'](query='msgflux')")


@pytest.mark.asyncio
async def test_python_sandbox_exposes_async_artifacts_without_persisting_namespace(
    tmp_path,
):
    artifact_path = tmp_path / "README.md"
    artifact_path.write_text("msgFlux runtime artifacts", encoding="utf-8")
    sandbox = Sandbox.python("local")
    sandbox.set_artifacts(normalize_artifacts({"readme": artifact_path}))

    result = await sandbox.acall(
        "result = await artifacts['read']('readme', offset=8, limit=7)"
    )

    assert result == "runtime"
    assert "artifacts" not in sandbox._globals


def test_monty_sandbox_executes_python_when_available():
    pytest.importorskip("pydantic_monty")
    sandbox = Sandbox.python("monty")
    sandbox.set_vars({"ticket": "MSGFLUX-42"})
    result = sandbox("print('debug')\nresult = vars['ticket']")

    assert result == "debug\nMSGFLUX-42"


@pytest.mark.asyncio
async def test_monty_sandbox_can_call_allowed_ptc_tool_when_available():
    pytest.importorskip("pydantic_monty")
    sandbox = Sandbox.python("monty")
    sandbox.set_tools({"search": search})

    with ptc_context({"search"}):
        result = await sandbox.acall(
            "result = await tools['search'](query='msgflux')"
        )

    assert result == "found:msgflux"


@pytest.mark.asyncio
async def test_monty_sandbox_exposes_artifacts_when_available(tmp_path):
    pytest.importorskip("pydantic_monty")
    artifact_path = tmp_path / "README.md"
    artifact_path.write_text("msgFlux runtime artifacts", encoding="utf-8")
    sandbox = Sandbox.python("monty")
    sandbox.set_artifacts(normalize_artifacts({"readme": artifact_path}))

    result = await sandbox.acall(
        "result = await artifacts['read']('readme', offset=8, limit=7)"
    )

    assert result == "runtime"


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
    assert "search" not in agent.tool_library.get_tool_names()
    assert agent.code_interpreter is not None
    assert "search" in agent.code_interpreter.get_tool_names()


def test_code_interpreter_exposes_usage_guidance_property():
    sandbox = Sandbox.python("local")

    assert "result" in sandbox.usage_guidance
    assert "print(...)" in sandbox.usage_guidance


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

    assert 'await tools["search"]' in description
    assert 'tools["search"].acall' not in description
    assert 'tools["send_user_message"]' not in description
    assert 'await artifacts["read"]' not in description


def test_code_interpreter_description_mentions_artifacts_when_enabled():
    agent = Agent(
        name="agent",
        model=MockModel(),
        tools=[search],
        code_interpreter=Sandbox.python("local"),
        config={
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": "*"},
            }
        },
    )

    params = agent.inspect_model_execution_params(
        "hello",
        tool_filter={"allow": ["python_interpreter", "search"]},
    )
    schemas = params.tool_definitions.schemas
    interpreter_schema = next(
        schema
        for schema in schemas
        if schema["function"]["name"] == "python_interpreter"
    )
    description = interpreter_schema["function"]["description"]

    assert 'await artifacts["info"]' in description
    assert 'await artifacts["read"]' in description
    assert "keyword arguments" in description


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

    assert any("<runtime_context " in content for content in contents)
    assert any('name="tickets"' in content for content in contents)
    assert any('type="list"' in content for content in contents)
    assert any('size="1"' in content for content in contents)
    assert any('unit="items"' in content for content in contents)
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


def test_code_interpreter_vars_notice_is_logged_in_verbose(capsys):
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"verbose": True},
    )

    agent.inspect_model_execution_params("hello", vars={"ticket": "MSG-1"})
    assert capsys.readouterr().out == ""

    messages = [{"role": "user", "content": "<task>hello</task>"}]
    agent._prepare_model_execution(messages=messages, vars={"ticket": "MSG-1"})

    output = capsys.readouterr().out
    assert "[agent][runtime_context]" in output
    assert "<runtime_context " in output
    assert 'name="ticket"' in output
    assert 'unit="chars"' in output


def test_code_interpreter_vars_notice_is_not_repeated_for_same_vars(capsys):
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"verbose": True},
    )
    messages = [{"role": "user", "content": "<task>hello</task>"}]

    agent._prepare_model_execution(messages=messages, vars={"ticket": "MSG-1"})
    first_output = capsys.readouterr().out
    agent._prepare_model_execution(messages=messages, vars={"ticket": "MSG-1"})
    second_output = capsys.readouterr().out

    assert "[agent][runtime_context]" in first_output
    assert "[agent][runtime_context]" not in second_output


def test_code_interpreter_artifacts_notice_is_added_to_model_messages(tmp_path):
    artifact_path = tmp_path / "commands.txt"
    artifact_path.write_text("echo hello", encoding="utf-8")
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True, "artifacts": True}},
    )

    params = agent.inspect_model_execution_params(
        "hello",
        artifacts={"commands": artifact_path},
    )
    contents = [message["content"] for message in params.messages]

    assert any("<artifacts " in content for content in contents)
    assert any('name="commands"' in content for content in contents)
    assert any('filename="commands.txt"' in content for content in contents)
    assert any('unit="bytes"' in content for content in contents)


def test_code_interpreter_artifacts_requires_config(tmp_path):
    artifact_path = tmp_path / "commands.txt"
    artifact_path.write_text("echo hello", encoding="utf-8")
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True}},
    )

    with pytest.raises(ValueError, match="artifacts"):
        agent.inspect_model_execution_params(
            "hello",
            artifacts={"commands": artifact_path},
        )


def test_code_interpreter_artifacts_notice_is_not_repeated(tmp_path, capsys):
    artifact_path = tmp_path / "commands.txt"
    artifact_path.write_text("echo hello", encoding="utf-8")
    artifacts = normalize_artifacts({"commands": artifact_path})
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={
            "verbose": True,
            "code_interpreter": {"ptc": True, "artifacts": True},
        },
    )
    messages = [{"role": "user", "content": "<task>hello</task>"}]

    agent._prepare_model_execution(messages=messages, vars={}, artifacts=artifacts)
    first_output = capsys.readouterr().out
    agent._prepare_model_execution(messages=messages, vars={}, artifacts=artifacts)
    second_output = capsys.readouterr().out

    assert "[agent][artifacts]" in first_output
    assert "[agent][artifacts]" not in second_output


@pytest.mark.asyncio
async def test_code_interpreter_empty_result_still_adds_tool_message():
    agent = Agent(
        name="agent",
        model=MockModel(),
        code_interpreter=Sandbox.python("local"),
        config={"code_interpreter": {"ptc": True, "ptc_tools": {"allow": "*"}}},
    )

    responses = await agent.tool_library.acall(
        [
            (
                "call_1",
                "python_interpreter",
                {"code": "counter = 1"},
            )
        ]
    )
    raw_response = ToolCallAggregator()
    raw_response.process(0, "call_1", "python_interpreter", '{"code":"counter = 1"}')
    id_results = {
        call.id: call.error if call.error is not None else call.result
        for call in responses.tool_calls
    }
    raw_response.insert_results(id_results)

    messages = raw_response.get_messages()

    assert len(messages) == 2
    assert messages[1]["role"] == "tool"
    assert messages[1]["content"] == ""


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
                    {
                        "code": (
                            "result = await tools['search'](query='msgflux')"
                        )
                    },
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
                            "ticket = await tools['search'](query='msgflux')\n"
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
async def test_code_interpreter_accepts_single_mapping_for_ptc_tool():
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
                    {"code": "result = await tools['search']({'query': 'msgflux'})"},
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
                    {"code": "result = tools['search'](query='msgflux')"},
                )
            ]
        )

    response = responses.get_by_name("python_interpreter")
    assert response is not None
    assert response.error is not None
    assert "Tool `search` is not available" in response.error
