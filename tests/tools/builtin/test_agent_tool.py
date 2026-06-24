from unittest.mock import Mock

import msgflux as mf
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.context import execution_context
from msgflux.tools.builtin import AgentTool


class _ScriptedModel:
    def __init__(self, *texts: str):
        self.model_type = "chat_completion"
        self._texts = list(texts or ("ok",))
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._texts:
            raise AssertionError("Scripted model exhausted.")
        return _response(self._texts.pop(0))

    async def acall(self, **kwargs):
        return self(**kwargs)


def _response(text: str) -> Mock:
    resp = Mock(spec=ModelResponse)
    resp.response_type = "text_generation"
    resp.consume.return_value = text
    resp.data = text
    resp.reasoning = None
    resp.metadata = {}
    return resp


def _mock_model(*texts: str) -> _ScriptedModel:
    return _ScriptedModel(*texts)


def _extract_task_id(result: str) -> str:
    return result.split("task_id='", maxsplit=1)[1].split("'", maxsplit=1)[0]


def test_agent_tool_exposes_single_agent_tool_with_name_and_message_params():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    tool = AgentTool([reviewer])
    library = ToolLibrary(name="lib", tools=[tool])

    schema = library.get_tool_json_schemas()[0]

    assert schema["function"]["name"] == "agent"
    properties = schema["function"]["parameters"]["properties"]
    assert set(properties) == {"name", "message"}


def test_agent_tool_collects_agent_usage_guidance():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    reviewer.tool_config = {"usage_guidance": "Use for code review."}
    planner.usage_guidance = "Use for planning."

    tool = AgentTool([reviewer, planner])

    assert "reviewer: Use for code review." in tool.usage_guidance
    assert "planner: Use for planning." in tool.usage_guidance


def test_agent_tool_dispatches_to_selected_agent():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    planner = Agent(name="planner", model=_mock_model("planned"))
    library = ToolLibrary(name="lib", tools=[AgentTool([reviewer, planner])])

    response = library([("call_1", "agent", {"name": "reviewer", "message": "Go"})])

    assert response.tool_calls[0].result == "reviewed"
    assert len(reviewer.model.calls) == 1
    assert planner.model.calls == []


def test_agent_tool_rejects_unknown_agent_name():
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    library = ToolLibrary(name="lib", tools=[AgentTool([reviewer])])

    response = library([("call_1", "agent", {"name": "missing", "message": "Go"})])

    assert response.tool_calls[0].result is None
    assert "Agent `missing` not found" in response.tool_calls[0].error
    assert "reviewer" in response.tool_calls[0].error


def test_agent_tool_background_run_uses_task_id_as_child_run_id():
    store = InMemoryCheckpointStore()
    reviewer = Agent(name="reviewer", model=_mock_model("reviewed"))
    agent_tool = mf.tool_config(allow_background=True)(AgentTool([reviewer]))
    library = ToolLibrary(name="lib", tools=[agent_tool])

    with execution_context(
        thread_id="user_42",
        namespace="root",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library(
            [
                (
                    "call_1",
                    "agent",
                    {
                        "name": "reviewer",
                        "message": "Review this",
                        "run_in_background": True,
                    },
                )
            ]
        )

    task_id = _extract_task_id(dispatch.tool_calls[0].result)
    wait = library([("call_2", "task_wait", {"task_id": task_id, "timeout": 1.0})])

    assert wait.tool_calls[0].result == "reviewed"
    assert store.load_state("reviewer", "user_42", task_id)["status"] == "completed"


def test_agent_tool_background_task_message_resumes_selected_agent():
    store = InMemoryCheckpointStore()
    model = _mock_model("first", "second")
    reviewer = Agent(name="reviewer", model=model)
    agent_tool = mf.tool_config(allow_background=True)(AgentTool([reviewer]))
    library = ToolLibrary(name="lib", tools=[agent_tool])

    with execution_context(
        thread_id="user_42",
        namespace="root",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library(
            [
                (
                    "call_1",
                    "agent",
                    {
                        "name": "reviewer",
                        "message": "First",
                        "run_in_background": True,
                    },
                )
            ]
        )

    task_id = _extract_task_id(dispatch.tool_calls[0].result)
    wait = library([("call_2", "task_wait", {"task_id": task_id, "timeout": 1.0})])
    assert wait.tool_calls[0].result == "first"

    with execution_context(checkpoint_store=store):
        resumed = library(
            [("call_3", "task_message", {"task_id": task_id, "message": "Second"})]
        )
        assert resumed.tool_calls[0].result["status"] == "resumed"
        wait_resumed = library(
            [("call_4", "task_wait", {"task_id": task_id, "timeout": 1.0})]
        )

    assert wait_resumed.tool_calls[0].result == "second"
    old_state = store.load_state("reviewer", "user_42", task_id)
    assert old_state["status"] == "completed"
    task_state = library([("call_5", "task_status", {"task_id": task_id})])
    resumed_run_id = task_state.tool_calls[0].result["metadata"]["checkpoint_run_id"]
    assert resumed_run_id != task_id
    new_state = store.load_state("reviewer", "user_42", resumed_run_id)
    assert new_state["status"] == "completed"
