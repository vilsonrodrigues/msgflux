"""Tests for public Module event streams."""

from dataclasses import replace

import pytest
from unittest.mock import Mock

from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import BeforeRun
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType


class EchoModule(Module):
    def forward(self, value):
        return value.upper()

    async def aforward(self, value):
        return value.upper()


class StreamingModule(Module):
    def forward(self):
        response = ModelStreamResponse()
        response.set_response_type("text_generation")
        response.add_reasoning("think")
        response.add("hello")
        response.add(" world")
        response.finish()
        return response


class FailingModule(Module):
    def forward(self):
        raise RuntimeError("boom")


def make_agent(*, hooks=None):
    model = Mock()
    model.model_type = "chat_completion"
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("hello")
    model.return_value = response
    return Agent(name="agent", model=model, hooks=hooks), model

    async def aforward(self):
        raise RuntimeError("boom")


def test_stream_events_yields_ordered_message_lifecycle():
    events = list(EchoModule().stream_events("hello"))

    assert [event.sequence for event in events] == list(range(len(events)))
    assert [event.type for event in events] == [
        EventType.RUN_START,
        EventType.TURN_START,
        EventType.MESSAGE_START,
        EventType.MESSAGE_END,
        EventType.TURN_END,
        EventType.RUN_END,
    ]
    assert events[-1].data["output"] == "HELLO"


def test_stream_events_owns_and_consumes_model_stream():
    events = list(StreamingModule().stream_events())

    content = [
        event.data["delta"] for event in events if event.type == EventType.MESSAGE_DELTA
    ]
    reasoning = [
        event.data["delta"]
        for event in events
        if event.type == EventType.REASONING_DELTA
    ]

    assert content == ["hello", " world"]
    assert reasoning == ["think"]
    assert events[-1].data["output"] == "hello world"


@pytest.mark.asyncio
async def test_astream_events_yields_events_while_running():
    events = [event async for event in EchoModule().astream_events("hello")]

    assert [event.type for event in events] == [
        EventType.RUN_START,
        EventType.TURN_START,
        EventType.MESSAGE_START,
        EventType.MESSAGE_END,
        EventType.TURN_END,
        EventType.RUN_END,
    ]


def test_stream_events_reraises_execution_error_after_error_event():
    seen = []

    with pytest.raises(RuntimeError, match="boom"):
        for event in FailingModule().stream_events():
            seen.append(event)

    assert seen[-1].type == EventType.RUN_ERROR
    assert seen[-1].data["error"] == "boom"


@pytest.mark.asyncio
async def test_astream_events_reraises_execution_error_after_error_event():
    seen = []

    with pytest.raises(RuntimeError, match="boom"):
        async for event in FailingModule().astream_events():
            seen.append(event)

    assert seen[-1].type == EventType.RUN_ERROR


def test_agent_events_include_resolved_execution_identity_and_model_boundaries():
    agent, _ = make_agent()

    events = list(agent.stream_events("hi"))

    assert all(event.thread_id for event in events)
    assert all(event.run_id for event in events)
    assert EventType.MODEL_REQUEST in [event.type for event in events]
    assert EventType.MODEL_RESPONSE in [event.type for event in events]


def test_agent_lifecycle_hooks_transform_canonical_response():
    def add_suffix(response):
        response.data += "!"
        return response

    agent, _ = make_agent(hooks=[Hook(event="after_response", handler=add_suffix)])

    assert agent("hi") == "hello!"


def test_tool_events_wrap_validated_foreground_execution():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    library = ToolLibrary(name="math", tools=[double])

    events = list(
        library.stream_events(tool_callings=[("call_1", "double", {"value": 3})])
    )
    tool_events = [
        event
        for event in events
        if event.type in {EventType.TOOL_START, EventType.TOOL_END}
    ]

    assert [event.type for event in tool_events] == [
        EventType.TOOL_START,
        EventType.TOOL_END,
    ]
    assert tool_events[0].data["tool_call_id"] == "call_1"
    assert tool_events[1].data["result"] == 6


def test_transform_output_changes_return_without_changing_canonical_history():
    agent, _ = make_agent(
        hooks=[
            Hook(
                event="transform_output",
                handler=lambda output: output.replace("hello", "expanded report"),
            )
        ]
    )
    history = ChatMessages()

    output = agent("hi", messages=history)
    assistant_messages = [item for item in history if item.get("role") == "assistant"]

    assert output == "expanded report"
    assert assistant_messages[-1]["content"] == "hello"


def test_transform_output_buffers_stream_content_but_keeps_reasoning_live():
    module = StreamingModule()
    Hook(
        event="transform_output",
        handler=lambda output: output.upper(),
    ).register(module)

    events = list(module.stream_events())

    assert not [event for event in events if event.type == EventType.MESSAGE_DELTA]
    assert [
        event.data["delta"]
        for event in events
        if event.type == EventType.REASONING_DELTA
    ] == ["think"]
    message_start = next(
        event for event in events if event.type == EventType.MESSAGE_START
    )
    message_end = next(event for event in events if event.type == EventType.MESSAGE_END)
    assert message_start.data["buffered"] is True
    assert message_end.data["content"] == "HELLO WORLD"


@pytest.mark.asyncio
async def test_async_transform_output_runs_after_stream_accumulation():
    async def transform(output):
        return f"[{output}]"

    module = StreamingModule()
    Hook(event="transform_output", handler=transform).register(module)

    events = [event async for event in module.astream_events()]

    assert events[-1].data["output"] == "[hello world]"


def test_before_run_can_replace_fresh_agent_input():
    agent, model = make_agent(
        hooks=[
            Hook(
                event="before_run",
                handler=lambda event: replace(event, message="replacement"),
            )
        ]
    )

    agent("original")

    sent_messages = model.call_args.kwargs["messages"]
    assert sent_messages[-1]["content"] == "<task>replacement</task>"


def test_before_run_receives_typed_payload():
    received = []

    def capture(event):
        received.append(event)

    agent, _ = make_agent(hooks=[Hook(event="before_run", handler=capture)])

    agent("hello")

    assert isinstance(received[0], BeforeRun)


def test_before_and_after_tool_hooks_modify_validated_tool_call():
    def double(value: int) -> int:
        """Double a value."""
        return value * 2

    owner = EchoModule()
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, arguments={"value": 4}),
    ).register(owner)
    Hook(
        event="after_tool",
        handler=lambda event: replace(event, result=event.result + 1),
    ).register(owner)
    library = ToolLibrary(name="math", tools=[double])
    library.set_lifecycle_owner(owner)

    response = library(tool_callings=[("call_1", "double", {"value": 3})])

    assert response.tool_calls[0].parameters == {"value": 4}
    assert response.tool_calls[0].result == 9


def test_before_tool_preserves_injected_tool_context():
    seen = {}

    def inspect(value: int, messages: list) -> int:
        """Inspect a value and the inherited conversation."""
        seen["messages"] = messages
        return value

    inspect.tool_config = {"inject_messages": True}
    owner = EchoModule()
    Hook(
        event="before_tool",
        handler=lambda event: replace(event, arguments={"value": 7}),
    ).register(owner)
    library = ToolLibrary(name="inspection", tools=[inspect])
    library.set_lifecycle_owner(owner)
    messages = [{"role": "user", "content": "context"}]

    response = library(
        tool_callings=[("call_1", "inspect", {"value": 3})],
        messages=messages,
    )

    assert response.tool_calls[0].result == 7
    assert seen["messages"] is messages


def test_before_tool_hook_failure_blocks_execution():
    calls = []

    def double(value: int) -> int:
        """Double a value."""
        calls.append(value)
        return value * 2

    def fail(_event):
        raise RuntimeError("policy unavailable")

    owner = EchoModule()
    Hook(event="before_tool", handler=fail).register(owner)
    library = ToolLibrary(name="math", tools=[double])
    library.set_lifecycle_owner(owner)

    response = library(tool_callings=[("call_1", "double", {"value": 3})])

    assert calls == []
    assert "failed closed" in response.tool_calls[0].error
