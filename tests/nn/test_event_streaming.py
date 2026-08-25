"""Tests for public Module event streams."""

import pytest
from unittest.mock import Mock

from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.hooks import Hook
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
