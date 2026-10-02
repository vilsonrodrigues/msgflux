"""Tool end events snapshot structured results without changing tool outcomes."""

from dataclasses import dataclass

import msgspec
import pytest

from msgflux.nn.hooks import Hook
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType, _CURRENT_EVENT_SINK, _EventSink
from msgflux.tools.shell import ShellCommandResult, ShellResult


@dataclass
class DataclassResult:
    count: int
    label: str


class StructResult(msgspec.Struct):
    count: int
    label: str


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("result_factory", "event_result"),
    [
        (lambda: DataclassResult(2, "ok"), {"count": 2, "label": "ok"}),
        (lambda: StructResult(3, "ready"), {"count": 3, "label": "ready"}),
        (
            lambda: ShellResult(
                results=(
                    ShellCommandResult(status="exited", returncode=0, stdout="ok"),
                )
            ),
            {
                "results": (
                    {
                        "status": "exited",
                        "returncode": 0,
                        "stdout": "ok",
                        "stderr": "",
                    },
                )
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_tool_end_event_snapshots_structured_result_but_preserves_object(
    asynchronous, result_factory, event_result
):
    result = result_factory()
    returned = []
    hooked = []

    if asynchronous:

        async def produce() -> object:
            return result

    else:

        def produce() -> object:
            return result

    owner = Module()

    def capture_after_tool(event):
        hooked.append(event.result)
        return event

    Hook(event="after_tool", handler=capture_after_tool).register(owner)
    library = ToolLibrary(name="results", tools=[produce])
    library.set_lifecycle_owner(owner)
    events = []
    token = _CURRENT_EVENT_SINK.set(_EventSink(events.append))
    try:
        if asynchronous:
            response = await library.acall(
                tool_callings=[("call_result", "produce", {})]
            )
        else:
            response = library(tool_callings=[("call_result", "produce", {})])
    finally:
        _CURRENT_EVENT_SINK.reset(token)

    returned.append(response.tool_calls[0].result)
    tool_end = next(event for event in events if event.type == EventType.TOOL_END)

    assert tool_end.data["result"] == event_result
    assert hooked == [result]
    assert hooked[0] is result
    assert returned[0] is result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("falsey", [None, False, 0, ""])
@pytest.mark.asyncio
async def test_tool_end_event_preserves_falsey_results(asynchronous, falsey):
    if asynchronous:

        async def produce() -> object:
            return falsey

    else:

        def produce() -> object:
            return falsey

    library = ToolLibrary(name="falsey", tools=[produce])
    events = []
    token = _CURRENT_EVENT_SINK.set(_EventSink(events.append))
    try:
        if asynchronous:
            response = await library.acall(
                tool_callings=[("call_falsey", "produce", {})]
            )
        else:
            response = library(tool_callings=[("call_falsey", "produce", {})])
    finally:
        _CURRENT_EVENT_SINK.reset(token)

    tool_end = next(event for event in events if event.type == EventType.TOOL_END)
    assert tool_end.data["result"] == falsey
    assert response.tool_calls[0].result == falsey
