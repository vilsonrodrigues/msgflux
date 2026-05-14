import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType
from msgflux.tools.builtin import Brief


def test_brief_tool_emits_user_facing_event():
    tool = Brief()

    with execution_context(session_id="session_a", namespace="agent"):
        with EventStream() as stream:
            result = tool(
                message="I am checking the repository now.",
                title="Progress",
                metadata={"surface": "cli"},
            )
            stream.close()
            events = stream.events

    assert result == {"message": "Brief sent to the user."}
    assert [event.name for event in events] == [EventType.BRIEF_MESSAGE]
    assert events[0].attributes["message"] == "I am checking the repository now."
    assert events[0].attributes["title"] == "Progress"
    assert events[0].attributes["metadata"] == {"surface": "cli"}
    assert events[0].attributes["scope"]["session_id"] == "session_a"
    assert events[0].attributes["scope"]["namespace"] == "agent"


def test_brief_tool_library_executes_and_preserves_metadata():
    library = ToolLibrary("agent", [Brief()])

    with EventStream() as stream:
        responses = library(
            [
                (
                    "call_1",
                    "brief",
                    {
                        "message": "Running the test suite.",
                        "metadata": {"phase": "tests"},
                    },
                )
            ]
        )
        stream.close()
        events = stream.events

    response = responses.get_by_name("brief")
    assert response is not None
    assert response.error is None
    assert response.result == {"message": "Brief sent to the user."}
    assert library.library["brief"].display_name == "Brief"
    assert library.library["brief"].description == Brief.description

    names = [event.name for event in events]
    assert EventType.TOOL_STARTED in names
    assert EventType.BRIEF_MESSAGE in names
    assert EventType.TOOL_RESULT in names

    brief = next(event for event in events if event.name == EventType.BRIEF_MESSAGE)
    assert brief.attributes["message"] == "Running the test suite."
    assert brief.attributes["metadata"] == {"phase": "tests"}


@pytest.mark.asyncio
async def test_brief_tool_async_call_emits_event():
    tool = Brief()

    with EventStream() as stream:
        result = await tool.acall("Still working on it.")
        stream.close()
        events = stream.events

    assert result == {"message": "Brief sent to the user."}
    assert [event.name for event in events] == [EventType.BRIEF_MESSAGE]


def test_brief_tool_rejects_empty_message():
    tool = Brief()

    with pytest.raises(ValueError, match="Brief `message` cannot be empty"):
        tool("   ")
