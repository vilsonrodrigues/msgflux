import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType
from msgflux.tools.builtin import Brief, SendUserMessage


def test_send_user_message_tool_emits_user_facing_event():
    tool = SendUserMessage()

    with execution_context(session_id="session_a", namespace="agent"):
        with EventStream() as stream:
            result = tool(
                message="I am checking the repository now.",
                title="Progress",
                attachments=[
                    {
                        "path": "/workspace/screenshot.png",
                        "mime_type": "image/png",
                        "name": "screenshot",
                    }
                ],
                metadata={"surface": "cli"},
            )
            stream.close()
            events = stream.events

    assert result == {"message": "Message sent to the user."}
    assert [event.name for event in events] == [EventType.USER_MESSAGE_SENT]
    assert events[0].attributes["message"] == "I am checking the repository now."
    assert events[0].attributes["title"] == "Progress"
    assert events[0].attributes["attachments"] == [
        {
            "path": "/workspace/screenshot.png",
            "mime_type": "image/png",
            "name": "screenshot",
        }
    ]
    assert events[0].attributes["metadata"] == {"surface": "cli"}
    assert events[0].attributes["scope"]["session_id"] == "session_a"
    assert events[0].attributes["scope"]["namespace"] == "agent"


def test_send_user_message_tool_library_executes_and_preserves_metadata():
    library = ToolLibrary("agent", [SendUserMessage()])

    with EventStream() as stream:
        responses = library(
            [
                (
                    "call_1",
                    "SendUserMessage",
                    {
                        "message": "Running the test suite.",
                        "attachments": [{"url": "https://example.com/report.png"}],
                        "metadata": {"phase": "tests"},
                    },
                )
            ]
        )
        stream.close()
        events = stream.events

    response = responses.get_by_name("SendUserMessage")
    assert response is not None
    assert response.error is None
    assert response.result == {"message": "Message sent to the user."}
    assert library.library["SendUserMessage"].display_name == "Send User Message"
    assert (
        library.library["SendUserMessage"].description == SendUserMessage.description
    )

    names = [event.name for event in events]
    assert EventType.TOOL_STARTED in names
    assert EventType.USER_MESSAGE_SENT in names
    assert EventType.TOOL_RESULT in names

    event = next(event for event in events if event.name == EventType.USER_MESSAGE_SENT)
    assert event.attributes["message"] == "Running the test suite."
    assert event.attributes["attachments"] == [
        {"url": "https://example.com/report.png"}
    ]
    assert event.attributes["metadata"] == {"phase": "tests"}


@pytest.mark.asyncio
async def test_send_user_message_tool_async_call_emits_event():
    tool = SendUserMessage()

    with EventStream() as stream:
        result = await tool.acall("Still working on it.")
        stream.close()
        events = stream.events

    assert result == {"message": "Message sent to the user."}
    assert [event.name for event in events] == [EventType.USER_MESSAGE_SENT]


def test_send_user_message_tool_rejects_empty_message():
    tool = SendUserMessage()

    with pytest.raises(ValueError, match="SendUserMessage `message` cannot be empty"):
        tool("   ")


def test_brief_alias_points_to_send_user_message():
    assert Brief is SendUserMessage
