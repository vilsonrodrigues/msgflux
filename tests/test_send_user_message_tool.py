import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType
from msgflux.tools.builtin import Brief, SEND_USER_MESSAGE_TOOL_NAME, SendUserMessage


def test_send_user_message_tool_emits_user_facing_event():
    tool = SendUserMessage()

    with execution_context(session_id="session_a", namespace="agent"):
        with EventStream() as stream:
            result = tool(
                message="I am checking the repository now.",
                status="progress",
                title="Progress",
                attachments="/workspace/screenshot.png",
            )
            stream.close()
            events = stream.events

    assert result == {"message": "Message sent to the user."}
    assert [event.name for event in events] == [EventType.USER_MESSAGE_SENT]
    assert events[0].attributes["message"] == "I am checking the repository now."
    assert events[0].attributes["status"] == "progress"
    assert events[0].attributes["title"] == "Progress"
    assert events[0].attributes["attachments"] == ["/workspace/screenshot.png"]
    assert events[0].attributes["scope"]["session_id"] == "session_a"
    assert events[0].attributes["scope"]["namespace"] == "agent"


def test_send_user_message_tool_library_executes_and_preserves_metadata():
    library = ToolLibrary("agent", [SendUserMessage()])

    with EventStream() as stream:
        responses = library(
            [
                (
                    "call_1",
                    SEND_USER_MESSAGE_TOOL_NAME,
                    {
                        "message": "Running the test suite.",
                        "status": "success",
                        "attachments": [
                            "/workspace/report.png",
                            "/workspace/summary.png",
                        ],
                    },
                )
            ]
        )
        stream.close()
        events = stream.events

    response = responses.get_by_name(SEND_USER_MESSAGE_TOOL_NAME)
    assert response is not None
    assert response.error is None
    assert response.result == {"message": "Message sent to the user."}
    assert library.library[SEND_USER_MESSAGE_TOOL_NAME].display_name == (
        "Send User Message"
    )
    assert (
        library.library[SEND_USER_MESSAGE_TOOL_NAME].description
        == SendUserMessage.description
    )

    names = [event.name for event in events]
    assert EventType.TOOL_STARTED in names
    assert EventType.USER_MESSAGE_SENT in names
    assert EventType.TOOL_RESULT in names

    event = next(event for event in events if event.name == EventType.USER_MESSAGE_SENT)
    assert event.attributes["message"] == "Running the test suite."
    assert event.attributes["status"] == "success"
    assert event.attributes["attachments"] == [
        "/workspace/report.png",
        "/workspace/summary.png",
    ]


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


def test_send_user_message_tool_rejects_invalid_status():
    tool = SendUserMessage()

    with pytest.raises(ValueError, match="Invalid SendUserMessage status"):
        tool("Working.", status="done")


def test_send_user_message_tool_rejects_non_string_attachment():
    tool = SendUserMessage()

    with pytest.raises(TypeError, match="attachments must be paths as strings"):
        tool("Working.", attachments=["/workspace/report.png", 42])


def test_brief_alias_points_to_send_user_message():
    assert Brief is SendUserMessage
