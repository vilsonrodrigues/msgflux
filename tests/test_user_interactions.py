import asyncio
from threading import Thread

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType
from msgflux.runtime.interactions import (
    AskUserManager,
    UserInteractionCancelledError,
    UserInteractionTimeoutError,
    normalize_questions,
)
from msgflux.tools.builtin import AskUser


def _questions():
    return [
        {
            "question": "Which storage backend should we use?",
            "header": "Storage",
            "options": [
                {
                    "label": "SQLite",
                    "description": "Local durable storage with minimal setup.",
                },
                {
                    "label": "Redis",
                    "description": "Networked storage for server deployments.",
                },
            ],
        },
        {
            "question": "Which features should be enabled?",
            "header": "Features",
            "multi_select": True,
            "options": [
                {
                    "label": "Streaming",
                    "description": "Emit runtime events as execution progresses.",
                },
                {
                    "label": "Inbox",
                    "description": "Allow incoming messages during execution.",
                },
            ],
        },
    ]


async def _wait_for_pending(manager: AskUserManager):
    for _ in range(20):
        pending = manager.list_pending()
        if pending:
            return pending
        await asyncio.sleep(0)
    return manager.list_pending()


@pytest.mark.asyncio
async def test_ask_user_manager_blocks_until_answered_and_emits_events():
    manager = AskUserManager()

    with EventStream() as stream:
        task = asyncio.create_task(manager.request(_questions()))

        [pending] = await _wait_for_pending(manager)
        answer = manager.answer(
            pending.request_id,
            {
                "Which storage backend should we use?": "SQLite",
                "Which features should be enabled?": "Streaming, Inbox",
            },
        )
        result = await task
        stream.close()
        events = stream.events

    assert result == answer
    assert result.answers["Which storage backend should we use?"] == "SQLite"
    assert result.answers["Which features should be enabled?"] == "Streaming, Inbox"
    assert [event.name for event in events] == [
        EventType.USER_INTERACTION_REQUESTED,
        EventType.USER_INTERACTION_ANSWERED,
    ]
    assert events[0].attributes["questions"][0]["header"] == "Storage"
    assert events[0].attributes["questions"][1]["multi_select"] is True
    assert events[1].attributes["answers"] == dict(result.answers)


@pytest.mark.asyncio
async def test_ask_user_manager_answer_is_thread_safe():
    manager = AskUserManager()
    task = asyncio.create_task(manager.request([_questions()[0]]))

    [pending] = await _wait_for_pending(manager)
    thread = Thread(
        target=lambda: manager.answer(
            pending.request_id,
            {"Which storage backend should we use?": "Redis"},
        )
    )
    thread.start()
    thread.join()

    answer = await asyncio.wait_for(task, timeout=1)

    assert answer.answers["Which storage backend should we use?"] == "Redis"


@pytest.mark.asyncio
async def test_ask_user_manager_timeout_cancels_pending_request():
    manager = AskUserManager(timeout=0.001)

    with EventStream() as stream:
        with pytest.raises(UserInteractionTimeoutError):
            await manager.request([_questions()[0]])
        stream.close()
        events = stream.events

    assert manager.list_pending() == []
    assert [event.name for event in events] == [
        EventType.USER_INTERACTION_REQUESTED,
        EventType.USER_INTERACTION_CANCELLED,
    ]


@pytest.mark.asyncio
async def test_ask_user_manager_cancel_rejects_pending_request():
    manager = AskUserManager()
    task = asyncio.create_task(manager.request([_questions()[0]]))

    [pending] = await _wait_for_pending(manager)
    manager.cancel(pending.request_id, reason="User declined to answer.")

    with pytest.raises(UserInteractionCancelledError, match="declined"):
        await task
    assert manager.list_pending() == []


@pytest.mark.asyncio
async def test_ask_user_tool_library_executes_async_only():
    manager = AskUserManager()
    library = ToolLibrary("agent", [AskUser(manager)])
    task = asyncio.create_task(
        library.aforward(
            [
                (
                    "call_1",
                    "ask_user",
                    {"questions": [_questions()[0]]},
                )
            ]
        )
    )
    await _wait_for_pending(manager)

    [pending] = manager.list_pending()
    manager.answer(
        pending.request_id,
        {"Which storage backend should we use?": "Redis"},
    )
    responses = await task

    response = responses.get_by_name("ask_user")
    assert response is not None
    assert response.error is None
    assert response.result["answers"] == {
        "Which storage backend should we use?": "Redis"
    }
    assert "Continue with the user's answers in mind" in response.result["message"]
    assert library.library["ask_user"].display_name == "Ask User"
    assert library.library["ask_user"].description == AskUser.description


def test_ask_user_sync_call_raises_clear_error():
    manager = AskUserManager()
    with pytest.raises(RuntimeError, match="async-only"):
        AskUser(manager)(_questions())


def test_ask_user_validates_question_shape():
    with pytest.raises(ValueError, match="1 to 4"):
        normalize_questions([])

    with pytest.raises(ValueError, match="2 to 4"):
        normalize_questions(
            [{"question": "Pick one?", "header": "Pick", "options": []}]
        )

    with pytest.raises(ValueError, match="unique"):
        normalize_questions([_questions()[0], _questions()[0]])
