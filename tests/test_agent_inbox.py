"""Tests for msgflux.agent_inbox."""

from msgflux.agent_inbox import AgentControlMessage, AgentInbox


def test_agent_inbox_verbose_publish_and_drain_are_printed(capsys):
    inbox = AgentInbox(verbose=True, owner="assistant")

    inbox.publish(
        {
            "source": "task",
            "ref": "task_123",
            "status": "completed",
            "metadata": {"tool": "worker"},
        }
    )
    inbox.drain()

    captured = capsys.readouterr()
    assert "[assistant][notification_publish]" in captured.out
    assert (
        '<notification source="task" ref="task_123" status="completed">' in captured.out
    )
    assert "tool=worker" in captured.out
    assert "[assistant][notification_drain]" in captured.out
    assert "1 notification(s)" in captured.out
    assert "<system_note>" in captured.out
    assert "</system_note>" in captured.out


def test_agent_inbox_verbose_replace_is_printed(capsys):
    inbox = AgentInbox(verbose=True, owner="assistant")

    inbox.publish(
        {
            "source": "tool_status",
            "ref": "task_123",
            "status": "prepare",
            "dedupe_key": "progress:task_123",
        }
    )
    inbox.publish(
        {
            "source": "tool_status",
            "ref": "task_123",
            "status": "process",
            "dedupe_key": "progress:task_123",
        }
    )

    captured = capsys.readouterr()
    assert "[assistant][notification_replace]" in captured.out
    assert 'status="process"' in captured.out
    assert "dedupe_key=progress:task_123" in captured.out


def test_agent_inbox_accepts_control_messages():
    inbox = AgentInbox()

    inbox.publish(AgentControlMessage(command="pause", reason="operator request"))

    notification = inbox.drain()[0]
    assert notification.source == "control"
    assert notification.status == "pause"
    assert notification.hint == "operator request"


def test_agent_inbox_renders_incoming_user_message():
    inbox = AgentInbox()

    inbox.user_message("Please adjust the answer.", metadata={"user_id": "u1"})
    rendered = inbox.render(inbox.drain())

    assert rendered == {
        "role": "user",
        "content": (
            "<incoming_user_message>\n"
            "Please adjust the answer.\n"
            "user_id=u1\n"
            "</incoming_user_message>"
        ),
    }
