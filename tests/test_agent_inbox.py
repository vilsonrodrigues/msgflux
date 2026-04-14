"""Tests for msgflux.agent_inbox."""

from msgflux.agent_inbox import AgentInbox


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
    assert "source=task ref=task_123 status=completed" in captured.out
    assert "[assistant][notification_drain] 1 notification(s)" in captured.out


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
    assert "status=process" in captured.out
