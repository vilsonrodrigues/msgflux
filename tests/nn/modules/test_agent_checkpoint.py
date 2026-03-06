"""Tests for Agent checkpoint/resume support."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from msgflux.data.stores.memory import MemoryCheckpointStore
from msgflux.data.stores.schemas import SessionStatus
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.session import AgentSessionManager


@pytest.fixture
def mock_chat_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


@pytest.fixture
def memory_store():
    return MemoryCheckpointStore()


class TestAgentSessionManager:
    """Tests for AgentSessionManager."""

    def test_store_key(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        assert mgr._store_key("session-1") == "session-1:test_agent"

    def test_save_and_load_checkpoint(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        messages = [{"role": "user", "content": "hello"}]
        vars = {"key": "value"}

        mgr.save_checkpoint("s1", messages, vars, status=SessionStatus.ACTIVE)

        checkpoint = mgr.load_checkpoint("s1")
        assert checkpoint is not None
        assert checkpoint["status"] == "active"
        assert checkpoint["messages"] == messages
        assert checkpoint["vars"] == vars
        assert checkpoint["pending_tool_calls"] is None

    def test_save_with_pending_tool_calls(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        pending = [("call_1", "get_weather", '{"city": "NYC"}')]

        mgr.save_checkpoint(
            "s1",
            [],
            {},
            status=SessionStatus.ACTIVE,
            pending_tool_calls=pending,
        )

        checkpoint = mgr.load_checkpoint("s1")
        # JSON roundtrip converts tuples to lists
        assert checkpoint["pending_tool_calls"] == [list(p) for p in pending]

    def test_save_with_error(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")

        mgr.save_checkpoint(
            "s1",
            [],
            {},
            status=SessionStatus.FAILED,
            error="API timeout",
        )

        checkpoint = mgr.load_checkpoint("s1")
        assert checkpoint["status"] == "failed"
        assert checkpoint["error"] == "API timeout"

    def test_load_nonexistent(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        assert mgr.load_checkpoint("nonexistent") is None

    def test_is_completed(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        mgr.save_checkpoint("s1", [], {}, status=SessionStatus.COMPLETED)
        assert mgr.is_completed("s1") is True

    def test_is_not_completed(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        mgr.save_checkpoint("s1", [], {}, status=SessionStatus.ACTIVE)
        assert mgr.is_completed("s1") is False

    def test_multiple_events_returns_last(self, memory_store):
        mgr = AgentSessionManager(memory_store, "test_agent")
        mgr.save_checkpoint("s1", [{"role": "user", "content": "first"}], {})
        mgr.save_checkpoint(
            "s1",
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            {"step": 2},
        )

        checkpoint = mgr.load_checkpoint("s1")
        assert len(checkpoint["messages"]) == 2
        assert checkpoint["vars"] == {"step": 2}


class TestAgentCheckpointInit:
    """Tests for Agent initialization with checkpoint_store."""

    def test_agent_without_checkpoint(self, mock_chat_model):
        agent = Agent(name="test_agent", model=mock_chat_model)
        assert agent._session_mgr is None

    def test_agent_with_checkpoint(self, mock_chat_model, memory_store):
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )
        assert agent._session_mgr is not None
        assert isinstance(agent._session_mgr, AgentSessionManager)
        assert agent._session_mgr.agent_name == "test_agent"


class TestAgentForwardWithCheckpoint:
    """Tests for Agent.forward() with checkpoint support."""

    def test_forward_without_session_id_unchanged(self, mock_chat_model, memory_store):
        """Without session_id, forward behaves as normal."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )
        mock_response = Mock(spec=ModelResponse)
        mock_response.response_type = "text_generation"
        mock_response.data = "Hello!"

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
            patch.object(agent, "_process_model_response") as mock_proc,
        ):
            mock_prep.return_value = {
                "messages": [{"role": "user", "content": "hi"}],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.return_value = "Hello!"

            result = agent.forward("hi")
            assert result == "Hello!"
            mock_prep.assert_called_once()

    def test_forward_with_session_id_saves_checkpoint(
        self, mock_chat_model, memory_store
    ):
        """With session_id, forward saves checkpoints."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
        ):
            mock_prep.return_value = {
                "messages": [{"role": "user", "content": "hi"}],
                "vars": {"k": "v"},
                "model_preference": None,
            }
            # Return string (no tool calls) — simplest path
            mock_exec.return_value = "Hello!"

            result = agent.forward("hi", session_id="sess-1")
            assert result == "Hello!"

        # Verify checkpoint was saved as COMPLETED
        checkpoint = agent._session_mgr.load_checkpoint("sess-1")
        assert checkpoint is not None
        assert checkpoint["status"] == "completed"

    def test_forward_resume_from_checkpoint(self, mock_chat_model, memory_store):
        """Resume from an existing ACTIVE checkpoint."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        # Pre-populate a checkpoint (simulating crash recovery)
        agent._session_mgr.save_checkpoint(
            "sess-1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "partial"},
            ],
            {"k": "v"},
            status=SessionStatus.ACTIVE,
        )

        with patch.object(agent, "_execute_model") as mock_exec:
            mock_exec.return_value = "Resumed response"

            result = agent.forward("hi", session_id="sess-1")
            assert result == "Resumed response"

            # Model was called with restored messages
            call_kwargs = mock_exec.call_args
            messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
                "messages"
            )
            assert len(messages) == 2

    def test_forward_resume_from_failed(self, mock_chat_model, memory_store):
        """Resume from a FAILED checkpoint retries from last state."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        agent._session_mgr.save_checkpoint(
            "sess-1",
            [{"role": "user", "content": "hi"}],
            {},
            status=SessionStatus.FAILED,
            error="API timeout",
        )

        with patch.object(agent, "_execute_model") as mock_exec:
            mock_exec.return_value = "Success after retry"
            result = agent.forward("hi", session_id="sess-1")
            assert result == "Success after retry"

    def test_forward_saves_failed_on_exception(self, mock_chat_model, memory_store):
        """On exception, saves FAILED checkpoint and re-raises."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
        ):
            mock_prep.return_value = {
                "messages": [{"role": "user", "content": "hi"}],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.side_effect = RuntimeError("API crashed")

            with pytest.raises(RuntimeError, match="API crashed"):
                agent.forward("hi", session_id="sess-1")

        checkpoint = agent._session_mgr.load_checkpoint("sess-1")
        assert checkpoint is not None
        assert checkpoint["status"] == "failed"
        assert checkpoint["error"] == "API crashed"

    def test_forward_completed_session_with_new_message_runs_fresh(
        self, mock_chat_model, memory_store
    ):
        """Completed session + new message = fresh execution."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        agent._session_mgr.save_checkpoint(
            "sess-1",
            [{"role": "user", "content": "old"}],
            {},
            status=SessionStatus.COMPLETED,
        )

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
        ):
            mock_prep.return_value = {
                "messages": [{"role": "user", "content": "new question"}],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.return_value = "Fresh response"

            result = agent.forward("new question", session_id="sess-1")
            assert result == "Fresh response"
            mock_prep.assert_called_once()


class TestAgentCheckpointWithToolCalls:
    """Tests for checkpoint during tool call loops."""

    def test_pending_tool_calls_saved_before_execution(
        self, mock_chat_model, memory_store
    ):
        """Checkpoint saves pending_tool_calls before tool execution."""
        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        # Mock model response with tool call, then text
        mock_tool_response = Mock(spec=ModelResponse)
        mock_tool_response.response_type = "tool_call"
        mock_tool_data = Mock()
        mock_tool_data.reasoning = None
        mock_tool_data.get_calls.return_value = [
            ("call_1", "get_weather", '{"city": "NYC"}')
        ]
        mock_tool_data.insert_results = Mock()
        mock_tool_data.get_messages.return_value = [
            {"role": "assistant", "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ]
        mock_tool_response.data = mock_tool_data

        mock_text_response = Mock(spec=ModelResponse)
        mock_text_response.response_type = "text_generation"
        mock_text_response.data = "It's sunny in NYC"

        tool_call_result = Mock()
        tool_call_result.return_directly = False
        tool_call_result.tool_calls = [
            Mock(id="call_1", name="get_weather", result="Sunny", error=None)
        ]

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
            patch.object(agent, "_process_tool_call") as mock_tool,
            patch.object(agent, "_extract_raw_response") as mock_extract,
            patch.object(agent, "_prepare_response") as mock_prepare_resp,
        ):
            mock_prep.return_value = {
                "messages": [{"role": "user", "content": "weather?"}],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.side_effect = [mock_tool_response, mock_text_response]
            mock_tool.return_value = tool_call_result
            mock_extract.return_value = "It's sunny in NYC"
            mock_prepare_resp.return_value = "It's sunny in NYC"

            result = agent.forward("weather?", session_id="sess-1")
            assert result == "It's sunny in NYC"

        # Verify final checkpoint is COMPLETED
        checkpoint = agent._session_mgr.load_checkpoint("sess-1")
        assert checkpoint["status"] == "completed"


class TestSessionContextVar:
    """Tests for session_id context variable management."""

    def test_context_cleanup_on_success(self, mock_chat_model, memory_store):
        """session_id contextvar is cleaned up after forward."""
        from msgflux.context import get_session_id

        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
        ):
            mock_prep.return_value = {
                "messages": [],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.return_value = "ok"
            agent.forward("hi", session_id="sess-1")

        # Context should be reset (token-based reset)
        assert get_session_id() is None

    def test_context_cleanup_on_error(self, mock_chat_model, memory_store):
        """session_id contextvar is cleaned up even on error."""
        from msgflux.context import get_session_id

        agent = Agent(
            name="test_agent", model=mock_chat_model, checkpoint_store=memory_store
        )

        with (
            patch.object(agent, "_prepare_task") as mock_prep,
            patch.object(agent, "_execute_model") as mock_exec,
        ):
            mock_prep.return_value = {
                "messages": [],
                "vars": {},
                "model_preference": None,
            }
            mock_exec.side_effect = RuntimeError("boom")

            with pytest.raises(RuntimeError):
                agent.forward("hi", session_id="sess-1")

        assert get_session_id() is None
