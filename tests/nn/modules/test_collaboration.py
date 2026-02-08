"""Tests for msgflux.nn.modules.collaboration and workspace modules."""

from unittest.mock import patch  # noqa: I001

import pytest

from msgflux.nn.modules.collaboration import (
    AllParallel,
    Debate,
    RoundRobin,
    Team,
)
from msgflux.nn.modules.container import ModuleList
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.modules.workspace import Workspace


# ── Helpers ──────────────────────────────────────────────────────────


class MockAgent(Module):
    """Minimal agent-like module for testing."""

    def __init__(self, name: str, response: str = "ok"):
        super().__init__()
        self.set_name(name)
        self._response = response

    def forward(self, _message=None, **_kwargs):
        return self._response

    async def aforward(self, _message=None, **_kwargs):
        return self._response


class WorkspaceWriterAgent(Module):
    """Agent that writes to the workspace via vars."""

    def __init__(self, name: str, key: str, value: str):
        super().__init__()
        self.set_name(name)
        self._key = key
        self._value = value

    def forward(self, _message=None, **kwargs):
        ws = kwargs.get("vars", {}).get("workspace")
        if ws is not None:
            ws.put(self._key, self._value, author=self.name)
        return self._value

    async def aforward(self, _message=None, **kwargs):
        return self.forward(_message, **kwargs)


class FinalAnswerAgent(Module):
    """Agent that writes a final_answer to workspace."""

    def __init__(self, name: str, answer: str = "done"):
        super().__init__()
        self.set_name(name)
        self._answer = answer

    def forward(self, _message=None, **kwargs):
        ws = kwargs.get("vars", {}).get("workspace")
        if ws is not None:
            ws.put("final_answer", self._answer, author=self.name)
        return self._answer

    async def aforward(self, _message=None, **kwargs):
        return self.forward(_message, **kwargs)


class MockAgentWithTools(Module):
    """Agent-like module with a tool_library for testing Team tool injection."""

    def __init__(self, name: str, response: str = "ok"):
        super().__init__()
        self.set_name(name)
        self._response = response
        self.tool_library = ToolLibrary(name, [])

    def forward(self, _message=None, **_kwargs):
        return self._response

    async def aforward(self, _message=None, **_kwargs):
        return self._response


# ── Workspace Tests ──────────────────────────────────────────────────


class TestWorkspace:
    """Test suite for Workspace."""

    def test_put_and_get(self):
        ws = Workspace()
        ws.put("key1", "value1")
        assert ws.get("key1") == "value1"

    def test_get_default(self):
        ws = Workspace()
        assert ws.get("missing") is None
        assert ws.get("missing", "default") == "default"

    def test_list_keys(self):
        ws = Workspace()
        ws.put("a", 1)
        ws.put("b", 2)
        keys = ws.list_keys()
        assert sorted(keys) == ["a", "b"]

    def test_list_keys_empty(self):
        ws = Workspace()
        assert ws.list_keys() == []

    def test_snapshot(self):
        ws = Workspace()
        ws.put("name", "test")
        snap = ws.snapshot()
        assert "name" in snap
        assert "test" in snap

    def test_snapshot_empty(self):
        ws = Workspace()
        snap = ws.snapshot()
        assert snap == "{}"

    def test_history(self):
        ws = Workspace()
        ws.put("k1", "v1", author="agent_a")
        ws.put("k2", "v2", author="agent_b")
        history = ws.history
        assert len(history) == 2
        assert history[0]["key"] == "k1"
        assert history[0]["author"] == "agent_a"
        assert history[1]["key"] == "k2"

    def test_overwrite_artifact(self):
        ws = Workspace()
        ws.put("key", "v1")
        ws.put("key", "v2")
        assert ws.get("key") == "v2"
        assert len(ws.history) == 2

    def test_get_tools(self):
        ws = Workspace()
        tools = ws.get_tools()
        assert len(tools) == 3
        assert all(callable(t) for t in tools)


class TestWorkspaceTools:
    """Test suite for workspace tool functions."""

    def test_write_artifact(self):
        ws = Workspace()
        tools = ws.get_tools()
        write_fn = tools[0]  # _write_artifact
        result = write_fn(
            key="draft",
            content="hello",
            vars={"workspace": ws, "_agent_name": "writer"},
        )
        assert "successfully" in result
        assert ws.get("draft") == "hello"

    def test_read_artifact(self):
        ws = Workspace()
        ws.put("draft", "hello")
        tools = ws.get_tools()
        read_fn = tools[1]  # _read_artifact
        result = read_fn(key="draft", vars={"workspace": ws})
        assert result == "hello"

    def test_read_artifact_not_found(self):
        ws = Workspace()
        tools = ws.get_tools()
        read_fn = tools[1]
        result = read_fn(key="missing", vars={"workspace": ws})
        assert "not found" in result

    def test_list_artifacts(self):
        ws = Workspace()
        ws.put("a", 1)
        ws.put("b", 2)
        tools = ws.get_tools()
        list_fn = tools[2]  # _list_artifacts
        result = list_fn(vars={"workspace": ws})
        assert "a" in result
        assert "b" in result

    def test_list_artifacts_empty(self):
        ws = Workspace()
        tools = ws.get_tools()
        list_fn = tools[2]
        result = list_fn(vars={"workspace": ws})
        assert "No artifacts" in result


# ── Debate Tests ─────────────────────────────────────────────────────


class TestDebate:
    """Test suite for Debate container."""

    def test_debate_init(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        judge = MockAgent("judge")
        debate = Debate(agents=agents, judge=judge, rounds=2)
        assert len(debate.debaters) == 2
        assert debate.rounds == 2

    def test_debate_requires_at_least_2_agents(self):
        with pytest.raises(ValueError, match="at least 2"):
            Debate(agents=[MockAgent("a1")], judge=MockAgent("judge"))

    def test_debate_requires_at_least_1_round(self):
        with pytest.raises(ValueError, match="at least 1"):
            Debate(
                agents=[MockAgent("a1"), MockAgent("a2")],
                judge=MockAgent("judge"),
                rounds=0,
            )

    def test_debate_submodule_registration(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        judge = MockAgent("judge")
        debate = Debate(agents=agents, judge=judge)
        modules = dict(debate.named_modules())
        assert "debaters" in modules
        assert "judge" in modules

    def test_debate_forward(self):
        """Test that debate produces output through all phases."""
        agents = [MockAgent("a1", "pos_a"), MockAgent("a2", "pos_b")]
        judge = MockAgent("judge", "synthesis")
        debate = Debate(agents=agents, judge=judge, rounds=1)

        with patch(
            "msgflux.nn.functional.bcast_gather",
            return_value=("pos_a", "pos_b"),
        ):
            result = debate("test topic")

        assert result == "synthesis"

    def test_debate_forward_multi_round(self):
        """Test multi-round debate."""
        agents = [MockAgent("a1", "pos_a"), MockAgent("a2", "pos_b")]
        judge = MockAgent("judge", "final")
        debate = Debate(agents=agents, judge=judge, rounds=2)

        with (
            patch(
                "msgflux.nn.functional.bcast_gather",
                return_value=("pos_a", "pos_b"),
            ),
            patch(
                "msgflux.nn.functional.scatter_gather",
                return_value=("revised_a", "revised_b"),
            ),
        ):
            result = debate("test topic")

        assert result == "final"

    @pytest.mark.asyncio
    async def test_debate_aforward(self):
        """Test async debate."""
        agents = [MockAgent("a1", "pos_a"), MockAgent("a2", "pos_b")]
        judge = MockAgent("judge", "async_synthesis")
        debate = Debate(agents=agents, judge=judge, rounds=1)

        result = await debate.acall("test topic")
        assert result == "async_synthesis"


# ── TeamStrategy Tests ───────────────────────────────────────────────


class TestRoundRobin:
    """Test suite for RoundRobin strategy."""

    def test_selects_one_agent_per_round(self):
        strategy = RoundRobin()
        ws = Workspace()
        agents = ModuleList([MockAgent("a1"), MockAgent("a2"), MockAgent("a3")])

        selected = strategy.select_agents(0, ws, agents)
        assert len(selected) == 1
        assert selected[0].name == "a1"

        selected = strategy.select_agents(1, ws, agents)
        assert selected[0].name == "a2"

        selected = strategy.select_agents(2, ws, agents)
        assert selected[0].name == "a3"

    def test_wraps_around(self):
        strategy = RoundRobin()
        ws = Workspace()
        agents = ModuleList([MockAgent("a1"), MockAgent("a2")])

        selected = strategy.select_agents(2, ws, agents)
        assert selected[0].name == "a1"

    def test_is_complete_with_final_answer(self):
        strategy = RoundRobin()
        ws = Workspace()
        assert strategy.is_complete(0, ws) is False

        ws.put("final_answer", "done")
        assert strategy.is_complete(0, ws) is True


class TestAllParallel:
    """Test suite for AllParallel strategy."""

    def test_selects_all_agents(self):
        strategy = AllParallel()
        ws = Workspace()
        agents = ModuleList([MockAgent("a1"), MockAgent("a2"), MockAgent("a3")])

        selected = strategy.select_agents(0, ws, agents)
        assert len(selected) == 3

    def test_is_complete_with_final_answer(self):
        strategy = AllParallel()
        ws = Workspace()
        assert strategy.is_complete(0, ws) is False

        ws.put("final_answer", "result")
        assert strategy.is_complete(0, ws) is True


# ── Team Tests ───────────────────────────────────────────────────────


class TestTeam:
    """Test suite for Team container."""

    def test_team_init(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        team = Team(agents=agents)
        assert len(team.members) == 2
        assert team.max_rounds == 5
        assert isinstance(team.strategy, RoundRobin)

    def test_team_requires_at_least_1_agent(self):
        with pytest.raises(ValueError, match="at least 1"):
            Team(agents=[])

    def test_team_requires_at_least_1_round(self):
        with pytest.raises(ValueError, match="at least 1"):
            Team(agents=[MockAgent("a1")], max_rounds=0)

    def test_team_submodule_registration(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        team = Team(agents=agents)
        modules = dict(team.named_modules())
        assert "members" in modules

    def test_team_with_custom_workspace(self):
        ws = Workspace()
        ws.put("existing", "data")
        team = Team(agents=[MockAgent("a1")], workspace=ws)
        assert team.workspace is ws
        assert team.workspace.get("existing") == "data"

    def test_team_forward_with_final_answer(self):
        """Test that Team stops when final_answer is produced."""
        finalizer = FinalAnswerAgent("finalizer", "the answer")
        team = Team(
            agents=[finalizer],
            strategy=RoundRobin(),
            max_rounds=5,
        )

        result = team("solve this")
        assert result == "the answer"
        assert team.workspace.get("goal") == "solve this"

    def test_team_forward_exhausts_rounds(self):
        """Test that Team returns snapshot when rounds are exhausted."""
        writer = WorkspaceWriterAgent("writer", "notes", "some notes")
        team = Team(
            agents=[writer],
            strategy=RoundRobin(),
            max_rounds=2,
        )

        result = team("write something")
        # No final_answer, should return snapshot
        assert "notes" in result

    def test_team_forward_with_all_parallel(self):
        """Test Team with AllParallel strategy."""
        agent1 = FinalAnswerAgent("agent1", "parallel result")
        agent2 = MockAgent("agent2", "helper")
        team = Team(
            agents=[agent1, agent2],
            strategy=AllParallel(),
            max_rounds=3,
        )

        with patch(
            "msgflux.nn.functional.scatter_gather",
        ) as mock_sg:

            def side_effect(agents, args_list, kwargs_list):
                results = []
                for agent, args, kwargs in zip(agents, args_list, kwargs_list):
                    results.append(agent(*args, **kwargs))
                return tuple(results)

            mock_sg.side_effect = side_effect
            result = team("parallel task")

        assert result == "parallel result"

    @pytest.mark.asyncio
    async def test_team_aforward_with_final_answer(self):
        """Test async Team with final_answer."""
        finalizer = FinalAnswerAgent("finalizer", "async answer")
        team = Team(
            agents=[finalizer],
            strategy=RoundRobin(),
            max_rounds=5,
        )

        result = await team.acall("async task")
        assert result == "async answer"

    @pytest.mark.asyncio
    async def test_team_aforward_exhausts_rounds(self):
        """Test async Team returns snapshot when rounds exhausted."""
        writer = WorkspaceWriterAgent("writer", "data", "value")
        team = Team(
            agents=[writer],
            strategy=RoundRobin(),
            max_rounds=2,
        )

        result = await team.acall("do something")
        assert "data" in result


# ── Workspace Permissions Tests ─────────────────────────────────────


class TestWorkspacePermissions:
    """Test suite for Workspace permissions feature."""

    def test_permissions_allow_valid_write(self):
        ws = Workspace(permissions={"writer": ["draft"]})
        error = ws.put("draft", "hello", author="writer")
        assert error is None
        assert ws.get("draft") == "hello"

    def test_permissions_block_invalid_write(self):
        ws = Workspace(permissions={"writer": ["draft"]})
        error = ws.put("final_answer", "oops", author="writer")
        assert error is not None
        assert "cannot write" in error
        assert ws.get("final_answer") is None

    def test_permissions_allow_unlisted_agent(self):
        """Agents not in permissions dict can write to any key."""
        ws = Workspace(permissions={"writer": ["draft"]})
        error = ws.put("anything", "value", author="unknown_agent")
        assert error is None
        assert ws.get("anything") == "value"

    def test_permissions_allow_no_author(self):
        """Writes without author bypass permissions."""
        ws = Workspace(permissions={"writer": ["draft"]})
        error = ws.put("secret", "value")
        assert error is None
        assert ws.get("secret") == "value"

    def test_permissions_empty_allows_all(self):
        """Empty permissions dict allows all writes."""
        ws = Workspace(permissions={})
        error = ws.put("any_key", "value", author="any_agent")
        assert error is None

    def test_permissions_multiple_allowed_keys(self):
        ws = Workspace(permissions={"writer": ["draft", "notes"]})
        assert ws.put("draft", "d", author="writer") is None
        assert ws.put("notes", "n", author="writer") is None
        error = ws.put("final_answer", "f", author="writer")
        assert error is not None

    def test_permissions_tool_returns_error_message(self):
        ws = Workspace(permissions={"writer": ["draft"]})
        tools = ws.get_tools()
        write_fn = tools[0]  # _write_artifact
        result = write_fn(
            key="final_answer",
            content="oops",
            vars={"workspace": ws, "_agent_name": "writer"},
        )
        assert "cannot write" in result
        assert ws.get("final_answer") is None

    def test_permissions_tool_allows_valid_write(self):
        ws = Workspace(permissions={"writer": ["draft"]})
        tools = ws.get_tools()
        write_fn = tools[0]
        result = write_fn(
            key="draft",
            content="hello",
            vars={"workspace": ws, "_agent_name": "writer"},
        )
        assert "successfully" in result
        assert ws.get("draft") == "hello"


# ── Team Auto Tool Injection Tests ──────────────────────────────────


class TestTeamToolInjection:
    """Test suite for Team auto tool injection."""

    def test_team_injects_workspace_tools(self):
        """Team automatically injects workspace tools into agents."""
        agent = MockAgentWithTools("agent1")
        team = Team(agents=[agent], max_rounds=1)

        tool_names = agent.tool_library.get_tool_names()
        assert "_write_artifact" in tool_names
        assert "_read_artifact" in tool_names
        assert "_list_artifacts" in tool_names
        assert team.workspace is not None

    def test_team_does_not_duplicate_tools(self):
        """If agent already has workspace tools, don't add duplicates."""
        ws = Workspace()
        agent = MockAgentWithTools("agent1")
        for tool in ws.get_tools():
            agent.tool_library.add(tool)

        original_count = len(agent.tool_library.get_tool_names())
        Team(agents=[agent], workspace=ws, max_rounds=1)
        assert len(agent.tool_library.get_tool_names()) == original_count

    def test_team_enable_ask_false_by_default(self):
        """ask_agent not injected when enable_ask is False."""
        agent1 = MockAgentWithTools("a1")
        agent2 = MockAgentWithTools("a2")
        Team(agents=[agent1, agent2], max_rounds=1)

        assert "ask_agent" not in agent1.tool_library.get_tool_names()

    def test_team_enable_ask_injects_tool(self):
        """ask_agent is injected when enable_ask=True."""
        agent1 = MockAgentWithTools("a1")
        agent2 = MockAgentWithTools("a2")
        Team(agents=[agent1, agent2], max_rounds=1, enable_ask=True)

        assert "ask_agent" in agent1.tool_library.get_tool_names()
        assert "ask_agent" in agent2.tool_library.get_tool_names()

    def test_team_enable_ask_not_injected_single_agent(self):
        """ask_agent not injected for a team with only 1 agent."""
        agent = MockAgentWithTools("solo")
        Team(agents=[agent], max_rounds=1, enable_ask=True)

        assert "ask_agent" not in agent.tool_library.get_tool_names()
