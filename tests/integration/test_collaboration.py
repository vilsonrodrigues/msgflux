"""Integration tests for multi-agent collaboration (Debate, Team, Workspace).

These tests verify end-to-end collaboration flows using mock agents that
simulate real agent behavior (reading/writing workspace, producing responses).

To run:
    uv run pytest tests/integration/test_collaboration.py -v
"""

import concurrent.futures

import pytest

from msgflux.nn.modules.collaboration import (
    AllParallel,
    Debate,
    RoundRobin,
    Team,
)
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.workspace import Workspace

# ── Mock agents that simulate real behavior ──────────────────────────


class EchoAgent(Module):
    """Agent that echoes input with its name prefixed."""

    def __init__(self, name: str):
        super().__init__()
        self.set_name(name)

    def forward(self, message=None, **_kwargs):
        return f"[{self.name}] {message}"

    async def aforward(self, message=None, **_kwargs):
        return f"[{self.name}] {message}"


class WorkspaceAgent(Module):
    """Agent that reads workspace and writes a contribution."""

    def __init__(self, name: str, contribution_key: str, contribution_value: str):
        super().__init__()
        self.set_name(name)
        self._key = contribution_key
        self._value = contribution_value

    def forward(self, _message=None, **kwargs):
        ws = kwargs.get("vars", {}).get("workspace")
        if ws is not None:
            # Read goal
            goal = ws.get("goal", "unknown")
            # Write contribution
            ws.put(self._key, f"{self._value} (goal: {goal})", author=self.name)
        return self._value

    async def aforward(self, _message=None, **kwargs):
        return self.forward(_message, **kwargs)


class ConditionalFinalizerAgent(Module):
    """Agent that writes final_answer after seeing enough artifacts."""

    def __init__(self, name: str, required_keys: list, answer: str):
        super().__init__()
        self.set_name(name)
        self._required_keys = required_keys
        self._answer = answer

    def forward(self, _message=None, **kwargs):
        ws = kwargs.get("vars", {}).get("workspace")
        if ws is not None:
            existing = ws.list_keys()
            if all(k in existing for k in self._required_keys):
                ws.put("final_answer", self._answer, author=self.name)
                return self._answer
        return "waiting for dependencies"

    async def aforward(self, _message=None, **kwargs):
        return self.forward(_message, **kwargs)


# ── Workspace integration ────────────────────────────────────────────


class TestWorkspaceIntegration:
    """End-to-end workspace tests."""

    def test_multiple_agents_share_workspace(self):
        """Multiple agents can read and write to the same workspace."""
        ws = Workspace()
        ws.put("goal", "build something")

        agent1 = WorkspaceAgent("researcher", "research", "data collected")
        agent2 = WorkspaceAgent("writer", "draft", "article written")

        agent1("task", vars={"workspace": ws})
        agent2("task", vars={"workspace": ws})

        assert ws.get("research") == "data collected (goal: build something)"
        assert ws.get("draft") == "article written (goal: build something)"
        assert len(ws.history) == 3  # goal + research + draft

    def test_workspace_tools_end_to_end(self):
        """Tools write/read/list work correctly through the full flow."""
        ws = Workspace()
        tools = ws.get_tools()
        write_fn, read_fn, list_fn = tools

        # Agent 1 writes
        vars1 = {"workspace": ws, "_agent_name": "planner"}
        write_fn(key="plan", content="Step 1: Research", vars=vars1)

        # Agent 2 reads and writes
        plan = read_fn(key="plan", vars={"workspace": ws})
        assert plan == "Step 1: Research"
        vars2 = {"workspace": ws, "_agent_name": "executor"}
        write_fn(key="execution", content=f"Executed: {plan}", vars=vars2)

        # Agent 3 lists all
        keys = list_fn(vars={"workspace": ws})
        assert "plan" in keys
        assert "execution" in keys

        # Verify audit trail
        assert len(ws.history) == 2
        assert ws.history[0]["author"] == "planner"
        assert ws.history[1]["author"] == "executor"

    def test_workspace_snapshot_reflects_all_changes(self):
        """Snapshot contains all artifacts at any point."""
        ws = Workspace()
        ws.put("a", "1")
        snap1 = ws.snapshot()
        assert "a" in snap1

        ws.put("b", "2")
        snap2 = ws.snapshot()
        assert "a" in snap2
        assert "b" in snap2

        # Overwrite
        ws.put("a", "updated")
        snap3 = ws.snapshot()
        assert "updated" in snap3


# ── Debate integration ───────────────────────────────────────────────


class TestDebateIntegration:
    """End-to-end debate tests."""

    def test_debate_single_round_produces_judge_output(self):
        """Single-round debate: debaters respond, judge synthesizes."""
        debater1 = EchoAgent("optimist")
        debater2 = EchoAgent("pessimist")
        judge = EchoAgent("judge")

        debate = Debate(agents=[debater1, debater2], judge=judge, rounds=1)
        result = debate("Should we use AI?")

        # Judge should have received the topic and positions
        assert "[judge]" in result
        assert "Debate positions" in result

    def test_debate_multi_round_includes_revision(self):
        """Multi-round debate: debaters see each other's positions and revise."""
        debater1 = EchoAgent("alice")
        debater2 = EchoAgent("bob")
        judge = EchoAgent("judge")

        debate = Debate(agents=[debater1, debater2], judge=judge, rounds=2)
        result = debate("Is remote work better?")

        assert "[judge]" in result

    @pytest.mark.asyncio
    async def test_debate_async_end_to_end(self):
        """Async debate produces correct output."""
        debater1 = EchoAgent("agent_a")
        debater2 = EchoAgent("agent_b")
        judge = EchoAgent("judge")

        debate = Debate(agents=[debater1, debater2], judge=judge, rounds=1)
        result = await debate.acall("What is the meaning of life?")

        assert "[judge]" in result
        assert "positions" in result.lower()

    def test_debate_preserves_agent_names_in_positions(self):
        """Agent names appear in the judge's prompt."""
        debater1 = EchoAgent("expert_a")
        debater2 = EchoAgent("expert_b")
        judge = EchoAgent("judge")

        debate = Debate(agents=[debater1, debater2], judge=judge, rounds=1)
        result = debate("Topic X")

        # Judge output contains the names
        assert "expert_a" in result
        assert "expert_b" in result


# ── Team integration ─────────────────────────────────────────────────


class TestTeamIntegration:
    """End-to-end team collaboration tests."""

    def test_team_round_robin_sequential_collaboration(self):
        """Agents take turns and build on each other's workspace artifacts."""
        researcher = WorkspaceAgent("researcher", "research_data", "findings")
        writer = WorkspaceAgent("writer", "draft", "article based on findings")
        finalizer = ConditionalFinalizerAgent(
            "finalizer",
            required_keys=["research_data", "draft"],
            answer="Final polished article",
        )

        team = Team(
            agents=[researcher, writer, finalizer],
            strategy=RoundRobin(),
            max_rounds=5,
        )

        result = team("Write a research article")

        assert result == "Final polished article"
        assert team.workspace.get("research_data") is not None
        assert team.workspace.get("draft") is not None
        assert team.workspace.get("final_answer") == "Final polished article"

    def test_team_stops_early_on_final_answer(self):
        """Team stops before max_rounds when final_answer appears."""
        instant_finalizer = ConditionalFinalizerAgent(
            "finalizer",
            required_keys=["goal"],  # goal is set automatically
            answer="done instantly",
        )

        team = Team(
            agents=[instant_finalizer],
            strategy=RoundRobin(),
            max_rounds=10,
        )

        result = team("Quick task")
        assert result == "done instantly"
        # Should have stopped after round 0
        assert len(team.workspace.history) == 2  # goal + final_answer

    def test_team_returns_snapshot_when_no_final_answer(self):
        """When max_rounds is exhausted, returns workspace snapshot."""
        worker = WorkspaceAgent("worker", "progress", "partial work")

        team = Team(
            agents=[worker],
            strategy=RoundRobin(),
            max_rounds=2,
        )

        result = team("Do something")

        # No final_answer, should get snapshot
        assert "progress" in result
        assert "goal" in result

    def test_team_with_custom_workspace(self):
        """Pre-populated workspace is accessible to agents."""
        ws = Workspace()
        ws.put("context", "Important background info")

        worker = WorkspaceAgent("worker", "output", "result using context")

        team = Team(agents=[worker], workspace=ws, max_rounds=1)
        team("Process the context")

        assert ws.get("output") is not None
        assert ws.get("context") == "Important background info"
        assert ws.get("goal") == "Process the context"

    @pytest.mark.asyncio
    async def test_team_async_round_robin(self):
        """Async team with RoundRobin works end-to-end."""
        agent1 = WorkspaceAgent("agent1", "data", "collected")
        finalizer = ConditionalFinalizerAgent(
            "finalizer",
            required_keys=["data"],
            answer="async done",
        )

        team = Team(
            agents=[agent1, finalizer],
            strategy=RoundRobin(),
            max_rounds=5,
        )

        result = await team.acall("Async task")
        assert result == "async done"

    @pytest.mark.asyncio
    async def test_team_async_all_parallel(self):
        """Async team with AllParallel runs agents concurrently."""
        agent1 = WorkspaceAgent("writer1", "part1", "section A")
        agent2 = WorkspaceAgent("writer2", "part2", "section B")
        finalizer = ConditionalFinalizerAgent(
            "editor",
            required_keys=["part1", "part2"],
            answer="Merged article",
        )

        team = Team(
            agents=[agent1, agent2, finalizer],
            strategy=AllParallel(),
            max_rounds=3,
        )

        result = await team.acall("Write article in parallel")

        # After round 0, all 3 agents ran; finalizer finds part1+part2 present
        # (they were written in same round), writes final_answer
        assert result == "Merged article"


# ── Cross-component integration ──────────────────────────────────────


class TestCrossComponentIntegration:
    """Tests verifying components work together correctly."""

    def test_debate_and_team_share_module_system(self):
        """Both containers properly register submodules."""
        agents = [EchoAgent("a"), EchoAgent("b")]

        debate = Debate(agents=agents, judge=EchoAgent("judge"))
        team = Team(agents=agents)

        # Both should have named_modules
        debate_modules = dict(debate.named_modules())
        team_modules = dict(team.named_modules())

        assert "debaters" in debate_modules
        assert "judge" in debate_modules
        assert "members" in team_modules

    def test_workspace_thread_safety_with_parallel_writes(self):
        """Workspace handles concurrent writes from scatter_gather."""
        ws = Workspace()

        def write_many(prefix, count):
            for i in range(count):
                ws.put(f"{prefix}_{i}", f"value_{i}", author=prefix)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(write_many, f"agent_{a}", 50) for a in range(4)]
            concurrent.futures.wait(futures)

        # 4 agents x 50 writes = 200 entries
        assert len(ws.list_keys()) == 200
        assert len(ws.history) == 200

    def test_nested_team_workspace_isolation(self):
        """Different Team instances have independent workspaces."""
        agent = WorkspaceAgent("worker", "output", "result")

        team1 = Team(agents=[agent], max_rounds=1)
        team2 = Team(agents=[agent], max_rounds=1)

        team1("Task 1")
        team2("Task 2")

        assert team1.workspace.get("goal") == "Task 1"
        assert team2.workspace.get("goal") == "Task 2"
        assert team1.workspace is not team2.workspace
