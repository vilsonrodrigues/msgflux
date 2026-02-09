"""Tests for msgflux.nn.modules.team module."""

from typing import Any, Optional

import pytest

from msgflux.message import Message
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.team import DeliberativeTeam


class StubAgent(Agent):
    """Lightweight Agent test double for DeliberativeTeam tests."""

    def __init__(
        self,
        *args,
        name: Optional[str] = None,
        responses: Optional[list[Any]] = None,
        fail_on_calls: Optional[set[int]] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        if args:
            if len(args) > 1:
                raise TypeError("StubAgent accepts at most one positional argument")
            name = args[0]
        if not isinstance(name, str) or name == "":
            raise ValueError("StubAgent requires a non-empty `name`")

        Module.__init__(self)
        self.set_name(name)
        self.set_description(description)
        self.kwargs = kwargs
        self.responses = list(responses or [])
        self.calls = 0
        self.messages: list[Any] = []
        self.fail_on_calls = set(fail_on_calls or set())

    def _next_response(self):
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise RuntimeError(f"forced failure for {self.name} on call {self.calls}")
        if self.responses:
            return self.responses.pop(0)
        return "APPROVE default vote"

    def forward(self, message=None, **kwargs):
        self.messages.append(message)
        response = self._next_response()
        if callable(response):
            return response(message, **kwargs)
        return response

    async def aforward(self, message=None, **kwargs):
        return self.forward(message=message, **kwargs)


def _build_team(
    *,
    agents: list[StubAgent],
    moderator: StubAgent,
    config: Optional[dict[str, Any]] = None,
    message_fields: Optional[dict[str, Any]] = None,
    response_mode: Optional[str] = None,
) -> DeliberativeTeam:
    return DeliberativeTeam(
        name="core_team",
        agents=agents,
        moderator=moderator,
        config=config,
        message_fields=message_fields,
        response_mode=response_mode,
    )


class TestTeamInitialization:
    def test_team_initialization(self):
        agents = [StubAgent("agent_a"), StubAgent("agent_b")]
        moderator = StubAgent("moderator")

        team = _build_team(agents=agents, moderator=moderator)

        assert team.name == "core_team"
        assert list(team.agents.keys()) == ["agent_a", "agent_b"]
        assert team.moderator.name == "moderator"
        assert team.config.strategy == "deliberative_v1"
        assert team.config.approval_threshold == 0.5

    def test_team_requires_at_least_two_agents(self):
        with pytest.raises(ValueError, match="at least 2 agents"):
            _build_team(
                agents=[StubAgent("agent_a")],
                moderator=StubAgent("moderator"),
            )

    def test_team_requires_agent_instances(self):
        with pytest.raises(TypeError, match="instances of `nn.Agent`"):
            DeliberativeTeam(
                name="core_team",
                agents=[StubAgent("agent_a"), object()],  # type: ignore[list-item]
                moderator=StubAgent("moderator"),
            )

    def test_team_requires_unique_agent_names(self):
        with pytest.raises(ValueError, match="Duplicate agent name"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_a")],
                moderator=StubAgent("moderator"),
            )

    def test_moderator_name_cannot_collide_with_agent(self):
        with pytest.raises(ValueError, match="cannot collide"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_b")],
                moderator=StubAgent("agent_a"),
            )

    def test_participants_must_be_subset_of_agents(self):
        with pytest.raises(ValueError, match="contains unknown agents"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_b")],
                moderator=StubAgent("moderator"),
                config={"participants": ["agent_a", "ghost"]},
            )

    def test_participants_cannot_contain_duplicates(self):
        with pytest.raises(ValueError, match="must not contain duplicate"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_b")],
                moderator=StubAgent("moderator"),
                config={"participants": ["agent_a", "agent_a"]},
            )

    def test_round_moderation_flags_must_be_bool(self):
        with pytest.raises(TypeError, match="enable_round_moderation"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_b")],
                moderator=StubAgent("moderator"),
                config={"enable_round_moderation": "yes"},  # type: ignore[arg-type]
            )

        with pytest.raises(TypeError, match="show_countdown"):
            _build_team(
                agents=[StubAgent("agent_a"), StubAgent("agent_b")],
                moderator=StubAgent("moderator"),
                config={"show_countdown": 1},  # type: ignore[arg-type]
            )


class TestTeamExecution:
    def test_team_completed_flow(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan ready",
                "APPROVE distribution ready",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE plan ready",
                "APPROVE distribution ready",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=["draft plan", "task distribution", "final synthesis"],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "completed"
        assert result["draft_plan"] == "draft plan"
        assert result["distribution"] == "task distribution"
        assert result["final_solution"] == "final synthesis"
        assert len(result["debate_turns"]) == 2
        assert len(result["plan_votes"]) == 2
        assert len(result["distribution_votes"]) == 2
        assert len(result["contributions"]) == 2

    def test_team_rejects_plan(self):
        agent_a = StubAgent(
            "agent_a",
            responses=["a debate", "REJECT missing details"],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=["b debate", "REJECT missing details"],
        )
        moderator = StubAgent("moderator", responses=["draft plan"])

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "rejected_plan"
        assert result["distribution"] is None
        assert result["final_solution"] is None

    def test_team_rejects_distribution(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE initial plan",
                "REJECT distribution not balanced",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE initial plan",
                "REJECT distribution not balanced",
            ],
        )
        moderator = StubAgent("moderator", responses=["draft plan", "distribution"])

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "rejected_distribution"
        assert result["distribution"] == "distribution"
        assert result["final_solution"] is None

    def test_team_max_revisions_for_plan(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "REJECT first plan weak",
                "APPROVE revised plan ok",
                "APPROVE distribution ok",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "REJECT first plan weak",
                "APPROVE revised plan ok",
                "APPROVE distribution ok",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=[
                "draft plan",
                "revised plan",
                "distribution",
                "final synthesis",
            ],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 1},
        )

        result = team("build a robust API")

        assert result["status"] == "completed"
        assert result["draft_plan"] == "revised plan"

    def test_team_participants_filter(self):
        active_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan",
                "APPROVE distribution",
                "a contribution",
            ],
        )
        inactive_b = StubAgent("agent_b")
        active_c = StubAgent(
            "agent_c",
            responses=[
                "c debate",
                "APPROVE plan",
                "APPROVE distribution",
                "c contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=["draft plan", "distribution", "final synthesis"],
        )

        team = _build_team(
            agents=[active_a, inactive_b, active_c],
            moderator=moderator,
            config={"participants": ["agent_a", "agent_c"], "max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "completed"
        assert result["participants"] == ["agent_a", "agent_c"]
        assert inactive_b.calls == 0

    def test_team_response_mode_with_message(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan",
                "APPROVE distribution",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE plan",
                "APPROVE distribution",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=["draft plan", "distribution", "final synthesis"],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 0},
            message_fields={"task_inputs": "content"},
            response_mode="outputs.team",
        )

        message = Message(content="build a robust API")
        response = team(message)

        assert isinstance(response, Message)
        assert response.outputs["team"]["status"] == "completed"

    def test_team_round_moderation_notes(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan",
                "APPROVE distribution",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE plan",
                "APPROVE distribution",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=[
                "round summary",
                "draft plan",
                "distribution",
                "final synthesis",
            ],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"enable_round_moderation": True, "max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "completed"
        assert len(result["moderator_round_notes"]) == 1
        assert result["moderator_round_notes"][0]["round"] == 1
        assert result["moderator_round_notes"][0]["note"] == "round summary"
        assert "Rounds remaining after this turn: 0" in agent_a.messages[0]

    def test_team_countdown_can_be_disabled(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan",
                "APPROVE distribution",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE plan",
                "APPROVE distribution",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=["draft plan", "distribution", "final synthesis"],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"show_countdown": False, "max_revisions": 0},
        )

        result = team("build a robust API")

        assert result["status"] == "completed"
        assert "Rounds remaining after this turn:" not in agent_a.messages[0]

    @pytest.mark.asyncio
    async def test_team_aforward(self):
        agent_a = StubAgent(
            "agent_a",
            responses=[
                "a debate",
                "APPROVE plan",
                "APPROVE distribution",
                "a contribution",
            ],
        )
        agent_b = StubAgent(
            "agent_b",
            responses=[
                "b debate",
                "APPROVE plan",
                "APPROVE distribution",
                "b contribution",
            ],
        )
        moderator = StubAgent(
            "moderator",
            responses=["draft plan", "distribution", "final synthesis"],
        )

        team = _build_team(
            agents=[agent_a, agent_b],
            moderator=moderator,
            config={"max_revisions": 0},
        )

        result = await team.acall("build a robust API")

        assert result["status"] == "completed"
        assert result["final_solution"] == "final synthesis"


class VoteObject:
    def __init__(self, approved, rationale: str):
        self.approved = approved
        self.rationale = rationale


class TestTeamVoteParsing:
    @pytest.fixture
    def team(self):
        return _build_team(
            agents=[StubAgent("agent_a"), StubAgent("agent_b")],
            moderator=StubAgent("moderator"),
        )

    def test_parse_vote_bool(self, team):
        approved, rationale = team._parse_vote(True)
        assert approved is True
        assert rationale == "Boolean vote"

    def test_parse_vote_dict(self, team):
        approved, rationale = team._parse_vote(
            {"approved": False, "reason": "missing details"}
        )
        assert approved is False
        assert rationale == "missing details"

    def test_parse_vote_dict_string_value(self, team):
        approved, rationale = team._parse_vote(
            {"approved": "REJECT", "rationale": "insufficient detail"}
        )
        assert approved is False
        assert rationale == "insufficient detail"

    def test_parse_vote_object(self, team):
        approved, rationale = team._parse_vote(VoteObject(True, "looks good"))
        assert approved is True
        assert rationale == "looks good"

    def test_parse_vote_string_tokens(self, team):
        approved_yes, rationale_yes = team._parse_vote("APPROVE ready to proceed")
        approved_no, rationale_no = team._parse_vote("REJECT needs more evidence")

        assert approved_yes is True
        assert rationale_yes == "APPROVE ready to proceed"
        assert approved_no is False
        assert rationale_no == "REJECT needs more evidence"

    def test_parse_vote_unrecognized_string_defaults_to_reject(self, team):
        approved, rationale = team._parse_vote("maybe")
        assert approved is False
        assert rationale == "maybe"

    def test_parse_vote_none_defaults_to_reject(self, team):
        approved, rationale = team._parse_vote(None)
        assert approved is False
        assert rationale == "None"
