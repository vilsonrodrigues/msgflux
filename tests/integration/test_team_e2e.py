"""E2E tests for nn.Team (team.py) with real OpenAI models.

Requires OPENAI_API_KEY in environment.

Run:
    OPENAI_API_KEY=... uv run pytest tests/integration/test_team_e2e.py -v -s
"""

import os

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.nn.modules.team import Team
from msgflux.utils.console import cprint

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

MODEL_ID = "openai/gpt-4.1-nano"


@pytest.fixture
def model():
    return mf.Model.chat_completion(MODEL_ID)


def _build_agent(name, model, role_description):
    """Build a concise agent with instructions to vote APPROVE/REJECT."""
    system_message = (
        f"You are {role_description}. "
        "Keep all responses concise (2-3 sentences max). "
        "When asked to vote, start your response with APPROVE or REJECT "
        "followed by a brief rationale."
    )
    return nn.Agent(
        name,
        model,
        system_message=system_message,
        config={"verbose": True},
    )


# ── 1. Full deliberative flow (sync) ──────────────────────────────────


@requires_openai
def test_team_full_deliberative_flow(model):
    """Validate the complete deliberative flow end-to-end."""
    backend = _build_agent(
        "backend_engineer",
        model,
        "a senior backend engineer focused on scalability and performance",
    )
    pm = _build_agent(
        "product_manager",
        model,
        "a product manager focused on user needs and business value",
    )
    moderator = _build_agent(
        "tech_lead",
        model,
        "a tech lead moderating the team discussion and synthesizing plans",
    )

    team = Team(
        name="caching_team",
        agents=[backend, pm],
        moderator=moderator,
        config={
            "debate_rounds": 1,
            "max_revisions": 1,
            "approval_threshold": 0.5,
            "verbose": True,
        },
    )

    result = team(
        "Design a caching strategy for a high-traffic e-commerce API with 10k req/s"
    )

    cprint(f"\n{'=' * 60}", bc="g")
    cprint(f"STATUS: {result['status']}", bc="g")
    cprint(f"DEBATE TURNS: {len(result['debate_turns'])}", bc="g")
    cprint(f"PLAN VOTES: {result['plan_votes']}", bc="g")
    cprint(f"DRAFT PLAN (preview): {str(result['draft_plan'])[:200]}", bc="g")
    cprint(f"DISTRIBUTION (preview): {str(result['distribution'])[:200]}", bc="g")
    cprint(f"CONTRIBUTIONS: {len(result['contributions'])}", bc="g")
    cprint(
        f"FINAL SOLUTION (preview): {str(result['final_solution'])[:300]}",
        bc="g",
    )
    cprint(f"{'=' * 60}", bc="g")

    assert result["status"] == "completed"
    assert len(result["debate_turns"]) == 2  # 1 round x 2 agents
    assert isinstance(result["draft_plan"], str)
    assert len(result["draft_plan"]) > 0
    assert isinstance(result["distribution"], str)
    assert len(result["distribution"]) > 0
    assert len(result["contributions"]) == 2
    assert isinstance(result["final_solution"], str)
    assert len(result["final_solution"]) > 50

    for vote in result["plan_votes"]:
        assert "agent" in vote
        assert "approved" in vote
        assert "rationale" in vote


# ── 2. Full deliberative flow (async) ─────────────────────────────────


@requires_openai
@pytest.mark.asyncio
async def test_team_full_deliberative_flow_async(model):
    """Async variant via team.acall()."""
    backend = _build_agent(
        "backend_engineer",
        model,
        "a senior backend engineer focused on scalability and performance",
    )
    pm = _build_agent(
        "product_manager",
        model,
        "a product manager focused on user needs and business value",
    )
    moderator = _build_agent(
        "tech_lead",
        model,
        "a tech lead moderating the team discussion and synthesizing plans",
    )

    team = Team(
        name="caching_team_async",
        agents=[backend, pm],
        moderator=moderator,
        config={
            "debate_rounds": 1,
            "max_revisions": 1,
            "approval_threshold": 0.5,
            "verbose": True,
        },
    )

    result = await team.acall(
        "Design a caching strategy for a high-traffic e-commerce API with 10k req/s"
    )

    cprint(f"\n{'=' * 60}", bc="b")
    cprint(f"[ASYNC] STATUS: {result['status']}", bc="b")
    cprint(f"[ASYNC] DEBATE TURNS: {len(result['debate_turns'])}", bc="b")
    cprint(f"[ASYNC] CONTRIBUTIONS: {len(result['contributions'])}", bc="b")
    cprint(
        f"[ASYNC] FINAL SOLUTION (preview): {str(result['final_solution'])[:300]}",
        bc="b",
    )
    cprint(f"{'=' * 60}", bc="b")

    assert result["status"] == "completed"
    assert len(result["debate_turns"]) == 2
    assert isinstance(result["draft_plan"], str)
    assert len(result["draft_plan"]) > 0
    assert isinstance(result["distribution"], str)
    assert len(result["distribution"]) > 0
    assert len(result["contributions"]) == 2
    assert isinstance(result["final_solution"], str)
    assert len(result["final_solution"]) > 50

    for vote in result["plan_votes"]:
        assert "agent" in vote
        assert "approved" in vote
        assert "rationale" in vote


# ── 3. Round moderation ───────────────────────────────────────────────


@requires_openai
def test_team_with_round_moderation(model):
    """Validate round moderation produces notes."""
    frontend = _build_agent(
        "frontend_dev",
        model,
        "a frontend developer focused on UX and API consumption patterns",
    )
    backend = _build_agent(
        "backend_dev",
        model,
        "a backend developer focused on API design and data modeling",
    )
    moderator = _build_agent(
        "architect",
        model,
        "a software architect moderating the discussion and providing guidance",
    )

    team = Team(
        name="api_team",
        agents=[frontend, backend],
        moderator=moderator,
        config={
            "debate_rounds": 2,
            "max_revisions": 1,
            "approval_threshold": 0.5,
            "enable_round_moderation": True,
            "verbose": True,
        },
    )

    result = team("Should we migrate from REST to GraphQL for our mobile API?")

    cprint(f"\n{'=' * 60}", bc="br4")
    cprint(f"STATUS: {result['status']}", bc="br4")
    cprint(f"DEBATE TURNS: {len(result['debate_turns'])}", bc="br4")
    cprint(
        f"MODERATOR NOTES: {result['moderator_round_notes']}",
        bc="br4",
    )
    cprint(f"{'=' * 60}", bc="br4")

    assert result["status"] == "completed"
    assert len(result["moderator_round_notes"]) == 2

    for note_entry in result["moderator_round_notes"]:
        assert "round" in note_entry
        assert "note" in note_entry
        assert isinstance(note_entry["note"], str)
        assert len(note_entry["note"]) > 0

    assert len(result["debate_turns"]) == 4  # 2 rounds x 2 agents


# ── 4. Participants filter ────────────────────────────────────────────


@requires_openai
def test_team_participants_filter_real(model):
    """Validate that only agents listed in participants actually participate."""
    agent_a = _build_agent(
        "analyst",
        model,
        "a data analyst focused on metrics and KPIs",
    )
    agent_b = _build_agent(
        "designer",
        model,
        "a UX designer focused on user experience",
    )
    agent_c = _build_agent(
        "devops",
        model,
        "a DevOps engineer focused on infrastructure and deployment",
    )
    moderator = _build_agent(
        "manager",
        model,
        "a project manager moderating the team discussion",
    )

    team = Team(
        name="filtered_team",
        agents=[agent_a, agent_b, agent_c],
        moderator=moderator,
        config={
            "participants": ["analyst", "devops"],
            "debate_rounds": 1,
            "max_revisions": 1,
            "approval_threshold": 0.5,
            "verbose": True,
        },
    )

    result = team(
        "Define monitoring and alerting strategy for a microservices platform"
    )

    cprint(f"\n{'=' * 60}", bc="m")
    cprint(f"STATUS: {result['status']}", bc="m")
    cprint(f"PARTICIPANTS: {result['participants']}", bc="m")
    debate_agents = [t["agent"] for t in result["debate_turns"]]
    cprint(f"DEBATE AGENTS: {debate_agents}", bc="m")
    cprint(f"{'=' * 60}", bc="m")

    assert result["participants"] == ["analyst", "devops"]
    assert "designer" not in [t["agent"] for t in result["debate_turns"]]
    assert result["status"] == "completed"


# ── 5. Response quality ───────────────────────────────────────────────


@requires_openai
def test_team_response_quality(model):
    """Validate that the final solution references the task's domain keywords."""
    doctor_agent = _build_agent(
        "clinical_engineer",
        model,
        "a clinical systems engineer focused on healthcare IT integrations",
    )
    alert_agent = _build_agent(
        "alerts_specialist",
        model,
        "a real-time alerts specialist focused on notification delivery",
    )
    moderator = _build_agent(
        "health_it_lead",
        model,
        "a health IT project lead synthesizing team plans",
    )

    team = Team(
        name="healthcare_team",
        agents=[doctor_agent, alert_agent],
        moderator=moderator,
        config={
            "debate_rounds": 1,
            "max_revisions": 1,
            "approval_threshold": 0.5,
            "verbose": True,
        },
    )

    result = team(
        "Design a notification system for a healthcare app that alerts "
        "doctors about critical patient vitals"
    )

    cprint(f"\n{'=' * 60}", bc="c")
    cprint(f"STATUS: {result['status']}", bc="c")
    cprint(f"DRAFT PLAN:\n{result['draft_plan']}", bc="c")
    cprint(f"FINAL SOLUTION:\n{result['final_solution']}", bc="c")
    cprint(f"{'=' * 60}", bc="c")

    assert result["status"] == "completed"

    solution_lower = str(result["final_solution"]).lower()
    relevant_keywords = [
        "notification",
        "alert",
        "doctor",
        "patient",
        "vital",
        "health",
        "clinical",
        "monitor",
    ]
    matches = [kw for kw in relevant_keywords if kw in solution_lower]
    assert len(matches) >= 1, (
        f"Final solution should reference at least one domain keyword. "
        f"Found: {matches}. Solution: {result['final_solution'][:300]}"
    )

    assert isinstance(result["draft_plan"], str)
    assert len(result["draft_plan"]) > 100

    for contribution in result["contributions"]:
        assert contribution["content"] is not None
