"""End-to-end collaboration test with real OpenAI models.

Requires OPENAI_API_KEY in environment.

Run:
    OPENAI_API_KEY=... uv run pytest tests/integration/test_collaboration_e2e.py -v -s
"""

import os
from typing import List

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.collaboration import (
    Debate,
    RoundRobin,
    Team,
    TeamStrategy,
)
from msgflux.nn.modules.container import ModuleList
from msgflux.nn.modules.workspace import Workspace
from msgflux.utils.console import cprint

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

MODEL_ID = "openai/gpt-4.1-nano"


class ParallelThenFinalize(TeamStrategy):
    """Run specialists in parallel on round 0, then a finalizer on round 1.

    Args:
        specialists: Names of agents to run in parallel on round 0.
        finalizer: Name of the agent to run on round 1.
    """

    def __init__(self, specialists: List[str], finalizer: str):
        self._specialists = set(specialists)
        self._finalizer = finalizer

    def select_agents(
        self,
        round_num: int,
        workspace: Workspace,  # noqa: ARG002
        agents: ModuleList,
    ) -> List[Agent]:
        if round_num == 0:
            return [a for a in agents if a.name in self._specialists]
        return [a for a in agents if a.name == self._finalizer]


@pytest.fixture
def model():
    return mf.Model.chat_completion(MODEL_ID)


# ── Debate E2E ───────────────────────────────────────────────────────


@requires_openai
def test_debate_product_strategy(model):
    """Three domain experts debate a product decision, judge synthesizes.

    The task requires decomposition because each expert must evaluate
    from their own perspective (engineering, business, UX), then revise
    after seeing the others' arguments.
    """
    engineer = nn.Agent(
        "engineer",
        model,
        system_message=(
            "You are a senior backend engineer. "
            "Evaluate proposals from a technical feasibility, "
            "scalability, and maintenance perspective. "
            "Be concise (2-3 sentences)."
        ),
        config={"verbose": True},
    )

    business = nn.Agent(
        "business_analyst",
        model,
        system_message=(
            "You are a business analyst. "
            "Evaluate proposals from a market fit, cost, "
            "and revenue perspective. "
            "Be concise (2-3 sentences)."
        ),
        config={"verbose": True},
    )

    ux = nn.Agent(
        "ux_designer",
        model,
        system_message=(
            "You are a UX designer. "
            "Evaluate proposals from a user experience, "
            "accessibility, and adoption perspective. "
            "Be concise (2-3 sentences)."
        ),
        config={"verbose": True},
    )

    judge = nn.Agent(
        "product_lead",
        model,
        system_message=(
            "You are the product lead. "
            "Synthesize the team's perspectives into a clear "
            "go/no-go recommendation with 3 key reasons. "
            "Keep it under 100 words."
        ),
        config={"verbose": True},
    )

    debate = Debate(
        agents=[engineer, business, ux],
        judge=judge,
        rounds=2,
    )

    task = (
        "Should we build a real-time collaborative whiteboard feature "
        "into our B2B project management SaaS? "
        "Our current stack is Python/React, 500 enterprise customers, "
        "team of 8 engineers."
    )

    result = debate(task)

    assert isinstance(result, str)
    assert len(result) > 50
    cprint(f"\n{'=' * 60}", bc="g")
    cprint(f"DEBATE RESULT:\n{result}", bc="g")
    cprint(f"{'=' * 60}", bc="g")


# ── Team E2E ─────────────────────────────────────────────────────────


@requires_openai
def test_team_research_article(model):
    """A team of agents collaborates to produce a short article.

    The task decomposes into:
    1. Researcher gathers key points
    2. Writer drafts the article using research
    3. Editor reviews and produces the final version

    Each agent reads from and writes to the shared workspace.
    """
    ws = Workspace()

    researcher = nn.Agent(
        "researcher",
        model,
        system_message=(
            "You are a researcher. Read the goal from the workspace. "
            "Write 3-5 bullet points of key facts to the workspace "
            "using the write_artifact tool with key='research_notes'. "
            "Do NOT write a final_answer."
        ),
        tools=ws.get_tools(),
        config={"verbose": True},
    )

    writer = nn.Agent(
        "writer",
        model,
        system_message=(
            "You are a writer. Read research_notes from the workspace "
            "using read_artifact. Then write a short article "
            "(3-4 sentences) to the workspace using write_artifact "
            "with key='draft'. Do NOT write a final_answer."
        ),
        tools=ws.get_tools(),
        config={"verbose": True},
    )

    editor = nn.Agent(
        "editor",
        model,
        system_message=(
            "You are an editor. Read the 'draft' from the workspace "
            "using read_artifact. Polish it and write the final "
            "version using write_artifact with key='final_answer'. "
            "The final_answer should be a clean, publication-ready "
            "text."
        ),
        tools=ws.get_tools(),
        config={"verbose": True},
    )

    team = Team(
        agents=[researcher, writer, editor],
        strategy=RoundRobin(),
        max_rounds=6,
        workspace=ws,
    )

    result = team(
        "Write a short article about why Python became "
        "the most popular language for AI development."
    )

    cprint(f"\n{'=' * 60}", bc="g")
    cprint(f"WORKSPACE KEYS: {ws.list_keys()}", bc="g")
    cprint(f"HISTORY: {ws.history}", bc="g")
    cprint(f"{'=' * 60}", bc="g")
    cprint(f"FINAL RESULT:\n{result}", bc="g")
    cprint(f"{'=' * 60}", bc="g")

    assert isinstance(result, str)
    assert len(result) > 30


# ── Team E2E with Permissions + ask_agent ─────────────────────────


@requires_openai
def test_team_startup_pitch(model):
    """Three specialists work in parallel with scoped permissions and ask_agent.

    The task decomposes into:
    1. CFO writes financial projections (key: financials)
    2. Product Lead defines MVP and roadmap (key: product_plan),
       can ask CFO about budget
    3. GTM Lead defines go-to-market strategy (key: gtm_strategy),
       can ask Product Lead about features
    4. Pitch Writer reads everything and produces the final_answer

    Permissions enforce that each specialist can only write to their own key.
    """
    ws = Workspace(
        permissions={
            "cfo": ["financials"],
            "product_lead": ["product_plan"],
            "gtm_lead": ["gtm_strategy"],
            "pitch_writer": ["final_answer"],
        }
    )

    cfo = nn.Agent(
        "cfo",
        model,
        system_message=(
            "You are the CFO. Write concise financial projections "
            "for a $500k seed-funded AI customer service startup. "
            "Include burn rate, runway, and revenue targets. "
            "Write your output using write_artifact with key='financials'. "
            "Keep it under 100 words."
        ),
        config={"verbose": True},
    )

    product_lead = nn.Agent(
        "product_lead",
        model,
        system_message=(
            "You are the Product Lead. Define a concise MVP and 6-month "
            "roadmap for an AI customer service automation tool. "
            "If you need budget info, use ask_agent to ask the 'cfo'. "
            "Write your output using write_artifact with key='product_plan'. "
            "Keep it under 100 words."
        ),
        config={"verbose": True},
    )

    gtm_lead = nn.Agent(
        "gtm_lead",
        model,
        system_message=(
            "You are the GTM Lead. Define a go-to-market strategy for "
            "an AI customer service startup targeting SMBs. "
            "If you need feature info, use ask_agent to ask 'product_lead'. "
            "Write your output using write_artifact with key='gtm_strategy'. "
            "Keep it under 100 words."
        ),
        config={"verbose": True},
    )

    pitch_writer = nn.Agent(
        "pitch_writer",
        model,
        system_message=(
            "You are the Pitch Writer. Read all artifacts from the workspace "
            "(financials, product_plan, gtm_strategy) using read_artifact. "
            "Synthesize everything into a compelling 1-paragraph pitch "
            "and write it using write_artifact with key='final_answer'. "
            "Keep it under 150 words."
        ),
        config={"verbose": True},
    )

    strategy = ParallelThenFinalize(
        specialists=["cfo", "product_lead", "gtm_lead"],
        finalizer="pitch_writer",
    )

    team = Team(
        agents=[cfo, product_lead, gtm_lead, pitch_writer],
        strategy=strategy,
        max_rounds=3,
        workspace=ws,
        enable_ask=True,
    )

    result = team(
        "Create a 1-page pitch deck summary for an AI startup that "
        "automates customer service. Budget: $500k seed."
    )

    cprint(f"\n{'=' * 60}", bc="g")
    cprint(f"WORKSPACE KEYS: {ws.list_keys()}", bc="g")
    cprint(f"HISTORY: {ws.history}", bc="g")
    cprint(f"{'=' * 60}", bc="g")
    cprint(f"FINAL RESULT:\n{result}", bc="g")
    cprint(f"{'=' * 60}", bc="g")

    assert isinstance(result, str)
    assert len(result) > 30
