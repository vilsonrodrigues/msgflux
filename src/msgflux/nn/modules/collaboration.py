import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.container import ModuleList
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.workspace import Workspace
from msgflux.tools.config import tool_config

__all__ = [
    "AllParallel",
    "Debate",
    "RoundRobin",
    "Team",
    "TeamStrategy",
]


class Debate(Module):
    """A container where N agents debate and a judge synthesizes.

    In each round, debaters produce or revise their positions. After all
    rounds, the judge agent synthesizes a final answer from the collected
    positions.

    Round 1 runs all debaters in parallel with the original task.
    Subsequent rounds present each debater with the other agents' positions
    so they can revise. The final judge call receives all positions from
    the last round.

    Args:
        agents: List of debater agents.
        judge: The judge agent that synthesizes the final answer.
        rounds: Number of debate rounds (default: 2).

    Examples:
        >>> import msgflux.nn as nn
        >>> debate = nn.Debate(
        ...     agents=[agent_a, agent_b, agent_c],
        ...     judge=judge_agent,
        ...     rounds=2,
        ... )
        >>> result = debate("Should we use microservices?")
    """

    def __init__(self, agents: List[Agent], judge: Agent, rounds: int = 2):
        super().__init__()
        if len(agents) < 2:
            raise ValueError("Debate requires at least 2 debater agents.")
        if rounds < 1:
            raise ValueError("Debate requires at least 1 round.")
        self.debaters = ModuleList(agents)
        self.judge = judge
        self.register_buffer("rounds", rounds)

    def forward(self, task: str) -> str:
        """Execute the debate synchronously.

        Args:
            task: The topic or question for the debate.

        Returns:
            The judge's synthesized answer.
        """
        from msgflux.nn.functional import bcast_gather, scatter_gather  # noqa: PLC0415

        n = len(self.debaters)

        # Round 1: all debaters respond independently
        positions = list(bcast_gather(list(self.debaters), task))

        # Rounds 2..N: each debater revises given others' positions
        for _ in range(1, self.rounds):
            revision_prompts = []
            for i in range(n):
                others = [
                    f"[{self.debaters[j].name}]: {positions[j]}"
                    for j in range(n)
                    if j != i
                ]
                prompt = (
                    f"Original task: {task}\n\n"
                    f"Other positions:\n" + "\n".join(others) + "\n\n"
                    f"Your previous position: {positions[i]}\n\n"
                    f"Revise your position considering the other perspectives."
                )
                revision_prompts.append(prompt)

            positions = list(
                scatter_gather(
                    list(self.debaters),
                    args_list=[(p,) for p in revision_prompts],
                )
            )

        # Judge synthesizes
        all_positions = "\n\n".join(
            f"[{self.debaters[i].name}]: {positions[i]}" for i in range(n)
        )
        judge_prompt = (
            f"Task: {task}\n\n"
            f"Debate positions:\n{all_positions}\n\n"
            f"Synthesize a final answer considering all perspectives."
        )
        return self.judge(judge_prompt)

    async def aforward(self, task: str) -> str:
        """Execute the debate asynchronously.

        Args:
            task: The topic or question for the debate.

        Returns:
            The judge's synthesized answer.
        """
        n = len(self.debaters)

        # Round 1: all debaters respond independently in parallel
        coros = [self.debaters[i].acall(task) for i in range(n)]
        positions = list(await asyncio.gather(*coros))

        # Rounds 2..N: each debater revises given others' positions
        for _ in range(1, self.rounds):
            revision_prompts = []
            for i in range(n):
                others = [
                    f"[{self.debaters[j].name}]: {positions[j]}"
                    for j in range(n)
                    if j != i
                ]
                prompt = (
                    f"Original task: {task}\n\n"
                    f"Other positions:\n" + "\n".join(others) + "\n\n"
                    f"Your previous position: {positions[i]}\n\n"
                    f"Revise your position considering the other perspectives."
                )
                revision_prompts.append(prompt)

            coros = [self.debaters[i].acall(revision_prompts[i]) for i in range(n)]
            positions = list(await asyncio.gather(*coros))

        # Judge synthesizes
        all_positions = "\n\n".join(
            f"[{self.debaters[i].name}]: {positions[i]}" for i in range(n)
        )
        judge_prompt = (
            f"Task: {task}\n\n"
            f"Debate positions:\n{all_positions}\n\n"
            f"Synthesize a final answer considering all perspectives."
        )
        return await self.judge.acall(judge_prompt)


class TeamStrategy(ABC):
    """Base class for team execution strategies.

    A strategy controls which agents are active each round and when
    the collaboration is complete.
    """

    @abstractmethod
    def select_agents(
        self,
        round_num: int,
        workspace: Workspace,
        agents: ModuleList,
    ) -> List[Agent]:
        """Select which agents should be active this round.

        Args:
            round_num: Current round number (0-indexed).
            workspace: The shared workspace.
            agents: All team members.

        Returns:
            List of agents to execute this round.
        """

    def is_complete(
        self,
        round_num: int,  # noqa: ARG002
        workspace: Workspace,
    ) -> bool:
        """Check if the collaboration should stop.

        Default: complete when ``workspace.get("final_answer")`` is not None.

        Args:
            round_num: Current round number (0-indexed).
            workspace: The shared workspace.

        Returns:
            True if collaboration is complete.
        """
        return workspace.get("final_answer") is not None


class RoundRobin(TeamStrategy):
    """Each agent takes a turn in order, one per round.

    The strategy cycles through agents sequentially. Collaboration stops
    when ``workspace.get("final_answer")`` exists.
    """

    def select_agents(
        self,
        round_num: int,
        workspace: Workspace,  # noqa: ARG002
        agents: ModuleList,
    ) -> List[Agent]:
        idx = round_num % len(agents)
        return [agents[idx]]


class AllParallel(TeamStrategy):
    """All agents execute every round in parallel.

    Collaboration stops when ``workspace.get("final_answer")`` exists.
    """

    def select_agents(
        self,
        round_num: int,  # noqa: ARG002
        workspace: Workspace,  # noqa: ARG002
        agents: ModuleList,
    ) -> List[Agent]:
        return list(agents)


class Team(Module):
    """A container where agents with roles collaborate via a shared workspace.

    Agents communicate by reading and writing artifacts in the workspace
    using workspace tools. The execution strategy controls which agents
    run each round and when collaboration is complete.

    Team automatically injects workspace tools into each agent's tool
    library so users don't need to pass ``ws.get_tools()`` manually.

    When ``enable_ask=True``, each agent also receives an ``ask_agent``
    tool that allows them to ask questions to other team members during
    execution.

    Args:
        agents: List of team member agents.
        strategy: Execution strategy (default: RoundRobin).
        max_rounds: Maximum number of collaboration rounds (default: 5).
        workspace: Optional pre-configured workspace. If None, a new one
            is created.
        enable_ask: If True, inject an ``ask_agent`` tool into each agent
            allowing inter-agent communication (default: False).

    Examples:
        >>> import msgflux.nn as nn
        >>> from msgflux.nn.modules.workspace import Workspace
        >>> ws = Workspace()
        >>> team = nn.Team(
        ...     agents=[researcher, writer, reviewer],
        ...     strategy=nn.modules.collaboration.RoundRobin(),
        ...     max_rounds=6,
        ...     workspace=ws,
        ... )
        >>> result = team("Write a blog post about AI safety")
    """

    def __init__(
        self,
        agents: List[Agent],
        strategy: Optional[TeamStrategy] = None,
        max_rounds: int = 5,
        workspace: Optional[Workspace] = None,
        *,
        enable_ask: bool = False,
    ):
        super().__init__()
        if len(agents) < 1:
            raise ValueError("Team requires at least 1 agent.")
        if max_rounds < 1:
            raise ValueError("Team requires at least 1 round.")
        self.members = ModuleList(agents)
        self.register_buffer("max_rounds", max_rounds)
        self.strategy = strategy or RoundRobin()
        self.workspace = workspace or Workspace()
        self._setup_agent_tools(enable_ask)

    def _setup_agent_tools(self, enable_ask: bool) -> None:  # noqa: FBT001
        """Inject workspace tools (and optionally ask_agent) into each agent.

        This modifies each agent's tool_library in place so users don't
        need to pass ``ws.get_tools()`` manually. Agents without a
        ``tool_library`` attribute are silently skipped.

        Args:
            enable_ask: Whether to also inject the ``ask_agent`` tool.
        """
        ws_tools = self.workspace.get_tools()
        members_by_name: Dict[str, Agent] = {a.name: a for a in self.members}
        for agent in self.members:
            if not hasattr(agent, "tool_library"):
                continue
            # Inject workspace tools
            existing_names = agent.tool_library.get_tool_names()
            for ws_tool in ws_tools:
                tool_name = getattr(ws_tool, "__name__", None) or getattr(
                    ws_tool, "name", None
                )
                if tool_name and tool_name not in existing_names:
                    agent.tool_library.add(ws_tool)
            # Inject ask_agent tool
            if enable_ask and len(members_by_name) > 1:
                ask_tool = self._make_ask_tool(agent, members_by_name)
                if "ask_agent" not in agent.tool_library.get_tool_names():
                    agent.tool_library.add(ask_tool)

    def _make_ask_tool(self, asking_agent: Agent, members: Dict[str, Agent]) -> Any:
        """Create an ask_agent tool closure for a specific agent.

        The returned tool allows the asking agent to send a question to
        another team member. The target agent responds based on its
        system_message and the current workspace state.

        Args:
            asking_agent: The agent that will use this tool.
            members: Mapping of agent names to agent instances.

        Returns:
            A tool function decorated with ``@tool_config(inject_vars=True)``.
        """
        workspace = self.workspace

        @tool_config(inject_vars=True)
        def ask_agent(agent_name: str, question: str, **kwargs) -> str:  # noqa: ARG001
            """Ask another team member a question.

            They will answer based on their expertise and the current
            workspace state.

            Args:
                agent_name: Name of the agent to ask.
                question: The question to ask.

            Returns:
                The agent's response, or an error message.
            """
            target = members.get(agent_name)
            if target is None:
                available = ", ".join(members.keys())
                return f"Agent '{agent_name}' not found. Available: {available}"
            if target.name == asking_agent.name:
                return "You cannot ask yourself."
            prompt = (
                f"A colleague ({asking_agent.name}) asks:\n{question}\n\n"
                f"Workspace context:\n{workspace.snapshot()}\n\n"
                f"Answer concisely based on your expertise."
            )
            return str(target(prompt))

        return ask_agent

    def _build_agent_prompt(self, task: str, agent: Agent) -> str:
        """Build the prompt for an agent including workspace context.

        Args:
            task: The original task.
            agent: The agent being prompted.

        Returns:
            The formatted prompt string.
        """
        snapshot = self.workspace.snapshot()
        return (
            f"Task: {task}\n\n"
            f"Workspace state:\n{snapshot}\n\n"
            f"You are '{agent.name}'. Use the workspace tools to read "
            f"existing artifacts and write your contributions. "
            f"When the task is fully complete, write the final result "
            f"to the 'final_answer' artifact."
        )

    def _build_vars(self, agent: Agent) -> dict:
        """Build the vars dict for an agent call.

        Args:
            agent: The agent being called.

        Returns:
            Dict with workspace reference and agent name.
        """
        return {"workspace": self.workspace, "_agent_name": agent.name}

    def forward(self, task: str) -> Union[str, Any]:
        """Execute the team collaboration synchronously.

        Args:
            task: The goal for the team.

        Returns:
            The final answer from the workspace, or a snapshot of all
            artifacts if no final_answer was produced.
        """
        from msgflux.nn.functional import scatter_gather  # noqa: PLC0415

        self.workspace.put("goal", task)

        for round_num in range(self.max_rounds):
            active = self.strategy.select_agents(
                round_num, self.workspace, self.members
            )

            if len(active) == 1:
                agent = active[0]
                prompt = self._build_agent_prompt(task, agent)
                agent(prompt, vars=self._build_vars(agent))
            else:
                prompts = [self._build_agent_prompt(task, a) for a in active]
                scatter_gather(
                    active,
                    args_list=[(p,) for p in prompts],
                    kwargs_list=[{"vars": self._build_vars(a)} for a in active],
                )

            if self.strategy.is_complete(round_num, self.workspace):
                break

        return self.workspace.get("final_answer") or self.workspace.snapshot()

    async def aforward(self, task: str) -> Union[str, Any]:
        """Execute the team collaboration asynchronously.

        Args:
            task: The goal for the team.

        Returns:
            The final answer from the workspace, or a snapshot of all
            artifacts if no final_answer was produced.
        """
        self.workspace.put("goal", task)

        for round_num in range(self.max_rounds):
            active = self.strategy.select_agents(
                round_num, self.workspace, self.members
            )

            coros = [
                a.acall(
                    self._build_agent_prompt(task, a),
                    vars=self._build_vars(a),
                )
                for a in active
            ]
            await asyncio.gather(*coros)

            if self.strategy.is_complete(round_num, self.workspace):
                break

        return self.workspace.get("final_answer") or self.workspace.snapshot()
