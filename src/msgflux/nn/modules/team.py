import asyncio
from typing import Any, Dict, List, Mapping, Optional, Union

from msgflux.auto import AutoParams
from msgflux.dotdict import dotdict
from msgflux.message import Message
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.utils.console import cprint

_VOTE_TRUE_TOKEN = "APPROVE"  # noqa: S105
_VOTE_FALSE_TOKEN = "REJECT"  # noqa: S105


class Team(Module, metaclass=AutoParams):
    """Orchestrates a team of agents without a leader, using a moderator."""

    _autoparams_use_docstring_for = "description"
    _autoparams_use_classname_for = "name"

    _supported_strategies = {"deliberative_v1"}

    def __init__(
        self,
        name: str,
        agents: List[Agent],
        moderator: Agent,
        *,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        annotations: Optional[Mapping[str, type]] = None,
    ):
        if annotations is None:
            annotations = {"message": str, "return": dict}

        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.set_annotations(dict(annotations))
        self._set_agents(agents)
        self._set_moderator(moderator)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self._set_config(config)

    def forward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Message]:
        """Execute team orchestration."""
        inputs = self._prepare_task(message, **kwargs)
        result = self._run_strategy(**inputs)
        return self._define_response_mode(result, message)

    async def aforward(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Message]:
        """Async version of forward."""
        inputs = self._prepare_task(message, **kwargs)
        result = await self._arun_strategy(**inputs)
        return self._define_response_mode(result, message)

    def _run_strategy(
        self,
        task: Any,
        context: Any,
        vars: Any,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        strategy = self.config["strategy"]
        if strategy == "deliberative_v1":
            return self._run_strategy_deliberative_v1(
                task=task,
                context=context,
                vars=vars,
                model_preference=model_preference,
            )
        raise ValueError(
            f"Unsupported strategy `{strategy}`. "
            f"Supported strategies are: {self._supported_strategies}"
        )

    async def _arun_strategy(
        self,
        task: Any,
        context: Any,
        vars: Any,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        strategy = self.config["strategy"]
        if strategy == "deliberative_v1":
            return await self._arun_strategy_deliberative_v1(
                task=task,
                context=context,
                vars=vars,
                model_preference=model_preference,
            )
        raise ValueError(
            f"Unsupported strategy `{strategy}`. "
            f"Supported strategies are: {self._supported_strategies}"
        )

    def _run_strategy_deliberative_v1(
        self,
        *,
        task: Any,
        context: Any,
        vars: Any,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        agents = self._get_active_agents()
        result = self._init_result(task, agents)
        self._vlog(
            "Starting deliberative_v1 strategy with "
            f"{len(agents)} participants and {self.config['debate_rounds']} rounds"
        )

        debate_turns, moderator_round_notes = self._debate(
            agents=agents,
            task=task,
            context=context,
            model_preference=model_preference,
        )
        result["debate_turns"] = debate_turns
        result["moderator_round_notes"] = moderator_round_notes

        draft_plan = self._build_plan(
            task=task,
            context=context,
            debate_turns=debate_turns,
            vars=vars,
            model_preference=model_preference,
        )

        draft_plan, plan_votes, approved = self._vote_and_revise_loop(
            agents=agents,
            label="draft_plan",
            vote_fn=self._vote_plan,
            revise_fn=self._revise_plan,
            proposal=draft_plan,
            task=task,
            context=context,
            model_preference=model_preference,
        )
        result["draft_plan"] = draft_plan
        result["plan_votes"] = plan_votes
        if not approved:
            result["status"] = "rejected_plan"
            self._vlog("Flow finished with status=rejected_plan")
            return result

        result["hiring_note"] = self._maybe_hiring_discussion(
            task=task,
            context=context,
            draft_plan=draft_plan,
            model_preference=model_preference,
        )

        distribution = self._distribute_tasks(
            task=task,
            context=context,
            draft_plan=draft_plan,
            participants=result["participants"],
            model_preference=model_preference,
        )

        distribution, dist_votes, approved = self._vote_and_revise_loop(
            agents=agents,
            label="distribution",
            vote_fn=self._vote_distribution,
            revise_fn=self._revise_distribution,
            proposal=distribution,
            task=task,
            context=context,
            draft_plan=draft_plan,
            model_preference=model_preference,
        )
        result["distribution"] = distribution
        result["distribution_votes"] = dist_votes
        if not approved:
            result["status"] = "rejected_distribution"
            self._vlog("Flow finished with status=rejected_distribution")
            return result

        contributions = self._solve(
            agents=agents,
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            model_preference=model_preference,
        )
        result["contributions"] = contributions

        result["final_solution"] = self._synthesize_solution(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            contributions=contributions,
            vars=vars,
            model_preference=model_preference,
        )
        result["status"] = "completed"
        self._vlog("Flow finished with status=completed")
        return result

    async def _arun_strategy_deliberative_v1(
        self,
        *,
        task: Any,
        context: Any,
        vars: Any,
        model_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        agents = self._get_active_agents()
        result = self._init_result(task, agents)
        self._vlog(
            "Starting async deliberative_v1 strategy with "
            f"{len(agents)} participants and {self.config['debate_rounds']} rounds"
        )

        debate_turns, moderator_round_notes = await self._adebate(
            agents=agents,
            task=task,
            context=context,
            model_preference=model_preference,
        )
        result["debate_turns"] = debate_turns
        result["moderator_round_notes"] = moderator_round_notes

        draft_plan = await self._abuild_plan(
            task=task,
            context=context,
            debate_turns=debate_turns,
            vars=vars,
            model_preference=model_preference,
        )

        draft_plan, plan_votes, approved = await self._avote_and_revise_loop(
            agents=agents,
            label="draft_plan",
            vote_fn=self._avote_plan,
            revise_fn=self._arevise_plan,
            proposal=draft_plan,
            task=task,
            context=context,
            model_preference=model_preference,
        )
        result["draft_plan"] = draft_plan
        result["plan_votes"] = plan_votes
        if not approved:
            result["status"] = "rejected_plan"
            self._vlog("Async flow finished with status=rejected_plan")
            return result

        result["hiring_note"] = await self._amaybe_hiring_discussion(
            task=task,
            context=context,
            draft_plan=draft_plan,
            model_preference=model_preference,
        )

        distribution = await self._adistribute_tasks(
            task=task,
            context=context,
            draft_plan=draft_plan,
            participants=result["participants"],
            model_preference=model_preference,
        )

        distribution, dist_votes, approved = await self._avote_and_revise_loop(
            agents=agents,
            label="distribution",
            vote_fn=self._avote_distribution,
            revise_fn=self._arevise_distribution,
            proposal=distribution,
            task=task,
            context=context,
            draft_plan=draft_plan,
            model_preference=model_preference,
        )
        result["distribution"] = distribution
        result["distribution_votes"] = dist_votes
        if not approved:
            result["status"] = "rejected_distribution"
            self._vlog("Async flow finished with status=rejected_distribution")
            return result

        contributions = await self._asolve(
            agents=agents,
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            model_preference=model_preference,
        )
        result["contributions"] = contributions

        result["final_solution"] = await self._asynthesize_solution(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            contributions=contributions,
            vars=vars,
            model_preference=model_preference,
        )
        result["status"] = "completed"
        self._vlog("Async flow finished with status=completed")
        return result

    def _vote_and_revise_loop(
        self,
        *,
        agents: List[Agent],
        label: str,
        vote_fn: Any,
        revise_fn: Any,
        proposal: Any,
        **kwargs,
    ) -> tuple[Any, List[Dict[str, Any]], bool]:
        """Generic vote-revise loop. Returns (proposal, votes, approved)."""
        display = label.replace("_", " ")
        votes: List[Dict[str, Any]] = []
        for revision in range(self.config["max_revisions"] + 1):
            self._vlog(
                f"Collecting {display} votes "
                f"(attempt {revision + 1}/{self.config['max_revisions'] + 1})"
            )
            votes = vote_fn(agents=agents, **{label: proposal, **kwargs})
            approved_count, total_votes, ratio = self._count_approved_votes(votes)
            self._vlog(
                f"{display.capitalize()} votes: "
                f"{approved_count}/{total_votes} approved "
                f"(ratio={ratio:.2f}, threshold>{self.config['approval_threshold']})"
            )
            if self._is_approved(votes):
                return proposal, votes, True
            if revision < self.config["max_revisions"]:
                self._vlog(f"{display.capitalize()} rejected, requesting revision")
                proposal = revise_fn(**{label: proposal, "votes": votes, **kwargs})
                self._vlog(f"Revised {display}: {self._preview_content(proposal)}")
        return proposal, votes, False

    async def _avote_and_revise_loop(
        self,
        *,
        agents: List[Agent],
        label: str,
        vote_fn: Any,
        revise_fn: Any,
        proposal: Any,
        **kwargs,
    ) -> tuple[Any, List[Dict[str, Any]], bool]:
        """Async generic vote-revise loop. Returns (proposal, votes, approved)."""
        display = label.replace("_", " ")
        votes: List[Dict[str, Any]] = []
        for revision in range(self.config["max_revisions"] + 1):
            self._vlog(
                f"Collecting async {display} votes "
                f"(attempt {revision + 1}/{self.config['max_revisions'] + 1})"
            )
            votes = await vote_fn(agents=agents, **{label: proposal, **kwargs})
            approved_count, total_votes, ratio = self._count_approved_votes(votes)
            self._vlog(
                f"{display.capitalize()} votes: "
                f"{approved_count}/{total_votes} approved "
                f"(ratio={ratio:.2f}, threshold>{self.config['approval_threshold']})"
            )
            if self._is_approved(votes):
                return proposal, votes, True
            if revision < self.config["max_revisions"]:
                self._vlog(
                    f"{display.capitalize()} rejected, requesting async revision"
                )
                proposal = await revise_fn(
                    **{label: proposal, "votes": votes, **kwargs}
                )
                self._vlog(f"Revised {display}: {self._preview_content(proposal)}")
        return proposal, votes, False

    def _init_result(self, task: Any, agents: List[Agent]) -> Dict[str, Any]:
        return {
            "status": "rejected_plan",
            "task": task,
            "participants": [agent.name for agent in agents],
            "debate_turns": [],
            "moderator_round_notes": [],
            "draft_plan": None,
            "plan_votes": [],
            "distribution": None,
            "distribution_votes": [],
            "contributions": [],
            "final_solution": None,
            "hiring_note": None,
            "future_flags": self._future_flags_snapshot(),
        }

    def _prepare_task(
        self, message: Optional[Union[str, Mapping[str, Any], Message]], **kwargs
    ) -> Dict[str, Any]:
        task_override = kwargs.pop("task_inputs", None)
        context_override = kwargs.pop("context_inputs", None)
        vars_override = kwargs.pop("vars", None)
        model_preference = kwargs.pop("model_preference", None)

        if message is None:
            task = task_override if task_override is not None else (kwargs or None)
            context = context_override
            vars = vars_override
        elif isinstance(message, Message):
            task = self._extract_message_values(self.task_inputs, message)
            context = self._extract_message_values(self.context_inputs, message)
            vars = self._extract_message_values(self.vars, message)

            if task_override is not None:
                task = task_override
            if context_override is not None:
                context = context_override
            if vars_override is not None:
                vars = vars_override

            if model_preference is None:
                model_preference = self.get_model_preference_from_message(message)
            if kwargs:
                raise ValueError(
                    "Named task arguments are not supported when message is provided"
                )
        elif isinstance(message, (str, Mapping)):
            if kwargs:
                raise ValueError(
                    "Cannot mix direct `message` input with named task arguments"
                )
            task = task_override if task_override is not None else message
            context = context_override
            vars = vars_override
        else:
            raise ValueError(f"Unsupported message type: `{type(message)}`")

        return {
            "task": task,
            "context": context,
            "vars": vars,
            "model_preference": model_preference,
        }

    def _future_flags_snapshot(self) -> Dict[str, bool]:
        return {
            "enable_hiring_discussion": self.config["enable_hiring_discussion"],
            "enable_direct_communication": self.config["enable_direct_communication"],
            "enable_user_confirmation": self.config["enable_user_confirmation"],
        }

    def _vlog(self, message: str):
        if self.config.get("verbose", False):
            cprint(f"[{self.name}][team] {message}", bc="br4", ls="b")

    @staticmethod
    def _preview_content(content: Any, *, max_chars: int = 220) -> str:
        if content is None:
            return "None"
        text = str(content).replace("\n", " ")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _count_approved_votes(votes: List[Dict[str, Any]]) -> tuple[int, int, float]:
        total = len(votes)
        approved = sum(1 for vote in votes if vote["approved"])
        ratio = approved / total if total > 0 else 0.0
        return approved, total, ratio

    def _get_active_agents(self) -> List[Agent]:
        participants = self.config.get("participants")
        if participants is None:
            return list(self.agents.values())
        return [self.agents[name] for name in participants]

    def _build_debate_prompt(
        self,
        *,
        task: Any,
        context: Any,
        debate_turns: List[Dict[str, Any]],
        round_index: int,
        rounds_remaining: int,
        agent_name: str,
        moderator_note: Optional[Any] = None,
    ) -> str:
        history = debate_turns[-5:] if debate_turns else []
        countdown_line = ""
        if self.config["show_countdown"]:
            countdown_line = f"Rounds remaining after this turn: {rounds_remaining}\n"
        moderator_line = ""
        if moderator_note is not None:
            moderator_line = f"Moderator guidance: {moderator_note}\n"
        return (
            "You are participating in a collaborative team discussion.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Round: {round_index}\n"
            f"{countdown_line}"
            f"Current agent: {agent_name}\n"
            f"{moderator_line}"
            f"Recent debate turns: {history}\n"
            "Return your analysis of problems, possible solutions "
            "and practical notes.\n"
            "Use concise sections: Problems, Solutions, Risks, Open Questions.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_plan_prompt(
        self,
        *,
        task: Any,
        context: Any,
        debate_turns: List[Dict[str, Any]],
        vars: Any,
    ) -> str:
        return (
            "You are the team moderator.\n"
            "Synthesize a practical plan from the team debate.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Debate turns: {debate_turns}\n"
            f"Vars: {vars}\n"
            "Return a concise plan with problems, possible solutions and "
            "execution steps.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_plan_vote_prompt(
        self, *, task: Any, context: Any, draft_plan: Any, agent_name: str
    ) -> str:
        return (
            f"You are `{agent_name}` reviewing a team plan.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Plan: {draft_plan}\n"
            "Vote using first token as APPROVE or REJECT, then add rationale."
        )

    def _build_plan_revision_prompt(
        self, *, task: Any, context: Any, draft_plan: Any, votes: List[Dict[str, Any]]
    ) -> str:
        return (
            "You are the moderator revising a rejected plan.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Current plan: {draft_plan}\n"
            f"Votes: {votes}\n"
            "Return a revised plan addressing the rejection rationale.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_distribution_prompt(
        self, *, task: Any, context: Any, draft_plan: Any, participants: List[str]
    ) -> str:
        return (
            "You are the moderator assigning tasks.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            f"Participants: {participants}\n"
            "Return a clear task distribution per participant.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_distribution_vote_prompt(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        agent_name: str,
    ) -> str:
        return (
            f"You are `{agent_name}` reviewing the task distribution.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            f"Distribution: {distribution}\n"
            "Vote using first token as APPROVE or REJECT, then add rationale."
        )

    def _build_distribution_revision_prompt(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        votes: List[Dict[str, Any]],
    ) -> str:
        return (
            "You are the moderator revising a rejected task distribution.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            f"Current distribution: {distribution}\n"
            f"Votes: {votes}\n"
            "Return a revised distribution.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_contribution_prompt(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        agent_name: str,
    ) -> str:
        return (
            f"You are `{agent_name}` executing your assigned part.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            f"Distribution: {distribution}\n"
            "Return your concrete contribution to the final solution.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_final_solution_prompt(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        contributions: List[Dict[str, Any]],
        vars: Any,
    ) -> str:
        return (
            "You are the moderator synthesizing the final team output.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            f"Distribution: {distribution}\n"
            f"Contributions: {contributions}\n"
            f"Vars: {vars}\n"
            "Return the final integrated solution.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_hiring_discussion_prompt(
        self, *, task: Any, context: Any, draft_plan: Any
    ) -> str:
        return (
            "You are the moderator. Discuss if the team needs new members in future.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Approved plan: {draft_plan}\n"
            "Do not hire anyone now. Return only a note.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _build_round_moderation_prompt(
        self,
        *,
        task: Any,
        context: Any,
        round_index: int,
        rounds_remaining: int,
        round_turns: List[Dict[str, Any]],
        previous_note: Optional[Any],
    ) -> str:
        countdown_line = ""
        if self.config["show_countdown"]:
            countdown_line = f"Rounds remaining: {rounds_remaining}\n"
        previous_note_line = ""
        if previous_note is not None:
            previous_note_line = f"Previous moderation note: {previous_note}\n"

        guidance = (
            "Summarize consensus, disagreements, and concrete focus points "
            "for the next round."
            if rounds_remaining > 0
            else "Summarize final debate alignment and unresolved questions "
            "to guide final planning."
        )

        return (
            "You are the moderator coordinating the team debate.\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Round completed: {round_index}\n"
            f"{countdown_line}"
            f"{previous_note_line}"
            f"Round turns: {round_turns}\n"
            f"{guidance}\n"
            "Keep it concise and actionable.\n"
            "Do not start your answer with APPROVE or REJECT."
        )

    def _debate(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        model_preference: Optional[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        turns: List[Dict[str, Any]] = []
        moderator_round_notes: List[Dict[str, Any]] = []
        previous_note: Optional[Any] = None
        for round_index in range(1, self.config["debate_rounds"] + 1):
            rounds_remaining = self.config["debate_rounds"] - round_index
            round_turns: List[Dict[str, Any]] = []
            for agent in agents:
                prompt = self._build_debate_prompt(
                    task=task,
                    context=context,
                    debate_turns=turns,
                    round_index=round_index,
                    rounds_remaining=rounds_remaining,
                    agent_name=agent.name,
                    moderator_note=previous_note,
                )
                content, error = self._call_member(
                    member=agent,
                    prompt=prompt,
                    model_preference=model_preference,
                )
                turn = {
                    "round": round_index,
                    "agent": agent.name,
                    "content": content,
                    "error": error,
                }
                turns.append(turn)
                round_turns.append(turn)

            if self.config["enable_round_moderation"]:
                note = self._moderate_round(
                    task=task,
                    context=context,
                    round_index=round_index,
                    rounds_remaining=rounds_remaining,
                    round_turns=round_turns,
                    previous_note=previous_note,
                    model_preference=model_preference,
                )
                moderator_round_notes.append({"round": round_index, "note": note})
                previous_note = note
                self._vlog(
                    f"Round {round_index} moderation note: "
                    f"{self._preview_content(note)}"
                )
        return turns, moderator_round_notes

    async def _adebate(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        model_preference: Optional[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        turns: List[Dict[str, Any]] = []
        moderator_round_notes: List[Dict[str, Any]] = []
        previous_note: Optional[Any] = None
        for round_index in range(1, self.config["debate_rounds"] + 1):
            rounds_remaining = self.config["debate_rounds"] - round_index
            calls = [
                {
                    "agent_name": agent.name,
                    "member": agent,
                    "prompt": self._build_debate_prompt(
                        task=task,
                        context=context,
                        debate_turns=turns,
                        round_index=round_index,
                        rounds_remaining=rounds_remaining,
                        agent_name=agent.name,
                        moderator_note=previous_note,
                    ),
                }
                for agent in agents
            ]
            results = await self._acall_member_batch(
                calls=calls,
                model_preference=model_preference,
            )
            round_turns = [
                {
                    "round": round_index,
                    "agent": call["agent_name"],
                    "content": content,
                    "error": error,
                }
                for call, (content, error) in zip(calls, results)
            ]
            turns.extend(round_turns)

            if self.config["enable_round_moderation"]:
                note = await self._amoderate_round(
                    task=task,
                    context=context,
                    round_index=round_index,
                    rounds_remaining=rounds_remaining,
                    round_turns=round_turns,
                    previous_note=previous_note,
                    model_preference=model_preference,
                )
                moderator_round_notes.append({"round": round_index, "note": note})
                previous_note = note
                self._vlog(
                    f"Async round {round_index} moderation note: "
                    f"{self._preview_content(note)}"
                )
        return turns, moderator_round_notes

    def _moderate_round(
        self,
        *,
        task: Any,
        context: Any,
        round_index: int,
        rounds_remaining: int,
        round_turns: List[Dict[str, Any]],
        previous_note: Optional[Any],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_round_moderation_prompt(
            task=task,
            context=context,
            round_index=round_index,
            rounds_remaining=rounds_remaining,
            round_turns=round_turns,
            previous_note=previous_note,
        )
        note, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return note

    async def _amoderate_round(
        self,
        *,
        task: Any,
        context: Any,
        round_index: int,
        rounds_remaining: int,
        round_turns: List[Dict[str, Any]],
        previous_note: Optional[Any],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_round_moderation_prompt(
            task=task,
            context=context,
            round_index=round_index,
            rounds_remaining=rounds_remaining,
            round_turns=round_turns,
            previous_note=previous_note,
        )
        note, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return note

    def _build_plan(
        self,
        *,
        task: Any,
        context: Any,
        debate_turns: List[Dict[str, Any]],
        vars: Any,
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_plan_prompt(
            task=task,
            context=context,
            debate_turns=debate_turns,
            vars=vars,
        )
        plan, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return plan

    async def _abuild_plan(
        self,
        *,
        task: Any,
        context: Any,
        debate_turns: List[Dict[str, Any]],
        vars: Any,
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_plan_prompt(
            task=task,
            context=context,
            debate_turns=debate_turns,
            vars=vars,
        )
        plan, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return plan

    def _vote_plan(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        votes = []
        for agent in agents:
            prompt = self._build_plan_vote_prompt(
                task=task,
                context=context,
                draft_plan=draft_plan,
                agent_name=agent.name,
            )
            raw_vote, error = self._call_member(
                member=agent,
                prompt=prompt,
                model_preference=model_preference,
            )
            approved, rationale = self._parse_vote(raw_vote, error=error)
            votes.append(
                {
                    "agent": agent.name,
                    "approved": approved,
                    "rationale": rationale,
                }
            )
        return votes

    async def _avote_plan(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        calls = []
        for agent in agents:
            calls.append(
                {
                    "agent_name": agent.name,
                    "member": agent,
                    "prompt": self._build_plan_vote_prompt(
                        task=task,
                        context=context,
                        draft_plan=draft_plan,
                        agent_name=agent.name,
                    ),
                }
            )

        results = await self._acall_member_batch(
            calls=calls,
            model_preference=model_preference,
        )

        votes = []
        for call, (raw_vote, error) in zip(calls, results):
            approved, rationale = self._parse_vote(raw_vote, error=error)
            votes.append(
                {
                    "agent": call["agent_name"],
                    "approved": approved,
                    "rationale": rationale,
                }
            )
        return votes

    def _revise_plan(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        votes: List[Dict[str, Any]],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_plan_revision_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            votes=votes,
        )
        plan, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return plan

    async def _arevise_plan(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        votes: List[Dict[str, Any]],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_plan_revision_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            votes=votes,
        )
        plan, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return plan

    def _distribute_tasks(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        participants: List[str],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_distribution_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            participants=participants,
        )
        distribution, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return distribution

    async def _adistribute_tasks(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        participants: List[str],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_distribution_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            participants=participants,
        )
        distribution, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return distribution

    def _vote_distribution(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        votes = []
        for agent in agents:
            prompt = self._build_distribution_vote_prompt(
                task=task,
                context=context,
                draft_plan=draft_plan,
                distribution=distribution,
                agent_name=agent.name,
            )
            raw_vote, error = self._call_member(
                member=agent,
                prompt=prompt,
                model_preference=model_preference,
            )
            approved, rationale = self._parse_vote(raw_vote, error=error)
            votes.append(
                {
                    "agent": agent.name,
                    "approved": approved,
                    "rationale": rationale,
                }
            )
        return votes

    async def _avote_distribution(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        calls = []
        for agent in agents:
            calls.append(
                {
                    "agent_name": agent.name,
                    "member": agent,
                    "prompt": self._build_distribution_vote_prompt(
                        task=task,
                        context=context,
                        draft_plan=draft_plan,
                        distribution=distribution,
                        agent_name=agent.name,
                    ),
                }
            )

        results = await self._acall_member_batch(
            calls=calls,
            model_preference=model_preference,
        )

        votes = []
        for call, (raw_vote, error) in zip(calls, results):
            approved, rationale = self._parse_vote(raw_vote, error=error)
            votes.append(
                {
                    "agent": call["agent_name"],
                    "approved": approved,
                    "rationale": rationale,
                }
            )
        return votes

    def _revise_distribution(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        votes: List[Dict[str, Any]],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_distribution_revision_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            votes=votes,
        )
        revised_distribution, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return revised_distribution

    async def _arevise_distribution(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        votes: List[Dict[str, Any]],
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_distribution_revision_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            votes=votes,
        )
        revised_distribution, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return revised_distribution

    def _solve(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        contributions = []
        for agent in agents:
            prompt = self._build_contribution_prompt(
                task=task,
                context=context,
                draft_plan=draft_plan,
                distribution=distribution,
                agent_name=agent.name,
            )
            content, error = self._call_member(
                member=agent,
                prompt=prompt,
                model_preference=model_preference,
            )
            contributions.append(
                {
                    "agent": agent.name,
                    "content": content,
                    "error": error,
                }
            )
        return contributions

    async def _asolve(
        self,
        *,
        agents: List[Agent],
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        model_preference: Optional[str],
    ) -> List[Dict[str, Any]]:
        calls = []
        for agent in agents:
            calls.append(
                {
                    "agent_name": agent.name,
                    "member": agent,
                    "prompt": self._build_contribution_prompt(
                        task=task,
                        context=context,
                        draft_plan=draft_plan,
                        distribution=distribution,
                        agent_name=agent.name,
                    ),
                }
            )

        results = await self._acall_member_batch(
            calls=calls,
            model_preference=model_preference,
        )

        contributions = []
        for call, (content, error) in zip(calls, results):
            contributions.append(
                {
                    "agent": call["agent_name"],
                    "content": content,
                    "error": error,
                }
            )
        return contributions

    def _synthesize_solution(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        contributions: List[Dict[str, Any]],
        vars: Any,
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_final_solution_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            contributions=contributions,
            vars=vars,
        )
        final_solution, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return final_solution

    async def _asynthesize_solution(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        distribution: Any,
        contributions: List[Dict[str, Any]],
        vars: Any,
        model_preference: Optional[str],
    ) -> Any:
        prompt = self._build_final_solution_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
            distribution=distribution,
            contributions=contributions,
            vars=vars,
        )
        final_solution, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return final_solution

    def _maybe_hiring_discussion(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        model_preference: Optional[str],
    ) -> Optional[Any]:
        if not self.config["enable_hiring_discussion"]:
            return None
        prompt = self._build_hiring_discussion_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
        )
        note, _error = self._call_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return note

    async def _amaybe_hiring_discussion(
        self,
        *,
        task: Any,
        context: Any,
        draft_plan: Any,
        model_preference: Optional[str],
    ) -> Optional[Any]:
        if not self.config["enable_hiring_discussion"]:
            return None
        prompt = self._build_hiring_discussion_prompt(
            task=task,
            context=context,
            draft_plan=draft_plan,
        )
        note, _error = await self._acall_member(
            member=self.moderator,
            prompt=prompt,
            model_preference=model_preference,
        )
        return note

    def _call_member(
        self, *, member: Agent, prompt: str, model_preference: Optional[str]
    ) -> tuple[Any, Optional[str]]:
        kwargs = {}
        if model_preference is not None:
            kwargs["model_preference"] = model_preference
        self._vlog(
            "Calling member "
            f"`{member.name}` with model_preference={model_preference or 'default'}"
        )
        try:
            response = member(prompt, **kwargs)
        except TypeError as e:
            if kwargs and "unexpected keyword argument" in str(e):
                try:
                    response = member(prompt)
                except Exception as fallback_error:
                    self._vlog(f"Member `{member.name}` failed: {fallback_error}")
                    return None, str(fallback_error)
            else:
                self._vlog(f"Member `{member.name}` failed: {e}")
                return None, str(e)
        except Exception as e:
            self._vlog(f"Member `{member.name}` failed: {e}")
            return None, str(e)
        normalized = self._normalize_member_response(response)
        self._vlog(
            f"Member `{member.name}` response: {self._preview_content(normalized)}"
        )
        return normalized, None

    async def _acall_member(
        self, *, member: Agent, prompt: str, model_preference: Optional[str]
    ) -> tuple[Any, Optional[str]]:
        kwargs = {}
        if model_preference is not None:
            kwargs["model_preference"] = model_preference
        self._vlog(
            "Calling member async "
            f"`{member.name}` with model_preference={model_preference or 'default'}"
        )
        try:
            response = await member.acall(prompt, **kwargs)
        except TypeError as e:
            if kwargs and "unexpected keyword argument" in str(e):
                try:
                    response = await member.acall(prompt)
                except Exception as fallback_error:
                    self._vlog(
                        f"Member `{member.name}` failed (async): {fallback_error}"
                    )
                    return None, str(fallback_error)
            else:
                self._vlog(f"Member `{member.name}` failed (async): {e}")
                return None, str(e)
        except Exception as e:
            self._vlog(f"Member `{member.name}` failed (async): {e}")
            return None, str(e)
        normalized = self._normalize_member_response(response)
        self._vlog(
            f"Member `{member.name}` response (async): "
            f"{self._preview_content(normalized)}"
        )
        return normalized, None

    async def _acall_member_batch(
        self,
        *,
        calls: List[Dict[str, Any]],
        model_preference: Optional[str],
    ) -> List[tuple[Any, Optional[str]]]:
        tasks = [
            self._acall_member(
                member=call["member"],
                prompt=call["prompt"],
                model_preference=model_preference,
            )
            for call in calls
        ]
        if not tasks:
            return []
        return await asyncio.gather(*tasks)

    def _normalize_member_response(self, response: Any) -> Any:
        if isinstance(response, Message):
            embedded_response = response.get_response()
            if embedded_response is not None:
                return embedded_response
        return response

    def _is_approved(self, votes: List[Dict[str, Any]]) -> bool:
        if not votes:
            return False
        approved = sum(1 for vote in votes if vote["approved"])
        ratio = approved / len(votes)
        return ratio > self.config["approval_threshold"]

    def _parse_vote(
        self, raw_vote: Any, *, error: Optional[str] = None
    ) -> tuple[bool, str]:
        if error is not None:
            return False, f"Error while collecting vote: {error}"

        if isinstance(raw_vote, bool):
            return raw_vote, "Boolean vote"

        if isinstance(raw_vote, Mapping):
            approved = self._extract_approved_value(raw_vote)
            rationale = self._extract_rationale(raw_vote)
            coerced = self._coerce_approved_value(approved)
            if coerced is not None:
                return coerced, rationale or str(raw_vote)
            return False, rationale or str(raw_vote)

        approved = self._extract_approved_value(raw_vote)
        coerced = self._coerce_approved_value(approved)
        if coerced is not None:
            rationale = self._extract_rationale(raw_vote)
            return coerced, rationale or str(raw_vote)

        if isinstance(raw_vote, str):
            normalized = raw_vote.strip()
            coerced = self._coerce_approved_value(normalized)
            if coerced is not None:
                return coerced, normalized
            return False, normalized

        return False, str(raw_vote)

    def _coerce_approved_value(self, value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None

            first_token = normalized.split(maxsplit=1)[0].upper()
            first_token = first_token.rstrip(":;,.!?")

            if first_token == _VOTE_TRUE_TOKEN:
                return True
            if first_token == _VOTE_FALSE_TOKEN:
                return False
            if first_token in {"TRUE", "YES", "1"}:
                return True
            if first_token in {"FALSE", "NO", "0"}:
                return False

        return None

    def _extract_approved_value(self, vote: Any) -> Optional[Any]:
        for field_name in ("approved", "approve", "is_approved"):
            if isinstance(vote, Mapping) and field_name in vote:
                return vote[field_name]
            if hasattr(vote, field_name):
                return getattr(vote, field_name)
        return None

    def _extract_rationale(self, vote: Any) -> Optional[str]:
        for field_name in ("rationale", "reason", "justification"):
            if isinstance(vote, Mapping) and field_name in vote:
                rationale = vote[field_name]
                return str(rationale) if rationale is not None else None
            if hasattr(vote, field_name):
                rationale = getattr(vote, field_name)
                return str(rationale) if rationale is not None else None
        return None

    def _set_agents(self, agents: List[Agent]):
        if not isinstance(agents, list):
            raise TypeError(f"`agents` must be a list, given `{type(agents)}`")
        if len(agents) < 2:
            raise ValueError("`agents` must contain at least 2 agents")

        names = []
        agents_dict = {}
        for agent in agents:
            if not isinstance(agent, Agent):
                raise TypeError(
                    "All team members in `agents` must be instances of `nn.Agent`"
                )
            agent_name = getattr(agent, "name", None)
            if not isinstance(agent_name, str) or agent_name == "":
                raise ValueError("Each team agent must have a valid non-empty name")
            if agent_name in names:
                raise ValueError(f"Duplicate agent name detected: `{agent_name}`")
            names.append(agent_name)
            agents_dict[agent_name] = agent

        self.agents = ModuleDict(agents_dict)
        self.register_buffer("agent_names", names)

    def _set_moderator(self, moderator: Agent):
        if not isinstance(moderator, Agent):
            raise TypeError("`moderator` must be an instance of `nn.Agent`")

        moderator_name = getattr(moderator, "name", None)
        if not isinstance(moderator_name, str) or moderator_name == "":
            raise ValueError("`moderator` must have a valid non-empty name")
        if moderator_name in self.agent_names:
            raise ValueError("`moderator.name` cannot collide with team agent names")

        self.moderator = moderator

    _config_defaults: Dict[str, Any] = {
        "strategy": "deliberative_v1",
        "debate_rounds": 1,
        "approval_threshold": 0.5,
        "max_revisions": 1,
        "participants": None,
        "verbose": False,
        "enable_round_moderation": False,
        "show_countdown": True,
        "enable_hiring_discussion": False,
        "enable_direct_communication": False,
        "enable_user_confirmation": False,
    }

    _config_bool_keys = (
        "verbose",
        "enable_round_moderation",
        "show_countdown",
        "enable_hiring_discussion",
        "enable_direct_communication",
        "enable_user_confirmation",
    )

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        merged = self._merge_config(config, self._config_defaults)
        self._validate_strategy(merged)
        self._validate_debate_config(merged)
        self._validate_vote_config(merged)
        self._validate_participants(merged, self.agent_names)
        self._validate_boolean_flags(merged, self._config_bool_keys)
        self.register_buffer("config", dotdict(merged))

    @staticmethod
    def _merge_config(
        config: Optional[Dict[str, Any]], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")
        valid_keys = set(defaults)
        invalid_keys = set(config) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid config keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )
        merged = defaults.copy()
        merged.update(config)
        return merged

    def _validate_strategy(self, merged: Dict[str, Any]) -> None:
        strategy = merged["strategy"]
        if strategy not in self._supported_strategies:
            raise ValueError(
                f"Unsupported strategy `{strategy}`. "
                f"Supported strategies are: {self._supported_strategies}"
            )

    @staticmethod
    def _validate_debate_config(merged: Dict[str, Any]) -> None:
        if not isinstance(merged["debate_rounds"], int) or merged["debate_rounds"] < 1:
            raise ValueError("`debate_rounds` must be an integer >= 1")

    @staticmethod
    def _validate_vote_config(merged: Dict[str, Any]) -> None:
        threshold = merged["approval_threshold"]
        if not isinstance(threshold, (int, float)):
            raise TypeError("`approval_threshold` must be a number in [0, 1]")
        if not 0 <= threshold <= 1:
            raise ValueError("`approval_threshold` must be in [0, 1]")
        if not isinstance(merged["max_revisions"], int) or merged["max_revisions"] < 0:
            raise ValueError("`max_revisions` must be an integer >= 0")

    @staticmethod
    def _validate_participants(merged: Dict[str, Any], agent_names: List[str]) -> None:
        participants = merged["participants"]
        if participants is None:
            return
        if not isinstance(participants, list):
            raise TypeError("`participants` must be a list of agent names or None")
        if not participants:
            raise ValueError("`participants` must not be an empty list")
        if not all(isinstance(name, str) for name in participants):
            raise TypeError("`participants` must contain only strings")
        if len(set(participants)) != len(participants):
            raise ValueError("`participants` must not contain duplicate names")
        invalid = [name for name in participants if name not in agent_names]
        if invalid:
            raise ValueError(
                f"`participants` contains unknown agents: {invalid}. "
                f"Available agents are: {agent_names}"
            )

    @staticmethod
    def _validate_boolean_flags(
        merged: Dict[str, Any], bool_keys: tuple[str, ...]
    ) -> None:
        for key in bool_keys:
            if not isinstance(merged[key], bool):
                raise TypeError(f"`{key}` must be a bool")

    def _set_context_inputs(
        self, context_inputs: Optional[Union[str, List[str]]] = None
    ):
        if isinstance(context_inputs, (str, list)) or context_inputs is None:
            if isinstance(context_inputs, str) and context_inputs == "":
                raise ValueError("`context_inputs` requires a non-empty string")
            if isinstance(context_inputs, list) and not context_inputs:
                raise ValueError("`context_inputs` requires a non-empty list")
            self.register_buffer("context_inputs", context_inputs)
        else:
            raise TypeError(
                "`context_inputs` requires a string, list or None "
                f"given `{type(context_inputs)}`"
            )

    def _set_vars(self, vars: Optional[str] = None):
        if isinstance(vars, str) or vars is None:
            self.register_buffer("vars", vars)
        else:
            raise TypeError(f"`vars` requires a string or None given `{type(vars)}`")

    def _set_message_fields(self, message_fields: Optional[Dict[str, Any]] = None):
        valid_keys = {
            "task_inputs",
            "context_inputs",
            "vars",
            "model_preference",
        }

        if message_fields is None:
            self._set_task_inputs(None)
            self._set_context_inputs(None)
            self._set_vars(None)
            self._set_model_preference(None)
            return

        if not isinstance(message_fields, dict):
            raise TypeError(
                "`message_fields` must be a dict or None, "
                f"given `{type(message_fields)}`"
            )

        invalid_keys = set(message_fields.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid message_fields keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        self._set_task_inputs(message_fields.get("task_inputs"))
        self._set_context_inputs(message_fields.get("context_inputs"))
        self._set_vars(message_fields.get("vars"))
        self._set_model_preference(message_fields.get("model_preference"))
