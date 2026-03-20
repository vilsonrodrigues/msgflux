"""COPRO — Coordinate Prompt Optimization via instruction search.

Generates candidate instructions per agent, evaluates them, and iteratively
refines using an LLM that receives the history of attempted instructions.

Reference: ``dspy/teleprompt/copro_optimizer.py`` (357 lines).
"""

import logging
from collections import defaultdict
from typing import Any, Callable, List, Optional

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.evaluate import Evaluate
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates used by the optimizer's internal LLM calls
# ---------------------------------------------------------------------------

_BASIC_GENERATE_INSTRUCTION = """\
You are an instruction optimizer for large language models.

I will give you the current system prompt fragments of an AI agent. \
Your task is to propose a single, improved system prompt that will lead \
the agent to perform the task well. Don't be afraid to be creative.

Current system prompt fragments:
{basic_instruction}

Respond with ONLY the improved system prompt, nothing else."""

_GENERATE_INSTRUCTION_GIVEN_ATTEMPTS = """\
You are an instruction optimizer for large language models.

I will give some system prompt instructions I've tried, along with their \
corresponding validation scores. The instructions are arranged in increasing \
order based on their scores, where higher scores indicate better quality.

Your task is to propose a new system prompt that will lead the agent to \
perform the task even better. Don't be afraid to be creative.

Previous attempts:
{attempted_instructions}

Respond with ONLY the improved system prompt, nothing else."""


class COPRO(Teleprompter):
    """Coordinate Prompt Optimization.

    Generates candidate system prompts per agent, evaluates each by running
    the full program, and iteratively refines using a prompt model that
    receives the history of attempted instructions with their scores.

    Args:
        metric: ``metric(example, prediction) -> float | bool``
        prompt_model: Model/Agent used to generate instruction candidates.
            Must be callable: ``prompt_model(message) -> str``.
        breadth: Number of candidate instructions generated per round.
        depth: Number of optimization rounds.
        init_temperature: Temperature for instruction generation.

    Example::

        optimizer = COPRO(
            metric=exact_match,
            prompt_model=generator_agent,
            breadth=5,
            depth=3,
        )
        compiled = optimizer.compile(student, trainset=examples)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        prompt_model: Any = None,
        breadth: int = 10,
        depth: int = 3,
        init_temperature: float = 1.4,
    ):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.prompt_model = prompt_model
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        eval_kwargs: Optional[dict] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Optimize system prompts per agent via breadth x depth search."""
        eval_kwargs = eval_kwargs or {}
        module = student.deepcopy()
        evaluate = Evaluate(devset=trainset, metric=self.metric, **eval_kwargs)

        agent_list = list(module.named_agents())
        if not agent_list:
            return module

        # Phase 1: generate initial candidates per agent
        latest_candidates, all_candidates = self._init_candidates(agent_list)

        module_clone = module.deepcopy()
        clone_agents = list(module_clone.named_agents())
        evaluated_candidates: dict = defaultdict(dict)
        total_calls = 0

        # Phase 2: depth iterations
        for d in range(self.depth):
            logger.info("Iteration Depth: %d/%d.", d + 1, self.depth)

            total_calls = self._eval_depth(
                d, agent_list, clone_agents, latest_candidates,
                all_candidates, evaluated_candidates, evaluate,
                module_clone, trainset, total_calls,
            )

            if d >= self.depth - 1:
                break

            latest_candidates = self._refine_candidates(
                agent_list, evaluated_candidates, all_candidates,
            )

        # Phase 3: select best program
        return self._select_best(
            agent_list, evaluated_candidates, total_calls,
        )

    def _init_candidates(
        self, agent_list: list,
    ) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
        """Generate initial candidates per agent."""
        latest: dict[int, list[str]] = {}
        all_cands: dict[int, list[str]] = {}

        for _name, agent in agent_list:
            basic_instruction = self._get_fragments(agent)
            candidates = self._generate_basic(basic_instruction)
            candidates.append(agent.get_system_prompt() or basic_instruction)
            latest[id(agent)] = candidates
            all_cands[id(agent)] = list(candidates)

        return latest, all_cands

    def _eval_depth(
        self, d, agent_list, clone_agents, latest_candidates,
        all_candidates, evaluated_candidates, evaluate,
        module_clone, trainset, total_calls,
    ) -> int:
        """Evaluate candidates at depth d."""
        for p_i, ((_name_old, agent_old), (_name_new, agent_new)) in enumerate(
            zip(agent_list, clone_agents)
        ):
            if len(agent_list) > 1:
                candidates_to_eval = all_candidates[id(agent_old)]
            else:
                candidates_to_eval = latest_candidates[id(agent_old)]

            for c_i, raw_instruction in enumerate(candidates_to_eval):
                instruction = raw_instruction.strip().strip('"')
                agent_new.optimized_system_prompt.data = instruction

                logger.info(
                    "Depth %d/%d, Candidate #%d/%d for Agent %d/%d.",
                    d + 1, self.depth, c_i + 1,
                    len(candidates_to_eval), p_i + 1, len(agent_list),
                )
                score = evaluate(module_clone, devset=trainset).score
                total_calls += 1

                key = (instruction,)
                prev = evaluated_candidates[id(agent_old)].get(key)
                if prev is None or prev["score"] < score:
                    evaluated_candidates[id(agent_old)][key] = {
                        "score": score,
                        "program": module_clone.deepcopy(),
                        "instruction": instruction,
                        "depth": d,
                    }

            best = max(
                evaluated_candidates[id(agent_old)].values(),
                key=lambda c: c["score"],
            )
            agent_new.optimized_system_prompt.data = best["instruction"]

        return total_calls

    def _refine_candidates(
        self, agent_list, evaluated_candidates, all_candidates,
    ) -> dict[int, list[str]]:
        """Generate next batch of candidates from previous attempts."""
        new_candidates: dict[int, list[str]] = {}
        for _name, agent in agent_list:
            attempts = self._build_attempts(evaluated_candidates[id(agent)])
            generated = self._generate_given_attempts(attempts)
            new_candidates[id(agent)] = generated
            all_candidates[id(agent)].extend(generated)
        return new_candidates

    def _select_best(
        self, agent_list, evaluated_candidates, total_calls,
    ) -> Module:
        """Select best program from all evaluated candidates."""
        all_scored = []
        for _name, agent in agent_list:
            all_scored.extend(evaluated_candidates[id(agent)].values())
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        all_scored = self._drop_duplicates(all_scored)

        best_program = all_scored[0]["program"]
        best_program.compile_(
            optimizer="COPRO",
            score=all_scored[0]["score"],
            total_calls=total_calls,
        )
        return best_program

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_fragments(self, agent: Any) -> str:
        """Extract the original prompt fragments as a text block."""
        parts = []
        if hasattr(agent, "system_message") and agent.system_message.data:
            parts.append(f"System message: {agent.system_message.data}")
        if hasattr(agent, "instructions") and agent.instructions.data:
            parts.append(f"Instructions: {agent.instructions.data}")
        if hasattr(agent, "expected_output") and agent.expected_output.data:
            parts.append(f"Expected output: {agent.expected_output.data}")
        if hasattr(agent, "examples") and agent.examples.data:
            parts.append(f"Examples: {agent.examples.data}")
        return "\n".join(parts) if parts else "No instructions provided."

    def _generate_basic(self, basic_instruction: str) -> list[str]:
        """Generate initial candidate instructions."""
        if self.prompt_model is None:
            return []

        prompt = _BASIC_GENERATE_INSTRUCTION.format(
            basic_instruction=basic_instruction
        )
        candidates = []
        for _ in range(self.breadth - 1):
            try:
                result = self.prompt_model(prompt)
                text = result if isinstance(result, str) else str(result)
                candidates.append(text.strip())
            except Exception:
                logger.warning("Failed to generate candidate", exc_info=True)
        return candidates

    def _build_attempts(self, evaluated: dict) -> str:
        """Format previous attempts for the refinement prompt."""
        items = sorted(evaluated.values(), key=lambda x: x["score"])
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"Attempt #{i}: {item['instruction']}")
            lines.append(f"Score #{i}: {item['score']:.4f}")
        return "\n".join(lines)

    def _generate_given_attempts(self, attempts_text: str) -> list[str]:
        """Generate new candidates given previous attempts."""
        if self.prompt_model is None:
            return []

        prompt = _GENERATE_INSTRUCTION_GIVEN_ATTEMPTS.format(
            attempted_instructions=attempts_text
        )
        candidates = []
        for _ in range(self.breadth):
            try:
                result = self.prompt_model(prompt)
                text = result if isinstance(result, str) else str(result)
                candidates.append(text.strip())
            except Exception:
                logger.warning("Failed to generate candidate", exc_info=True)
        return candidates

    @staticmethod
    def _drop_duplicates(candidates: list[dict]) -> list[dict]:
        """Remove duplicate candidates with the same instruction and score."""
        seen: set = set()
        unique = []
        for c in candidates:
            key = (c.get("instruction", ""), c.get("score", 0))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique
