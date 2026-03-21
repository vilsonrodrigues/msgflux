"""COPRO - Coordinate Prompt Optimization via instruction search.

Generates candidate instructions per agent, evaluates them, and iteratively
refines using an LM that receives the history of attempted instructions.

Reference: ``dspy/teleprompt/copro_optimizer.py``.
"""

import copy
import logging
import statistics
from collections import defaultdict
from typing import Any, Callable, List, Optional, Sequence

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.evaluate import Evaluate
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)

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
    """Coordinate Prompt Optimization."""

    def __init__(
        self,
        metric: Callable,
        *,
        prompt_model: Any = None,
        breadth: int = 10,
        depth: int = 3,
        init_temperature: float = 1.4,
        track_stats: bool = False,
    ):
        if breadth <= 1:
            raise ValueError("Breadth must be greater than 1")
        self.metric = metric
        self.prompt_model = prompt_model
        self.breadth = breadth
        self.depth = depth
        self.init_temperature = init_temperature
        self.track_stats = track_stats
        self.prompt_model_total_calls = 0

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
        prompt_model = self._resolve_prompt_model(module)
        self.prompt_model_total_calls = 0

        agent_list = list(module.named_agents())
        if not agent_list:
            return module

        latest_candidates, all_candidates = self._init_candidates(
            agent_list, prompt_model
        )
        module_clone = module.deepcopy()
        clone_agents = list(module_clone.named_agents())
        evaluated_candidates: dict[int, dict[tuple[str], dict[str, Any]]] = (
            defaultdict(dict)
        )
        total_calls = 0
        results_best = self._init_stats(agent_list)
        results_latest = self._init_stats(agent_list)

        for depth_idx in range(self.depth):
            logger.info("Iteration Depth: %d/%d.", depth_idx + 1, self.depth)

            total_calls = self._eval_depth(
                depth_idx,
                agent_list,
                clone_agents,
                latest_candidates,
                all_candidates,
                evaluated_candidates,
                evaluate,
                module_clone,
                trainset,
                total_calls,
                results_best=results_best,
                results_latest=results_latest,
            )

            if depth_idx >= self.depth - 1:
                break

            latest_candidates = self._refine_candidates(
                agent_list,
                evaluated_candidates,
                all_candidates,
                prompt_model,
            )

        return self._select_best(
            agent_list,
            evaluated_candidates,
            total_calls,
            results_best=results_best,
            results_latest=results_latest,
        )

    def _init_candidates(
        self,
        agent_list: list[tuple[str, Any]],
        prompt_model: Any,
    ) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
        latest: dict[int, list[str]] = {}
        all_candidates: dict[int, list[str]] = {}

        for _name, agent in agent_list:
            basic_instruction = self._get_fragments(agent)
            candidates = self._generate_basic(basic_instruction, prompt_model)
            candidates.append(agent.get_system_prompt() or basic_instruction)
            latest[id(agent)] = candidates
            all_candidates[id(agent)] = list(candidates)

        return latest, all_candidates

    def _eval_depth(
        self,
        depth_idx: int,
        agent_list: list[tuple[str, Any]],
        clone_agents: list[tuple[str, Any]],
        latest_candidates: dict[int, list[str]],
        all_candidates: dict[int, list[str]],
        evaluated_candidates: dict[int, dict[tuple[str], dict[str, Any]]],
        evaluate: Evaluate,
        module_clone: Module,
        trainset: list[Example],
        total_calls: int,
        *,
        results_best: dict[int, dict[str, list[float]]],
        results_latest: dict[int, dict[str, list[float]]],
    ) -> int:
        for agent_idx, (
            (_name_old, agent_old),
            (_name_new, agent_new),
        ) in enumerate(zip(agent_list, clone_agents, strict=False)):
            candidates_to_eval = latest_candidates[id(agent_old)]
            if len(agent_list) > 1:
                candidates_to_eval = all_candidates[id(agent_old)]

            latest_scores: list[float] = []
            for candidate_idx, raw_instruction in enumerate(candidates_to_eval):
                instruction = raw_instruction.strip().strip('"')
                agent_new.optimized_system_prompt.data = instruction

                logger.info(
                    "Depth %d/%d, Candidate #%d/%d for Agent %d/%d.",
                    depth_idx + 1,
                    self.depth,
                    candidate_idx + 1,
                    len(candidates_to_eval),
                    agent_idx + 1,
                    len(agent_list),
                )
                score = evaluate(module_clone, devset=trainset).score
                total_calls += 1

                if (
                    self.track_stats
                    and len(candidates_to_eval) - self.breadth <= candidate_idx
                ):
                    latest_scores.append(score)

                key = (instruction,)
                prev = evaluated_candidates[id(agent_old)].get(key)
                if prev is None or prev["score"] < score:
                    evaluated_candidates[id(agent_old)][key] = {
                        "score": score,
                        "program": module_clone.deepcopy(),
                        "instruction": instruction,
                        "depth": depth_idx,
                    }

            if self.track_stats and latest_scores:
                self._record_stats(
                    results_latest[id(agent_old)],
                    depth_idx,
                    latest_scores,
                )
                top_scores = [
                    candidate["score"]
                    for candidate in sorted(
                        evaluated_candidates[id(agent_old)].values(),
                        key=lambda item: item["score"],
                        reverse=True,
                    )[:10]
                ]
                self._record_stats(
                    results_best[id(agent_old)],
                    depth_idx,
                    top_scores,
                )

            best = max(
                evaluated_candidates[id(agent_old)].values(),
                key=lambda candidate: candidate["score"],
            )
            agent_new.optimized_system_prompt.data = best["instruction"]

        return total_calls

    def _refine_candidates(
        self,
        agent_list: list[tuple[str, Any]],
        evaluated_candidates: dict[int, dict[tuple[str], dict[str, Any]]],
        all_candidates: dict[int, list[str]],
        prompt_model: Any,
    ) -> dict[int, list[str]]:
        new_candidates: dict[int, list[str]] = {}
        for _name, agent in agent_list:
            attempts = self._build_attempts(evaluated_candidates[id(agent)])
            generated = self._generate_given_attempts(attempts, prompt_model)
            new_candidates[id(agent)] = generated
            all_candidates[id(agent)].extend(generated)
        return new_candidates

    def _select_best(
        self,
        agent_list: list[tuple[str, Any]],
        evaluated_candidates: dict[int, dict[tuple[str], dict[str, Any]]],
        total_calls: int,
        *,
        results_best: dict[int, dict[str, list[float]]],
        results_latest: dict[int, dict[str, list[float]]],
    ) -> Module:
        all_scored: list[dict[str, Any]] = []
        for _name, agent in agent_list:
            all_scored.extend(evaluated_candidates[id(agent)].values())
        all_scored.sort(key=lambda candidate: candidate["score"], reverse=True)
        all_scored = self._drop_duplicates(all_scored)

        best_program = all_scored[0]["program"]
        best_program.candidate_programs = all_scored
        best_program.total_calls = total_calls
        best_program.prompt_model_total_calls = self.prompt_model_total_calls
        if self.track_stats:
            best_program.results_best = results_best
            best_program.results_latest = results_latest
        best_program.compile_(
            optimizer="COPRO",
            score=all_scored[0]["score"],
            total_calls=total_calls,
            prompt_model_calls=self.prompt_model_total_calls,
        )
        return best_program

    def _resolve_prompt_model(self, module: Module) -> Any:
        if self.prompt_model is not None:
            return self.prompt_model

        first_agent = next(module.named_agents(), None)
        if first_agent is None:
            return None
        _name, agent = first_agent
        return getattr(agent, "model", None)

    def _get_fragments(self, agent: Any) -> str:
        parts = []
        if hasattr(agent, "system_message") and agent.system_message.data:
            parts.append(f"System message: {agent.system_message.data}")
        if hasattr(agent, "instructions") and agent.instructions.data:
            parts.append(f"Instructions: {agent.instructions.data}")
        if hasattr(agent, "examples") and agent.examples.data:
            parts.append(f"Examples: {agent.examples.data}")
        return "\n".join(parts) if parts else "No instructions provided."

    def _generate_basic(
        self,
        basic_instruction: str,
        prompt_model: Any,
    ) -> list[str]:
        if prompt_model is None:
            return []

        prompt = _BASIC_GENERATE_INSTRUCTION.format(
            basic_instruction=basic_instruction
        )
        candidates = []
        for _ in range(self.breadth - 1):
            try:
                candidates.append(self._call_prompt_model(prompt_model, prompt))
            except Exception:
                logger.warning("Failed to generate candidate", exc_info=True)
        return candidates

    def _build_attempts(
        self,
        evaluated: dict[tuple[str], dict[str, Any]],
    ) -> str:
        items = sorted(evaluated.values(), key=lambda candidate: candidate["score"])
        lines = []
        for idx, item in enumerate(items, 1):
            lines.append(f"Attempt #{idx}: {item['instruction']}")
            lines.append(f"Score #{idx}: {item['score']:.4f}")
        return "\n".join(lines)

    def _generate_given_attempts(
        self,
        attempts_text: str,
        prompt_model: Any,
    ) -> list[str]:
        if prompt_model is None:
            return []

        prompt = _GENERATE_INSTRUCTION_GIVEN_ATTEMPTS.format(
            attempted_instructions=attempts_text
        )
        candidates = []
        for _ in range(self.breadth):
            try:
                candidates.append(self._call_prompt_model(prompt_model, prompt))
            except Exception:
                logger.warning("Failed to generate candidate", exc_info=True)
        return candidates

    def _call_prompt_model(self, prompt_model: Any, prompt: str) -> str:
        callable_model = prompt_model
        if hasattr(prompt_model, "sampling_run_params"):
            callable_model = copy.deepcopy(prompt_model)
            sampling_run_params = dict(
                getattr(callable_model, "sampling_run_params", {})
            )
            sampling_run_params["temperature"] = self.init_temperature
            callable_model.sampling_run_params = sampling_run_params
        elif hasattr(prompt_model, "model") and hasattr(
            prompt_model.model, "sampling_run_params"
        ):
            callable_model = prompt_model.deepcopy()
            sampling_run_params = dict(
                getattr(callable_model.model, "sampling_run_params", {})
            )
            sampling_run_params["temperature"] = self.init_temperature
            callable_model.model.sampling_run_params = sampling_run_params

        raw_output = callable_model(prompt)
        self.prompt_model_total_calls += 1
        return _normalize_model_output(raw_output).strip()

    @staticmethod
    def _init_stats(
        agent_list: list[tuple[str, Any]],
    ) -> dict[int, dict[str, list[float]]]:
        return {
            id(agent): {
                "depth": [],
                "max": [],
                "average": [],
                "min": [],
                "std": [],
            }
            for _name, agent in agent_list
        }

    @staticmethod
    def _record_stats(
        stats_bucket: dict[str, list[float]],
        depth_idx: int,
        scores: list[float],
    ) -> None:
        if not scores:
            return
        stats_bucket["depth"].append(depth_idx)
        stats_bucket["max"].append(max(scores))
        stats_bucket["average"].append(sum(scores) / len(scores))
        stats_bucket["min"].append(min(scores))
        stats_bucket["std"].append(
            statistics.pstdev(scores) if len(scores) > 1 else 0.0
        )

    def _check_candidates_equal(
        self,
        candidate1: dict[str, Any],
        candidate2: dict[str, Any],
    ) -> bool:
        agents1 = list(candidate1["program"].named_agents())
        agents2 = list(candidate2["program"].named_agents())
        if len(agents1) != len(agents2):
            return False

        for (_name1, agent1), (_name2, agent2) in zip(
            agents1,
            agents2,
            strict=False,
        ):
            prompt1 = agent1.optimized_system_prompt.data or agent1.get_system_prompt()
            prompt2 = agent2.optimized_system_prompt.data or agent2.get_system_prompt()
            if prompt1 != prompt2:
                return False
        return True

    def _drop_duplicates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        unique: list[dict[str, Any]] = []
        last_batch: list[dict[str, Any]] = []
        last_batch_score: float | None = None
        for candidate in candidates:
            repeat = False
            if candidate["score"] == last_batch_score:
                for prior in last_batch:
                    if self._check_candidates_equal(candidate, prior):
                        repeat = True
                        break
                if not repeat:
                    last_batch.append(candidate)
            else:
                last_batch = [candidate]
                last_batch_score = candidate["score"]
            if not repeat:
                unique.append(candidate)
        return unique


def _normalize_model_output(raw_output: Any) -> str:
    current = _unwrap_model_output(raw_output)
    if isinstance(current, str):
        return current
    if isinstance(current, dict):
        if "text" in current:
            return str(current["text"])
        if "answer" in current:
            return str(current["answer"])
        return str(current)
    if isinstance(current, Sequence) and not isinstance(current, str | bytes):
        return "" if not current else _normalize_model_output(current[0])
    return str(current)


def _unwrap_model_output(raw_output: Any) -> Any:
    current = raw_output
    seen: set[int] = set()

    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "consume") and callable(current.consume):
            consumed = current.consume()
            if consumed is not current:
                current = consumed
                continue
        if hasattr(current, "data"):
            data = current.data
            if data is not None and data is not current:
                current = data
                continue
        break

    return current
