"""SIMBA — Stochastic Introspective Mini-Batch Ascent.

Uses mini-batch evaluation with self-reflection to iteratively improve
agent demos and instructions. Strategies include appending demonstrations
from successful traces and generating reflective rules from contrastive
trajectories.

Reference: ``dspy/teleprompt/simba.py`` (376 lines) +
           ``dspy/teleprompt/simba_utils.py`` (249 lines).
"""

import inspect
import json
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from msgflux.context import trace_context
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for the OfferFeedback agent (append_a_rule strategy)
# ---------------------------------------------------------------------------

_OFFER_FEEDBACK_PROMPT = """\
You will be given two trajectories of an LLM-driven program's execution. \
Your goal is to help the program's modules build up experience on how to \
maximize the reward value assigned to the program's outputs if it were to \
receive similar inputs in the future.

The module won't see its own history. It will rely on your advice balancing \
being concrete and being generalizable.

In your advice:
- Avoid boilerplate. Offer advice that would change the module's behavior \
for the better in the future.
- Ensure that advice offered to a module M is specific to that M's specific \
sub-task, not the overall program.
- Rely on contrasting the behavior of the worse trajectory against the \
better trajectory in making recommendations.
- Ensure each unique module name appears exactly once as a key in the advice.

---

Program code:
{program_code}

Module definitions:
{modules_defn}

Program inputs:
{program_inputs}

Oracle metadata:
{oracle_metadata}

Worse trajectory (score={worse_reward_value}):
{worse_program_trajectory}

Worse outputs:
{worse_program_outputs}

Better trajectory (score={better_reward_value}):
{better_program_trajectory}

Better outputs:
{better_program_outputs}

Module names: {module_names}

Respond with a JSON object mapping each module name to a string of advice. \
Example: {{"module_name": "If the module receives X, then it should Y."}}
Only output the JSON object, nothing else."""


class SIMBA(Teleprompter):
    """Stochastic Introspective Mini-Batch Ascent.

    Samples mini-batches, identifies challenging examples with high output
    variability, then either creates self-reflective rules or adds
    successful examples as demonstrations.

    Args:
        metric: ``metric(example, prediction) -> float | bool``
        bsize: Mini-batch size.
        num_candidates: Number of candidate programs per iteration.
        max_steps: Number of optimization steps.
        max_demos: Maximum demos per agent before dropping.
        prompt_model: Model/Agent for generating feedback rules.
            Must be callable: ``prompt_model(message) -> str``.
        demo_input_field_maxlen: Max chars for demo input fields.
        num_threads: Threads for parallel execution.
        temperature_for_sampling: Temperature for program pool sampling.
        temperature_for_candidates: Temperature for candidate source sampling.

    Example::

        optimizer = SIMBA(
            metric=exact_match,
            bsize=16,
            max_steps=8,
            num_candidates=4,
        )
        compiled = optimizer.compile(student, trainset=examples)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        max_demos: int = 4,
        prompt_model: Any = None,
        demo_input_field_maxlen: int = 100_000,
        num_threads: Optional[int] = None,
        temperature_for_sampling: float = 0.2,
        temperature_for_candidates: float = 0.2,
    ):
        self.metric = metric
        self.bsize = bsize
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.prompt_model = prompt_model
        self.demo_input_field_maxlen = demo_input_field_maxlen
        self.num_threads = num_threads
        self.temperature_for_sampling = temperature_for_sampling
        self.temperature_for_candidates = temperature_for_candidates

        if self.max_demos > 0:
            self.strategies: list[Callable] = [self._append_a_demo, self._append_a_rule]
        else:
            self.strategies = [self._append_a_rule]

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        seed: int = 0,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Optimize *student* using stochastic mini-batch ascent."""
        if len(trainset) < self.bsize:
            raise ValueError(
                f"Trainset too small: {len(trainset)} < {self.bsize}"
            )

        rng = random.Random(seed)  # noqa: S311
        state = _CompileState(rng=rng)

        # Initialize baseline (idx=0)
        student = student.deepcopy()
        student._simba_idx = 0
        state.programs.append(student)
        state.program_scores[0] = []
        state.winning_programs.append(student)

        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0

        for batch_idx in range(self.max_steps):
            logger.info(
                "Starting batch %d of %d.", batch_idx + 1, self.max_steps,
            )
            instance_idx, batch = self._get_batch(
                trainset, data_indices, instance_idx, rng,
            )
            self._run_step(batch_idx, batch, state)

        return self._final_validation(state, trainset)

    def _get_batch(
        self,
        trainset: list[Example],
        data_indices: list[int],
        instance_idx: int,
        rng: random.Random,
    ) -> tuple[int, list[Example]]:
        """Get next mini-batch, reshuffling if needed."""
        if instance_idx + self.bsize > len(trainset):
            rng.shuffle(data_indices)
            instance_idx = 0
        batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
        batch = [trainset[i] for i in batch_indices]
        return instance_idx + self.bsize, batch

    def _run_step(
        self,
        batch_idx: int,
        batch: list[Example],
        state: "_CompileState",
    ) -> None:
        """Execute one optimization step."""
        top_programs = state.top_k_plus_baseline(self.num_candidates)

        # Sample trajectories
        exec_results = self._sample_trajectories(
            state.programs, batch, top_programs, state.rng,
            state.softmax_sample,
        )

        # Bucket-sort by variability
        buckets = self._bucket_sort(exec_results, batch)

        all_scores = [r["score"] for r in exec_results]
        batch_10p = _percentile(all_scores, 10)
        batch_90p = _percentile(all_scores, 90)

        # Build agent name mappings
        agent2name = _build_agent2name(exec_results)

        # Build candidates via strategies
        system_candidates = self._build_candidates(
            batch_idx, buckets, state, agent2name, batch_10p, batch_90p,
        )

        # Evaluate candidates
        candidate_scores = self._evaluate_candidates(system_candidates, batch)

        if candidate_scores:
            best_idx = candidate_scores.index(max(candidate_scores))
            state.winning_programs.append(
                system_candidates[best_idx].deepcopy(),
            )

        # Register all new candidates
        for cand_i, cand in enumerate(system_candidates):
            cand_score = (
                [candidate_scores[cand_i]]
                if cand_i < len(candidate_scores) else []
            )
            state.register_new_program(cand, cand_score)

    def _build_candidates(
        self,
        batch_idx: int,
        buckets: list,
        state: "_CompileState",
        agent2name: dict[int, str],
        batch_10p: float,
        batch_90p: float,
    ) -> list[Module]:
        """Apply strategies to top buckets and build candidate programs."""
        rng = state.rng
        system_candidates: list[Module] = []

        for bucket_idx_i, (bucket, bucket_stats) in enumerate(buckets):
            gap, max_score, avg_gap = bucket_stats
            logger.info(
                "Batch %d: bucket #%d, max=%.4f, gap=%.4f, avg_gap=%.4f.",
                batch_idx + 1, bucket_idx_i + 1, max_score, gap, avg_gap,
            )

            src_idx = state.softmax_sample(
                rng, state.top_k_plus_baseline(self.num_candidates),
                self.temperature_for_candidates,
            )
            candidate = state.programs[src_idx].deepcopy()

            name2agent, num_drop = self._drop_demos(candidate, rng)

            strategy = rng.choice(self.strategies)
            logger.info(
                "Batch %d: strategy=%s, dropped %d demos.",
                batch_idx + 1, strategy.__name__, num_drop,
            )
            try:
                strategy(
                    bucket=bucket, system=candidate,
                    agent2name=agent2name, name2agent=name2agent,
                    batch_10p_score=batch_10p, batch_90p_score=batch_90p,
                )
            except Exception:
                logger.error("Strategy failed", exc_info=True)
                continue

            system_candidates.append(candidate)
            if len(system_candidates) >= self.num_candidates + 1:
                break

        return system_candidates

    def _drop_demos(
        self, candidate: Module, rng: random.Random,
    ) -> tuple[dict[str, Any], int]:
        """Poisson-distributed demo dropping for a candidate program."""
        name2agent: dict[str, Any] = {}
        num_demos_list: list[int] = []
        max_demos_tmp = self.max_demos if self.max_demos > 0 else 3

        for name, agent in candidate.named_agents():
            name2agent[name] = agent
            num_demos_list.append(len(agent.demos))

        num_demos = max(num_demos_list) if num_demos_list else 0
        lam = num_demos / max_demos_tmp if max_demos_tmp else 0
        num_drop = max(
            _poisson(rng, lam), int(num_demos >= max_demos_tmp),
        )
        num_drop = min(num_drop, num_demos)

        if num_demos > 0:
            demos_to_drop = {
                rng.randrange(num_demos) for _ in range(num_drop)
            }
        else:
            demos_to_drop = set()

        for _name, agent in name2agent.items():
            agent.demos = [
                d for i, d in enumerate(agent.demos)
                if i not in demos_to_drop
            ]

        return name2agent, num_drop

    def _final_validation(
        self, state: "_CompileState", trainset: list[Example],
    ) -> Module:
        """Select best program via full trainset validation."""
        m_count = len(state.winning_programs) - 1
        n_count = self.num_candidates + 1
        if m_count < 1:
            prog_indices = [0] * n_count
        else:
            prog_indices = [
                round(i * m_count / (n_count - 1)) for i in range(n_count)
            ]
        prog_indices = list(dict.fromkeys(prog_indices))

        candidates = [
            state.winning_programs[i].deepcopy() for i in prog_indices
        ]
        logger.info(
            "VALIDATION: evaluating %d programs on full trainset (%d).",
            len(candidates), len(trainset),
        )

        final_scores = self._evaluate_candidates(candidates, trainset)

        if final_scores:
            best_idx = final_scores.index(max(final_scores))
            best_program = candidates[best_idx].deepcopy()
            best_score = final_scores[best_idx]
        else:
            best_program = state.winning_programs[0].deepcopy()
            best_score = 0.0

        logger.info("Final scores: %s, best=%.4f", final_scores, best_score)
        best_program.compile_(optimizer="SIMBA", score=best_score)
        return best_program

    # ------------------------------------------------------------------
    # Trajectory sampling
    # ------------------------------------------------------------------

    def _sample_trajectories(
        self,
        programs: list[Module],
        batch: list[Example],
        top_programs: list[int],
        rng: random.Random,
        softmax_sample: Callable,
    ) -> list[dict]:
        """Execute programs on batch, collecting traces and scores."""
        tasks: list[tuple[Module, Example]] = []
        for _ in range(self.num_candidates):
            for example in batch:
                chosen_idx = softmax_sample(
                    rng, top_programs, self.temperature_for_sampling,
                )
                prog = programs[chosen_idx].deepcopy()
                tasks.append((prog, example))

        results = self._run_tasks(tasks)
        return results

    def _run_tasks(self, tasks: list[tuple[Module, Example]]) -> list[dict]:
        """Run (program, example) pairs, collecting trace and score."""
        def execute_one(prog: Module, example: Example) -> dict:
            return _wrap_program(prog, example, self.metric)

        if self.num_threads and self.num_threads > 1:
            results: list[dict] = []
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {
                    executor.submit(execute_one, prog, ex): i
                    for i, (prog, ex) in enumerate(tasks)
                }
                result_map: dict[int, dict] = {}
                for future in as_completed(futures):
                    idx = futures[future]
                    result_map[idx] = future.result()
                results = [result_map[i] for i in range(len(tasks))]
            return results
        else:
            return [execute_one(prog, ex) for prog, ex in tasks]

    # ------------------------------------------------------------------
    # Bucket sorting
    # ------------------------------------------------------------------

    def _bucket_sort(
        self,
        results: list[dict],
        batch: list[Example],
    ) -> list[tuple[list[dict], tuple[float, float, float]]]:
        """Sort examples into buckets by output variability."""
        buckets = []
        for idx in range(len(batch)):
            bucket = [
                results[i]
                for i in range(idx, len(results), self.bsize)
            ]
            bucket.sort(key=lambda x: x["score"], reverse=True)

            max_score = float(bucket[0]["score"]) if bucket else 0.0
            min_score = float(bucket[-1]["score"]) if bucket else 0.0
            avg_score = (
                sum(x["score"] for x in bucket) / len(bucket) if bucket else 0.0
            )
            gap = max_score - min_score
            avg_gap = max_score - avg_score

            buckets.append((bucket, (gap, max_score, avg_gap)))

        buckets.sort(key=lambda x: x[1], reverse=True)
        return buckets

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        candidates: list[Module],
        batch: list[Example],
    ) -> list[float]:
        """Evaluate each candidate on the batch, return avg scores."""
        if not candidates:
            return []

        tasks = [
            (cand, ex) for cand in candidates for ex in batch
        ]
        results = self._run_tasks(tasks)
        bsize = len(batch)

        scores = []
        for cand_i in range(len(candidates)):
            start = cand_i * bsize
            end = (cand_i + 1) * bsize
            cand_scores = [results[i]["score"] for i in range(start, end)]
            avg = sum(cand_scores) / len(cand_scores) if cand_scores else 0.0
            scores.append(avg)
        return scores

    # ------------------------------------------------------------------
    # Strategy: append_a_demo
    # ------------------------------------------------------------------

    def _append_a_demo(
        self,
        bucket: list[dict],
        system: Module,  # noqa: ARG002
        agent2name: dict[int, str],
        name2agent: dict[str, Any],
        batch_10p_score: float,
        **kwargs: Any,  # noqa: ARG002
    ) -> bool:
        """Extract a demo from the best trace and add to agents."""
        good = bucket[0]
        trace = good.get("trace", [])

        if good["score"] <= batch_10p_score:
            logger.info(
                "Skipping demo: score %.4f at or below 10th percentile.",
                good["score"],
            )
            return False

        name2demo: dict[str, Example] = {}
        for agent, turn_record in trace:
            name = agent2name.get(id(agent))
            if name is None:
                continue

            inputs_raw = turn_record.get("inputs", "")
            output_raw = turn_record.get("assistant_output", "")

            # Truncate long inputs
            if (
                self.demo_input_field_maxlen
                and len(str(inputs_raw)) > self.demo_input_field_maxlen
            ):
                inputs_raw = (
                    str(inputs_raw)[: self.demo_input_field_maxlen]
                    + "\n\t\t... <TRUNCATED>"
                )

            demo = Example(inputs=inputs_raw, labels=output_raw)
            name2demo[name] = demo

        for name, demo in name2demo.items():
            if name in name2agent:
                name2agent[name].demos.append(demo)

        logger.info("Added %d demos across agents.", len(name2demo))
        return True

    # ------------------------------------------------------------------
    # Strategy: append_a_rule
    # ------------------------------------------------------------------

    def _append_a_rule(
        self,
        bucket: list[dict],
        system: Module,
        agent2name: dict[int, str],
        name2agent: dict[str, Any],
        batch_10p_score: float,
        batch_90p_score: float,
        **kwargs: Any,  # noqa: ARG002
    ) -> bool:
        """Generate reflective advice by contrasting good vs bad traces."""
        if self.prompt_model is None:
            logger.info("No prompt_model set, skipping append_a_rule.")
            return False

        good, bad = bucket[0], bucket[-1]

        if good["score"] <= batch_10p_score or bad["score"] >= batch_90p_score:
            logger.info(
                "Skipping rule: good=%.4f (<=10p) or bad=%.4f (>=90p).",
                good["score"], bad["score"],
            )
            return False

        # Handle edge case where good <= bad
        if good["score"] <= bad["score"]:
            empty = {
                "trace": [], "score": "N/A",
                "prediction": "N/A",
            }
            if good["score"] > batch_90p_score:
                bad = {**empty, "example": bad.get("example")}
            else:
                good = {**empty, "example": good.get("example")}

        module_names = [name for name, _ in system.named_agents()]
        example = good.get("example")

        better_trajectory = self._format_trajectory(good.get("trace", []), agent2name)
        worse_trajectory = self._format_trajectory(bad.get("trace", []), agent2name)

        modules_defn = self._inspect_modules(system)

        try:
            program_code = inspect.getsource(system.__class__)
        except (OSError, TypeError):
            program_code = str(system.__class__.__name__)

        prompt = _OFFER_FEEDBACK_PROMPT.format(
            program_code=program_code,
            modules_defn=modules_defn,
            program_inputs=str(getattr(example, "inputs", "")),
            oracle_metadata=str(getattr(example, "labels", "")),
            worse_program_trajectory=worse_trajectory,
            worse_program_outputs=str(bad.get("prediction", "")),
            worse_reward_value=bad.get("score", 0),
            better_program_trajectory=better_trajectory,
            better_program_outputs=str(good.get("prediction", "")),
            better_reward_value=good.get("score", 0),
            module_names=str(module_names),
        )

        try:
            result = self.prompt_model(prompt)
            advice_text = result if isinstance(result, str) else str(result)
            advice = _parse_json_advice(advice_text, module_names)
        except Exception:
            logger.warning("Failed to generate rule advice", exc_info=True)
            return False

        applied = 0
        for name, agent in name2agent.items():
            if name in advice:
                logger.info("Advice for %s: %s", name, advice[name][:100])
                current = agent.optimized_system_prompt.data or ""
                if not current:
                    current = self._get_fragments(agent)
                agent.optimized_system_prompt.data = (
                    current + "\n\n" + advice[name]
                )
                applied += 1

        logger.info("Applied rules to %d agents.", applied)
        return applied > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_trajectory(
        trace: list, agent2name: dict[int, str],
    ) -> str:
        """Format a trace into a human-readable trajectory string."""
        if not trace:
            return "No trajectory available."
        lines = []
        for agent, turn_record in trace:
            name = agent2name.get(id(agent), "unknown")
            inputs = turn_record.get("inputs", "")
            output = turn_record.get("assistant_output", "")
            lines.append(
                f"Module {name}:\n  Input: {inputs}\n  Output: {output}"
            )
        return "\n".join(lines)

    @staticmethod
    def _inspect_modules(program: Module) -> str:
        """Build a description of each agent module in the program."""
        sep = "-" * 60
        parts = [sep]
        for name, agent in program.named_agents():
            parts.append(f"Module {name}")
            if hasattr(agent, "system_message") and agent.system_message.data:
                parts.append(f"  System message: {agent.system_message.data}")
            if hasattr(agent, "instructions") and agent.instructions.data:
                parts.append(f"  Instructions: {agent.instructions.data}")
            if hasattr(agent, "expected_output") and agent.expected_output.data:
                parts.append(f"  Expected output: {agent.expected_output.data}")
            parts.append(sep)
        return "\n".join(parts)

    @staticmethod
    def _get_fragments(agent: Any) -> str:
        """Extract original prompt fragments as text."""
        parts = []
        if hasattr(agent, "system_message") and agent.system_message.data:
            parts.append(agent.system_message.data)
        if hasattr(agent, "instructions") and agent.instructions.data:
            parts.append(agent.instructions.data)
        if hasattr(agent, "expected_output") and agent.expected_output.data:
            parts.append(agent.expected_output.data)
        return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Compile state — shared mutable state for the optimization loop
# ---------------------------------------------------------------------------


class _CompileState:
    """Holds mutable state shared across compile helper methods."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.programs: list[Module] = []
        self.program_scores: dict[int, list[float]] = {}
        self.next_program_idx: int = 0
        self.winning_programs: list[Module] = []

    def calc_average_score(self, prog_idx: int) -> float:
        scores = self.program_scores.get(prog_idx, [])
        return sum(scores) / len(scores) if scores else 0.0

    def top_k_plus_baseline(self, k: int) -> list[int]:
        scored = sorted(
            self.programs,
            key=lambda p: self.calc_average_score(p._simba_idx),
            reverse=True,
        )
        top_k = [p._simba_idx for p in scored[:k]]
        if 0 not in top_k and top_k:
            top_k[-1] = 0
        return list(dict.fromkeys(top_k))

    def softmax_sample(
        self,
        rng_obj: random.Random,
        idxs: list[int],
        temperature: float,
    ) -> int:
        if not idxs:
            raise ValueError("No programs for softmax sampling.")
        scores = [self.calc_average_score(i) for i in idxs]
        max_s = max(scores) if scores else 0.0
        exps = [math.exp((s - max_s) / temperature) for s in scores]
        total = sum(exps)
        if total <= 0:
            return rng_obj.choice(idxs)
        probs = [e / total for e in exps]
        return rng_obj.choices(idxs, weights=probs, k=1)[0]

    def register_new_program(
        self, prog: Module, score_list: list[float],
    ) -> None:
        self.next_program_idx += 1
        prog._simba_idx = self.next_program_idx
        self.programs.append(prog)
        self.program_scores[self.next_program_idx] = score_list


def _build_agent2name(exec_results: list[dict]) -> dict[int, str]:
    """Build agent id → name mapping from execution results."""
    agent2name: dict[int, str] = {}
    for r in exec_results:
        for agent, _turn in r.get("trace", []):
            if id(agent) not in agent2name:
                for name, a in r["program_ref"].named_agents():
                    agent2name[id(a)] = name
    return agent2name


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _wrap_program(
    program: Module, example: Example, metric: Callable,
) -> dict:
    """Execute program on example inside a trace context, return results."""
    prediction = None
    score = 0.0

    with trace_context() as trace_data:
        try:
            inputs = example.inputs if hasattr(example, "inputs") else example
            prediction = program(inputs)
        except Exception:
            logger.warning("Program execution failed", exc_info=True)

    try:
        output = metric(example, prediction)
        if isinstance(output, (int, float)):
            score = float(output)
        elif isinstance(output, bool):
            score = float(output)
        else:
            score = float(output)
    except Exception:
        logger.warning("Metric evaluation failed", exc_info=True)

    return {
        "prediction": prediction,
        "trace": list(trace_data),
        "score": score,
        "example": example,
        "program_ref": program,
    }


def _percentile(values: list[float], pct: float) -> float:
    """Compute the pct-th percentile of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _poisson(rng: random.Random, lam: float) -> int:
    """Sample from a Poisson distribution using the inverse CDF method."""
    if lam <= 0:
        return 0
    l_val = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l_val:
            return k - 1


def _parse_json_advice(text: str, module_names: list[str]) -> Dict[str, str]:
    """Parse JSON advice from LLM response, fallback to heuristic."""
    # Try to extract JSON from response
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [
            line for line in lines if not line.strip().startswith("```")
        ]
        text = "\n".join(lines)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {k: str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: apply entire text to all modules
    result = {}
    for name in module_names:
        if name in text:
            result[name] = text
    return result
