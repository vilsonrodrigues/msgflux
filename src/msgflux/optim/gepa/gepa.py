"""GEPA — Guided Evolution for Prompt Adaptation.

Uses the external ``gepa`` package (>=0.0.27) for reflective evolutionary
optimization of agent instructions. The adapter bridges msgflux's
Agent/Module system to GEPA's ``GEPAAdapter`` protocol.

When ``gepa`` is not installed a simpler built-in reflective loop is used
as fallback.

Reference: ``dspy/teleprompt/gepa/`` and ``gepa`` package.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional

from msgflux.context import trace_context
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.evaluate import Evaluate
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template for instruction proposal (built-in fallback)
# ---------------------------------------------------------------------------

_PROPOSE_INSTRUCTION = """\
You are an instruction optimizer for an AI agent module.

Current instruction for module "{module_name}":
{current_instruction}

Here are examples where the module could improve:
{reflective_examples}

Based on these examples and feedback, propose an improved instruction \
that would help the module perform better. Be specific and actionable.

Respond with ONLY the improved instruction, nothing else."""


def _import_gepa():
    try:
        import gepa  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name == "gepa":
            raise ImportError(
                "GEPA requires 'gepa'. Install with `pip install gepa`."
            ) from exc
        raise
    return gepa


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _extract_score(metric_output: Any) -> float:
    """Normalise a metric return value to a plain float.

    Supports: ``int | float | bool | dict(score=...) | dotdict``.
    """
    if isinstance(metric_output, (int, float, bool)):
        return float(metric_output)
    if isinstance(metric_output, dict):
        return float(metric_output.get("score", 0))
    # dspy-style Prediction with .score
    if hasattr(metric_output, "score"):
        return float(metric_output.score)
    return float(metric_output)


def _extract_feedback(metric_output: Any, score: float) -> str:
    """Extract textual feedback from a metric return value."""
    if isinstance(metric_output, dict) and "feedback" in metric_output:
        return str(metric_output["feedback"])
    if hasattr(metric_output, "feedback") and metric_output.feedback:
        return str(metric_output.feedback)
    return f"This trajectory got a score of {score}."


def _score_only_metric(metric: Callable) -> Callable:
    """Wrap *metric* so it always returns a plain float.

    Useful when passing a rich (dict/feedback) metric to ``Evaluate``
    which expects ``metric(...) -> float``.
    """

    def wrapper(example, prediction, **kwargs):
        return _extract_score(metric(example, prediction, **kwargs))

    return wrapper


# ===================================================================
# GEPA Teleprompter
# ===================================================================


class GEPA(Teleprompter):
    """Guided Evolution for Prompt Adaptation.

    Uses reflective feedback to iteratively improve agent instructions
    through an evolutionary optimization loop powered by the ``gepa``
    package.  Falls back to a simpler built-in loop when ``gepa`` is not
    installed.

    Args:
        metric: ``metric(example, prediction) -> float | bool | dict``.
            Can return a dict with ``score`` and ``feedback`` keys for
            richer reflective optimization.
        reflection_lm: Callable for generating improved instructions.
            Must be callable: ``reflection_lm(prompt) -> str``.
        num_threads: Thread count for parallel evaluation.
        max_iterations: Maximum optimisation iterations (built-in loop).
        max_metric_calls: Budget in metric calls (gepa package).
            When *None* (default), ``max_iterations`` is converted to an
            approximate budget.
        num_candidates: Candidate instructions per iteration.
        seed: Random seed.

    Example::

        optimizer = GEPA(
            metric=exact_match,
            reflection_lm=my_callable,
            max_iterations=5,
        )
        compiled = optimizer.compile(student, trainset=train, valset=val)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        reflection_lm: Any = None,
        num_threads: Optional[int] = None,
        max_iterations: int = 5,
        max_metric_calls: Optional[int] = None,
        num_candidates: int = 3,
        seed: int = 0,
    ):
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.num_threads = num_threads
        self.max_iterations = max_iterations
        self.max_metric_calls = max_metric_calls
        self.num_candidates = num_candidates
        self.seed = seed

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Optimise agent instructions via reflective evolution."""
        valset = valset or trainset

        try:
            gepa = _import_gepa()
            return self._compile_with_gepa(
                student, trainset, valset, gepa,
            )
        except ImportError:
            logger.info(
                "gepa package not available, using built-in loop.",
            )
            return self._compile_builtin(student, trainset, valset)

    # ------------------------------------------------------------------
    # Built-in reflective optimisation loop (no gepa dependency)
    # ------------------------------------------------------------------

    def _compile_builtin(
        self,
        student: Module,
        trainset: list[Example],
        valset: list[Example],
    ) -> Module:
        """Fallback optimisation loop without gepa package."""
        # Wrap metric so Evaluate always receives a plain float
        evaluate = Evaluate(
            devset=valset,
            metric=_score_only_metric(self.metric),
            num_threads=self.num_threads or 1,
        )

        program = student.deepcopy()
        agent_names = [name for name, _ in program.named_agents()]

        candidate: dict[str, str] = {}
        for name, agent in program.named_agents():
            candidate[name] = _get_agent_fragments(agent)

        best_score = evaluate(program).score
        best_candidate = dict(candidate)
        best_program = program.deepcopy()

        logger.info("Initial score: %.4f", best_score)

        for iteration in range(self.max_iterations):
            logger.info(
                "Iteration %d/%d.", iteration + 1, self.max_iterations,
            )

            eval_results = self._evaluate_with_traces(
                program, candidate, trainset,
            )

            reflective = self._make_reflective_dataset(
                eval_results, agent_names,
            )

            new_candidate = self._propose_new_texts(
                candidate, reflective, agent_names,
            )

            new_program = self._build_program(student, new_candidate)
            new_score = evaluate(new_program).score

            logger.info(
                "Iteration %d: score=%.4f (best=%.4f)",
                iteration + 1, new_score, best_score,
            )

            if new_score > best_score:
                best_score = new_score
                best_candidate = dict(new_candidate)
                best_program = new_program.deepcopy()

            candidate = dict(best_candidate)
            program = self._build_program(student, candidate)

        best_program.compile_(optimizer="GEPA", score=best_score)
        return best_program

    # ------------------------------------------------------------------
    # gepa-package integration
    # ------------------------------------------------------------------

    def _compile_with_gepa(
        self,
        student: Module,
        trainset: list[Example],
        valset: list[Example],
        gepa: Any,
    ) -> Module:
        """Use the ``gepa`` package for optimisation."""
        rng = random.Random(self.seed)  # noqa: S311

        adapter = _MsgfluxGEPAAdapter(
            student=student,
            metric=self.metric,
            reflection_lm=self.reflection_lm,
            num_threads=self.num_threads,
            rng=rng,
        )

        seed_candidate: dict[str, str] = {}
        for name, agent in student.named_agents():
            seed_candidate[name] = _get_agent_fragments(agent)

        # Budget: explicit max_metric_calls or derive from max_iterations
        budget = self.max_metric_calls
        if budget is None:
            budget = (
                len(trainset) * self.max_iterations
                + len(valset) * (self.max_iterations + 1)
            )

        reflection_lm_fn = None
        if self.reflection_lm is not None:
            def reflection_lm_fn(prompt):
                result = self.reflection_lm(prompt)
                return result if isinstance(result, str) else str(result)

        result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm_fn,
            max_metric_calls=budget,
            skip_perfect_score=True,
            seed=self.seed,
            display_progress_bar=False,
            raise_on_exception=False,
        )

        best_candidate = result.best_candidate
        best_program = adapter.build_program(best_candidate)
        best_score = (
            max(result.val_aggregate_scores)
            if result.val_aggregate_scores
            else None
        )
        best_program.compile_(optimizer="GEPA", score=best_score)
        return best_program

    # ------------------------------------------------------------------
    # Core methods (used by built-in loop)
    # ------------------------------------------------------------------

    def _evaluate_with_traces(
        self,
        program: Module,
        candidate: dict[str, str],
        batch: list[Example],
    ) -> list[dict]:
        """Evaluate program on batch, capturing traces."""
        built = self._build_program_from_base(program, candidate)
        results = []

        for example in batch:
            with trace_context() as trace:
                try:
                    inputs = (
                        example.inputs
                        if hasattr(example, "inputs") else example
                    )
                    prediction = built(inputs)
                except Exception:
                    logger.warning("Evaluation failed", exc_info=True)
                    prediction = None

            try:
                metric_output = self.metric(example, prediction)
                score = _extract_score(metric_output)
                feedback = _extract_feedback(metric_output, score)
            except Exception:
                score = 0.0
                feedback = "Evaluation error"

            results.append({
                "example": example,
                "prediction": prediction,
                "trace": list(trace),
                "score": score,
                "feedback": feedback,
            })

        return results

    def _make_reflective_dataset(
        self,
        eval_results: list[dict],
        agent_names: list[str],
    ) -> dict[str, list[dict]]:
        """Create reflective examples from evaluation traces."""
        reflective: dict[str, list[dict]] = {
            name: [] for name in agent_names
        }

        for result in eval_results:
            if result["score"] >= 1.0:
                continue

            trace = result.get("trace", [])
            example = result["example"]

            for agent, turn_record in trace:
                agent_name = None
                for name in agent_names:
                    if hasattr(agent, "get_module_name"):
                        if name in str(agent.get_module_name()):
                            agent_name = name
                            break

                if agent_name is None:
                    continue

                reflective[agent_name].append({
                    "inputs": turn_record.get("inputs", ""),
                    "outputs": turn_record.get("assistant_output", ""),
                    "expected": str(example.labels),
                    "feedback": result["feedback"],
                    "score": result["score"],
                })

        return reflective

    def _propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective: dict[str, list[dict]],
        agent_names: list[str],
    ) -> dict[str, str]:
        """Generate improved instructions using reflective examples."""
        if self.reflection_lm is None:
            return dict(candidate)

        new_candidate: dict[str, str] = {}

        for name in agent_names:
            examples = reflective.get(name, [])
            current_instruction = candidate.get(name, "")

            if not examples:
                new_candidate[name] = current_instruction
                continue

            example_lines = []
            for i, ex in enumerate(examples[:5], 1):
                example_lines.append(
                    f"Example {i}:\n"
                    f"  Input: {ex['inputs']}\n"
                    f"  Output: {ex['outputs']}\n"
                    f"  Expected: {ex['expected']}\n"
                    f"  Feedback: {ex['feedback']}"
                )
            examples_text = "\n\n".join(example_lines)

            prompt = _PROPOSE_INSTRUCTION.format(
                module_name=name,
                current_instruction=current_instruction,
                reflective_examples=examples_text,
            )

            try:
                result = self.reflection_lm(prompt)
                text = result if isinstance(result, str) else str(result)
                new_candidate[name] = text.strip()
            except Exception:
                logger.warning(
                    "Failed to propose for %s", name, exc_info=True,
                )
                new_candidate[name] = current_instruction

        return new_candidate

    def _build_program(
        self, student: Module, candidate: dict[str, str],
    ) -> Module:
        return self._build_program_from_base(student, candidate)

    @staticmethod
    def _build_program_from_base(
        base: Module, candidate: dict[str, str],
    ) -> Module:
        program = base.deepcopy()
        for name, agent in program.named_agents():
            if name in candidate:
                agent.optimized_system_prompt.data = candidate[name]
        return program


# ===================================================================
# gepa-package adapter (implements gepa.GEPAAdapter protocol)
# ===================================================================


class _MsgfluxGEPAAdapter:
    """Adapter bridging msgflux Agent/Module system to ``gepa.GEPAAdapter``.

    Implements the four methods required by ``gepa.GEPAAdapter``:

    - ``evaluate(batch, candidate, capture_traces)``
    - ``build_program(candidate)``
    - ``make_reflective_dataset(candidate, eval_batch, components)``
    - ``propose_new_texts(candidate, reflective_dataset, components)``
    """

    def __init__(
        self,
        student: Module,
        metric: Callable,
        reflection_lm: Any = None,
        num_threads: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ):
        self.student = student
        self.metric = metric
        self.reflection_lm = reflection_lm
        self.num_threads = num_threads
        self.rng = rng or random.Random(0)  # noqa: S311

    # ---- GEPAAdapter.evaluate ----

    def evaluate(
        self,
        batch: List[Example],
        candidate: Dict[str, str],
        capture_traces: bool = False,  # noqa: FBT001, FBT002
    ) -> Any:
        from gepa import EvaluationBatch  # noqa: PLC0415

        program = self.build_program(candidate)

        if capture_traces:
            return self._evaluate_with_traces(program, batch)

        evaluator = Evaluate(
            devset=batch,
            metric=_score_only_metric(self.metric),
            num_threads=self.num_threads or 1,
            return_all_scores=True,
        )
        result = evaluator(program)
        scores = [r[2] for r in result.results]
        outputs = [r[1] for r in result.results]
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=None,
        )

    def _evaluate_with_traces(self, program, batch):
        from gepa import EvaluationBatch  # noqa: PLC0415

        scores: list[float] = []
        outputs: list[Any] = []
        trajectories: list[dict] = []

        for example in batch:
            with trace_context() as trace:
                try:
                    inputs = (
                        example.inputs
                        if hasattr(example, "inputs") else example
                    )
                    prediction = program(inputs)
                except Exception:
                    logger.warning("Evaluation failed", exc_info=True)
                    prediction = None

            try:
                metric_output = self.metric(example, prediction)
                score = _extract_score(metric_output)
            except Exception:
                score = 0.0

            scores.append(score)
            outputs.append(prediction)
            trajectories.append({
                "example": example,
                "prediction": prediction,
                "trace": list(trace),
                "score": score,
            })

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    # ---- GEPAAdapter.build_program ----

    def build_program(self, candidate: Dict[str, str]) -> Module:
        program = self.student.deepcopy()
        for name, agent in program.named_agents():
            if name in candidate:
                agent.optimized_system_prompt.data = candidate[name]
        return program

    # ---- GEPAAdapter.make_reflective_dataset ----

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],  # noqa: ARG002
        eval_batch: Any,
        components_to_update: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        ret: Dict[str, List[Dict[str, Any]]] = {}
        trajectories = eval_batch.trajectories or []

        for comp_name in components_to_update:
            items = self._reflective_items_for(
                comp_name, trajectories,
            )
            if items:
                ret[comp_name] = items

        return ret

    def _reflective_items_for(
        self,
        comp_name: str,
        trajectories: list[dict],
    ) -> list[dict]:
        items: list[dict] = []
        for data in trajectories:
            score = data["score"]
            if score >= 1.0:
                continue

            example = data["example"]
            prediction = data["prediction"]

            for agent, turn_record in data.get("trace", []):
                if not _agent_matches(agent, comp_name):
                    continue

                inputs_str = str(turn_record.get("inputs", ""))
                outputs_str = str(
                    turn_record.get("assistant_output", ""),
                )

                try:
                    out = self.metric(example, prediction)
                    feedback = _extract_feedback(out, score)
                except Exception:
                    feedback = f"Score: {score}"

                items.append({
                    "Inputs": {"problem": inputs_str},
                    "Generated Outputs": {"answer": outputs_str},
                    "Feedback": feedback,
                })
        return items

    # ---- GEPAAdapter.propose_new_texts ----

    def propose_new_texts(
        self,
        candidate: Dict[str, str],
        reflective_dataset: Dict[str, List[Dict[str, Any]]],
        components_to_update: List[str],
    ) -> Dict[str, str]:
        if self.reflection_lm is None:
            return {k: candidate[k] for k in components_to_update}

        results: dict[str, str] = {}

        for name in components_to_update:
            base_instruction = candidate[name]
            dataset = reflective_dataset.get(name, [])

            if not dataset:
                results[name] = base_instruction
                continue

            example_lines = []
            for i, ex in enumerate(dataset[:5], 1):
                example_lines.append(
                    f"Example {i}:\n"
                    f"  Inputs: {ex.get('Inputs', '')}\n"
                    f"  Generated Outputs: {ex.get('Generated Outputs', '')}\n"
                    f"  Feedback: {ex.get('Feedback', '')}"
                )
            examples_text = "\n\n".join(example_lines)

            prompt = _PROPOSE_INSTRUCTION.format(
                module_name=name,
                current_instruction=base_instruction,
                reflective_examples=examples_text,
            )

            try:
                result = self.reflection_lm(prompt)
                text = result if isinstance(result, str) else str(result)
                results[name] = text.strip()
            except Exception:
                logger.warning(
                    "Failed to propose for %s", name, exc_info=True,
                )
                results[name] = base_instruction

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_matches(agent: Any, comp_name: str) -> bool:
    """Check if *agent* matches the component name."""
    if hasattr(agent, "get_module_name"):
        return comp_name in str(agent.get_module_name())
    return False


def _get_agent_fragments(agent: Any) -> str:
    """Extract original prompt fragments as text."""
    parts = []
    if hasattr(agent, "system_message") and agent.system_message.data:
        parts.append(agent.system_message.data)
    if hasattr(agent, "instructions") and agent.instructions.data:
        parts.append(agent.instructions.data)
    if hasattr(agent, "expected_output") and agent.expected_output.data:
        parts.append(agent.expected_output.data)
    return "\n".join(parts) if parts else "No instructions provided."
