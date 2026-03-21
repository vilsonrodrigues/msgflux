"""GEPA optimizer aligned with DSPy's GEPA integration."""

import inspect
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import log2
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from msgflux.context import trace_context
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)

AUTO_RUN_SETTINGS = {
    "light": {"n": 6},
    "medium": {"n": 12},
    "heavy": {"n": 18},
}

MsgfluxTrace = list[tuple[str, Mapping[str, Any]]]


class LoggerAdapter:
    """Minimal logger bridge expected by the external ``gepa`` package."""

    def __init__(self, base_logger: logging.Logger):
        self.base_logger = base_logger

    def log(self, message: str) -> None:
        self.base_logger.info(message)


def _import_gepa():
    try:
        import gepa  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name == "gepa":
            raise ImportError(
                "GEPA requires the optional dependency 'gepa[dspy]'. "
                "Install it with `pip install msgflux[optim]` "
                "or `pip install gepa[dspy]`."
            ) from exc
        raise
    return gepa


class GEPAFeedbackMetric(Protocol):
    def __call__(
        self,
        gold: Example,
        pred: Any,
        trace: MsgfluxTrace | None = None,
        pred_name: str | None = None,
        pred_trace: MsgfluxTrace | None = None,
    ) -> Any:
        """Return a score or a score+feedback object for GEPA."""


@dataclass(frozen=True)
class MsgfluxGEPAResult:
    """Detailed GEPA run data attached when ``track_stats=True``."""

    candidates: list[Module]
    parents: list[list[int | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[list[float]]
    per_val_instance_best_candidates: list[set[int]]
    discovery_eval_counts: list[int]
    best_outputs_valset: list[list[tuple[int, list[Any]]]] | None = None
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    log_dir: str | None = None
    seed: int | None = None

    @property
    def best_idx(self) -> int:
        scores = self.val_aggregate_scores
        return max(range(len(scores)), key=lambda idx: scores[idx])

    @property
    def best_candidate(self) -> Module:
        return self.candidates[self.best_idx]

    @property
    def highest_score_achieved_per_val_task(self) -> list[float]:
        return [
            self.val_subscores[next(iter(self.per_val_instance_best_candidates[val_idx]))][val_idx]
            for val_idx in range(len(self.val_subscores[0]))
        ]

    @staticmethod
    def from_gepa_result(
        gepa_result: Any,
        adapter: "_MsgfluxGEPAAdapter",
    ) -> "MsgfluxGEPAResult":
        return MsgfluxGEPAResult(
            candidates=[
                adapter.build_program(candidate)
                for candidate in gepa_result.candidates
            ],
            parents=gepa_result.parents,
            val_aggregate_scores=gepa_result.val_aggregate_scores,
            val_subscores=gepa_result.val_subscores,
            per_val_instance_best_candidates=(
                gepa_result.per_val_instance_best_candidates
            ),
            discovery_eval_counts=gepa_result.discovery_eval_counts,
            best_outputs_valset=getattr(gepa_result, "best_outputs_valset", None),
            total_metric_calls=getattr(gepa_result, "total_metric_calls", None),
            num_full_val_evals=getattr(gepa_result, "num_full_val_evals", None),
            log_dir=getattr(gepa_result, "run_dir", None),
            seed=getattr(gepa_result, "seed", None),
        )


def _extract_score(metric_output: Any, default: float = 0.0) -> float:
    """Normalize a metric return value to a plain float score."""
    if metric_output is None:
        return default
    if isinstance(metric_output, (int, float, bool)):
        return float(metric_output)
    if isinstance(metric_output, Mapping):
        return float(metric_output.get("score", default))
    if hasattr(metric_output, "score"):
        return float(metric_output.score)
    return float(metric_output)


def _extract_feedback(metric_output: Any, score: float) -> str:
    """Extract textual feedback from a metric return value."""
    if isinstance(metric_output, Mapping):
        feedback = metric_output.get("feedback")
        if feedback:
            return str(feedback)
    if hasattr(metric_output, "feedback") and metric_output.feedback:
        return str(metric_output.feedback)
    return f"This trajectory got a score of {score}."


def _stringify_for_reflection(value: Any) -> Any:
    """Convert values into a JSON-serializable structure for reflection."""
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return {
            str(key): _stringify_for_reflection(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_stringify_for_reflection(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_for_reflection(item) for item in value]
    return str(value)


def _normalize_reflection_output(raw_output: Any) -> str:
    """Normalize reflection LM output to a plain string."""
    raw_output = _unwrap_reflection_output(raw_output)
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, Mapping):
        if "text" in raw_output:
            return str(raw_output["text"])
        if "answer" in raw_output:
            return str(raw_output["answer"])
        return str(dict(raw_output))
    if isinstance(raw_output, Sequence) and not isinstance(raw_output, str | bytes):
        return "" if not raw_output else _normalize_reflection_output(raw_output[0])
    return str(raw_output)


def _unwrap_reflection_output(raw_output: Any) -> Any:
    """Unwrap response containers like ModelResponse before string normalization."""
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


def _get_agent_fragments(agent: Any) -> str:
    """Extract the prompt text currently optimized by msgflux optimizers."""
    parts = []
    if hasattr(agent, "system_message") and agent.system_message.data:
        parts.append(agent.system_message.data)
    if hasattr(agent, "instructions") and agent.instructions.data:
        parts.append(agent.instructions.data)
    return "\n".join(parts) if parts else "No instructions provided."


def _collect_tool_descriptions(program: Module) -> dict[str, str]:
    """Extract tool descriptions from all agents in a program.

    Returns a dict mapping ``"agent_name::tool_name"`` to description text.
    """
    descriptions: dict[str, str] = {}
    for name, agent in _named_target_agents(program):
        if not hasattr(agent, "tool_library"):
            continue
        for tool_name in agent.tool_library.library:
            tool = agent.tool_library.library[tool_name]
            desc = tool.get_module_description() or ""
            key = f"{name}::{tool_name}"
            descriptions[key] = desc
    return descriptions


def _apply_tool_descriptions(
    program: Module,
    optimized: dict[str, str],
) -> None:
    """Apply optimized tool descriptions back to the program's tools."""
    for name, agent in _named_target_agents(program):
        if not hasattr(agent, "tool_library"):
            continue
        for tool_name in agent.tool_library.library:
            key = f"{name}::{tool_name}"
            if key in optimized:
                tool = agent.tool_library.library[tool_name]
                tool.set_description(optimized[key])


def _agent_key(name: str, agent: Any) -> str:
    if hasattr(agent, "name") and agent.name:
        return str(agent.name)
    if name:
        return name
    return str(agent.get_module_name())


def _named_target_agents(program: Module) -> list[tuple[str, Any]]:
    """Return GEPA component names paired with their Agent instances."""
    named_agents: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for name, agent in program.named_agents():
        key = _agent_key(name, agent)
        if key in seen:
            raise ValueError(
                f"GEPA requires unique Agent names; found duplicate name {key!r}."
            )
        seen.add(key)
        named_agents.append((key, agent))
    return named_agents


class GEPA(Teleprompter):
    """Guided Evolutionary Prompt Adaptation for msgflux agents."""

    def __init__(
        self,
        metric: GEPAFeedbackMetric,
        *,
        auto: Literal["light", "medium", "heavy"] | None = None,
        max_full_evals: int | None = None,
        max_metric_calls: int | None = None,
        reflection_minibatch_size: int = 3,
        candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",
        reflection_lm: Callable[[Any], Any] | None = None,
        skip_perfect_score: bool = True,
        add_format_failure_as_feedback: bool = False,
        instruction_proposer: Callable | None = None,
        component_selector: Any = "round_robin",
        use_merge: bool = True,
        max_merge_invocations: int | None = 5,
        num_threads: int | None = None,
        failure_score: float = 0.0,
        perfect_score: float = 1.0,
        log_dir: str | None = None,
        track_stats: bool = False,
        use_wandb: bool = False,
        wandb_api_key: str | None = None,
        wandb_init_kwargs: dict[str, Any] | None = None,
        track_best_outputs: bool = False,
        warn_on_score_mismatch: bool = True,
        use_mlflow: bool = False,
        optimize_tools: bool = False,
        optimize_tools_budget: int | None = None,
        seed: int | None = 0,
        gepa_kwargs: dict[str, Any] | None = None,
    ):
        try:
            inspect.signature(metric).bind(None, None, None, None, None)
        except TypeError as exc:
            raise TypeError(
                "GEPA metric must accept five arguments: "
                "(gold, pred, trace, pred_name, pred_trace)."
            ) from exc

        if (
            (max_metric_calls is not None)
            + (max_full_evals is not None)
            + (auto is not None)
        ) != 1:
            raise ValueError(
                "Exactly one of `auto`, `max_full_evals`, "
                "or `max_metric_calls` must be set."
            )

        if reflection_lm is None and instruction_proposer is None:
            raise ValueError(
                "GEPA requires `reflection_lm` or `instruction_proposer`."
            )

        if track_best_outputs and not track_stats:
            raise ValueError(
                "`track_stats` must be True when `track_best_outputs` is enabled."
            )

        self.metric_fn = metric
        self.auto = auto
        self.max_full_evals = max_full_evals
        self.max_metric_calls = max_metric_calls
        self.reflection_minibatch_size = reflection_minibatch_size
        self.candidate_selection_strategy = candidate_selection_strategy
        self.reflection_lm = reflection_lm
        self.skip_perfect_score = skip_perfect_score
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.custom_instruction_proposer = instruction_proposer
        self.component_selector = component_selector
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.num_threads = num_threads
        self.failure_score = failure_score
        self.perfect_score = perfect_score
        self.log_dir = log_dir
        self.track_stats = track_stats
        self.use_wandb = use_wandb
        self.wandb_api_key = wandb_api_key
        self.wandb_init_kwargs = wandb_init_kwargs
        self.track_best_outputs = track_best_outputs
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.use_mlflow = use_mlflow
        self.optimize_tools = optimize_tools
        self.optimize_tools_budget = optimize_tools_budget
        self.seed = seed
        self.gepa_kwargs = gepa_kwargs or {}

    @staticmethod
    def auto_budget(
        num_preds: int,
        num_candidates: int,
        valset_size: int,
        minibatch_size: int = 35,
        full_eval_steps: int = 5,
    ) -> int:
        """Replicate DSPy's GEPA budget heuristic for ``auto`` runs."""
        if num_trials := int(
            max(2 * (num_preds * 2) * log2(num_candidates), 1.5 * num_candidates)
        ):
            pass
        else:
            num_trials = 0

        if full_eval_steps < 1:
            raise ValueError("`full_eval_steps` must be >= 1.")

        total = valset_size
        total += num_candidates * 5
        total += num_trials * minibatch_size

        if num_trials == 0:
            return total

        periodic_fulls = (num_trials + 1) // full_eval_steps + 1
        extra_final = 1 if num_trials < full_eval_steps else 0
        total += (periodic_fulls + extra_final) * valset_size
        return total

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        valset: list[Example] | None = None,
    ) -> Module:
        """Optimize the student's agent prompts with the GEPA engine."""
        gepa = _import_gepa()

        if not trainset:
            raise ValueError("`trainset` must be provided and non-empty.")
        if teacher is not None:
            raise ValueError("GEPA does not support `teacher`.")

        candidate_count = AUTO_RUN_SETTINGS[self.auto]["n"] if self.auto else None
        effective_max_metric_calls = self.max_metric_calls
        if candidate_count is not None:
            effective_max_metric_calls = self.auto_budget(
                num_preds=len(list(student.named_agents())),
                num_candidates=candidate_count,
                valset_size=len(valset) if valset is not None else len(trainset),
            )
        elif self.max_full_evals is not None:
            effective_max_metric_calls = self.max_full_evals * (
                len(trainset) + (len(valset) if valset is not None else 0)
            )

        resolved_valset = valset or trainset
        if valset is None:
            logger.warning(
                "No `valset` provided to GEPA; using `trainset` as validation set."
            )

        named_agents = _named_target_agents(student)
        seed_candidate = {
            name: _get_agent_fragments(agent)
            for name, agent in named_agents
        }
        if not seed_candidate:
            raise ValueError("GEPA requires at least one Agent component.")

        adapter = _MsgfluxGEPAAdapter(
            student=student,
            metric_fn=self.metric_fn,
            feedback_map={
                name: self._feedback_fn_creator(name)
                for name, _agent in named_agents
            },
            failure_score=self.failure_score,
            num_threads=self.num_threads,
            add_format_failure_as_feedback=self.add_format_failure_as_feedback,
            rng=random.Random(self.seed),  # noqa: S311
            reflection_lm=self.reflection_lm,
            custom_instruction_proposer=self.custom_instruction_proposer,
            warn_on_score_mismatch=self.warn_on_score_mismatch,
            reflection_minibatch_size=self.reflection_minibatch_size,
        )

        logger.info(
            "Running GEPA with budget ~= %s metric calls.",
            effective_max_metric_calls,
        )

        gepa_result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=resolved_valset,
            adapter=adapter,
            reflection_lm=(
                adapter.reflection_lm_call if self.reflection_lm is not None else None
            ),
            candidate_selection_strategy=self.candidate_selection_strategy,
            skip_perfect_score=self.skip_perfect_score,
            reflection_minibatch_size=self.reflection_minibatch_size,
            module_selector=self.component_selector,
            perfect_score=self.perfect_score,
            use_merge=self.use_merge,
            max_merge_invocations=(
                0 if self.max_merge_invocations is None else self.max_merge_invocations
            ),
            max_metric_calls=effective_max_metric_calls,
            run_dir=self.log_dir,
            use_wandb=self.use_wandb,
            wandb_api_key=self.wandb_api_key,
            wandb_init_kwargs=self.wandb_init_kwargs,
            use_mlflow=self.use_mlflow,
            track_best_outputs=self.track_best_outputs,
            logger=LoggerAdapter(logger),
            display_progress_bar=True,
            raise_on_exception=True,
            seed=self.seed,
            **self.gepa_kwargs,
        )

        best_program = adapter.build_program(gepa_result.best_candidate)
        best_score = (
            gepa_result.val_aggregate_scores[gepa_result.best_idx]
            if gepa_result.val_aggregate_scores
            else None
        )
        best_program.compile_(
            optimizer="GEPA",
            score=best_score,
            total_metric_calls=getattr(gepa_result, "total_metric_calls", None),
            num_full_val_evals=getattr(gepa_result, "num_full_val_evals", None),
        )

        if self.track_stats:
            best_program.detailed_results = MsgfluxGEPAResult.from_gepa_result(
                gepa_result, adapter
            )

        if self.optimize_tools:
            self._optimize_tool_descriptions(
                best_program, trainset, resolved_valset
            )

        return best_program

    def _optimize_tool_descriptions(
        self,
        program: Module,
        trainset: list[Example],
        valset: list[Example],
    ) -> None:
        """Use ``gepa.optimize_anything`` to optimize tool descriptions.

        Runs as a post-compile step on the already-optimized program.
        Each tool description is treated as an independent text artifact
        that GEPA can evolve using Actionable Side Information from the
        evaluator (which tool was selected vs. expected).
        """
        import gepa.optimize_anything as oa  # noqa: PLC0415
        from gepa.optimize_anything import (  # noqa: PLC0415
            EngineConfig,
            GEPAConfig,
            ReflectionConfig,
            optimize_anything,
        )

        tool_descs = _collect_tool_descriptions(program)
        if not tool_descs:
            logger.info("optimize_tools: no tools found, skipping.")
            return

        budget = self.optimize_tools_budget or max(
            6 * len(tool_descs), 12
        )

        logger.info(
            "optimize_tools: optimizing %d tool descriptions (budget=%d).",
            len(tool_descs),
            budget,
        )

        def _evaluator(
            candidate: dict[str, str],
            example: Example,
        ) -> float:
            """Run the program with candidate tool descriptions and score."""
            trial_program = program.deepcopy()
            _apply_tool_descriptions(trial_program, candidate)

            try:
                prediction = _MsgfluxGEPAAdapter._run_program(
                    trial_program, example
                )
            except Exception:
                oa.log("Program execution failed with these tool descriptions.")
                return self.failure_score

            metric_output = self.metric_fn(
                example, prediction, None, None, None,
            )
            score = _extract_score(metric_output, self.failure_score)
            feedback = _extract_feedback(metric_output, score)
            oa.log(f"Score: {score}")
            oa.log(f"Feedback: {feedback}")

            # Log which tools exist for the reflector
            for key, desc in candidate.items():
                oa.log(f"Tool [{key}]: {desc[:120]}")

            return score

        reflection_config = {}
        if self.reflection_lm is not None:
            reflection_config["reflection"] = ReflectionConfig(
                reflection_lm=self.reflection_lm,
            )

        result = optimize_anything(
            seed_candidate=tool_descs,
            evaluator=_evaluator,
            dataset=trainset,
            valset=valset,
            objective=(
                "Optimize the tool descriptions so the AI agent selects "
                "the correct tool for each task. Descriptions should be "
                "clear, specific, and help the model distinguish between "
                "similar tools."
            ),
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=budget),
                **reflection_config,
            ),
        )

        _apply_tool_descriptions(program, result.best_candidate)
        logger.info(
            "optimize_tools: done. Best score: %s",
            getattr(result, "best_score", "N/A"),
        )

    def _feedback_fn_creator(self, pred_name: str) -> Callable[..., dict[str, Any]]:
        def feedback_fn(
            predictor_output: Any,
            predictor_inputs: Any,
            module_inputs: Example,
            module_outputs: Any,
            captured_trace: MsgfluxTrace,
            pred_trace: MsgfluxTrace | None = None,
        ) -> dict[str, Any]:
            _ = predictor_output, predictor_inputs
            metric_output = self.metric_fn(
                module_inputs,
                module_outputs,
                captured_trace,
                pred_name,
                pred_trace,
            )
            score = _extract_score(metric_output, self.failure_score)
            feedback = _extract_feedback(metric_output, score)
            return {"score": score, "feedback": feedback}

        return feedback_fn


class _MsgfluxGEPAAdapter:
    """Bridge msgflux Agent programs into ``gepa.optimize``."""

    def __init__(
        self,
        *,
        student: Module,
        metric_fn: GEPAFeedbackMetric,
        feedback_map: dict[str, Callable[..., dict[str, Any]]],
        failure_score: float = 0.0,
        num_threads: int | None = None,
        add_format_failure_as_feedback: bool = False,
        rng: random.Random | None = None,
        reflection_lm: Callable[[Any], Any] | None = None,
        custom_instruction_proposer: Callable | None = None,
        warn_on_score_mismatch: bool = True,
        reflection_minibatch_size: int | None = None,
    ):
        self.student = student
        self.metric_fn = metric_fn
        self.feedback_map = feedback_map
        self.failure_score = failure_score
        self.num_threads = num_threads
        self.add_format_failure_as_feedback = add_format_failure_as_feedback
        self.rng = rng or random.Random(0)  # noqa: S311
        self.reflection_lm = reflection_lm
        self.custom_instruction_proposer = custom_instruction_proposer
        self.warn_on_score_mismatch = warn_on_score_mismatch
        self.reflection_minibatch_size = reflection_minibatch_size

    def reflection_lm_call(self, prompt: Any) -> str:
        if self.reflection_lm is None:
            raise ValueError("Reflection LM is not configured.")
        return _normalize_reflection_output(self.reflection_lm(prompt))

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.custom_instruction_proposer is not None:
            return self.custom_instruction_proposer(
                candidate=candidate,
                reflective_dataset=reflective_dataset,
                components_to_update=components_to_update,
            )

        if self.reflection_lm is None:
            return {name: candidate[name] for name in components_to_update}

        from gepa.strategies.instruction_proposal import (  # noqa: PLC0415
            InstructionProposalSignature,
        )

        results: dict[str, str] = {}
        for name in components_to_update:
            results[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm_call,
                input_dict={
                    "current_instruction_doc": candidate[name],
                    "dataset_with_feedback": reflective_dataset[name],
                },
            )["new_instruction"]
        return results

    def build_program(self, candidate: dict[str, str]) -> Module:
        program = self.student.deepcopy()
        for name, agent in _named_target_agents(program):
            if name in candidate:
                agent.optimized_system_prompt.data = candidate[name]
        return program

    def evaluate(
        self,
        batch: list[Example],
        candidate: dict[str, str],
        *,
        capture_traces: bool = False,
    ) -> Any:
        from gepa import EvaluationBatch  # noqa: PLC0415

        program = self.build_program(candidate)
        if not batch:
            return EvaluationBatch(
                outputs=[],
                scores=[],
                trajectories=[] if capture_traces else None,
            )

        if self.num_threads and self.num_threads > 1 and len(batch) > 1:
            results = self._evaluate_parallel(
                program,
                batch,
                capture_traces=capture_traces,
            )
        else:
            results = [
                self._evaluate_one(
                    program,
                    example,
                    capture_traces=capture_traces,
                )
                for example in batch
            ]

        outputs = [item[0] for item in results]
        scores = [item[1] for item in results]
        trajectories = [item[2] for item in results] if capture_traces else None

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def _evaluate_parallel(
        self,
        program: Module,
        batch: list[Example],
        *,
        capture_traces: bool,
    ) -> list[tuple[Any, float, dict[str, Any] | None]]:
        results: dict[int, tuple[Any, float, dict[str, Any] | None]] = {}
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {
                executor.submit(
                    self._evaluate_one,
                    program.deepcopy(),
                    example,
                    capture_traces=capture_traces,
                ): idx
                for idx, example in enumerate(batch)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [results[idx] for idx in range(len(batch))]

    def _evaluate_one(
        self,
        program: Module,
        example: Example,
        *,
        capture_traces: bool,
    ) -> tuple[Any, float, dict[str, Any] | None]:
        prediction = None
        metric_output = None
        trace: MsgfluxTrace = []

        if capture_traces:
            with trace_context() as raw_trace:
                try:
                    prediction = self._run_program(program, example)
                except Exception:
                    logger.warning(
                        "Program execution failed during GEPA.",
                        exc_info=True,
                    )
            trace = self._normalize_trace(program, raw_trace)
        else:
            try:
                prediction = self._run_program(program, example)
            except Exception:
                logger.warning("Program execution failed during GEPA.", exc_info=True)

        score = self.failure_score
        try:
            metric_output = self.metric_fn(
                example,
                prediction,
                trace or None,
                None,
                None,
            )
            score = _extract_score(metric_output, self.failure_score)
        except Exception:
            logger.warning("Metric evaluation failed during GEPA.", exc_info=True)

        trajectory = None
        if capture_traces:
            trajectory = {
                "example": example,
                "prediction": prediction,
                "trace": trace,
                "score": score,
                "metric_output": metric_output,
            }

        return prediction, score, trajectory

    @staticmethod
    def _run_program(program: Module, example: Example) -> Any:
        inputs = example.inputs if hasattr(example, "inputs") else example
        if isinstance(inputs, Mapping):
            try:
                return program(inputs)
            except TypeError as original_exc:
                try:
                    return program(**inputs)
                except TypeError:
                    raise original_exc from None
        return program(inputs)

    @staticmethod
    def _normalize_trace(program: Module, raw_trace: list) -> MsgfluxTrace:
        agent_names = {id(agent): name for name, agent in _named_target_agents(program)}
        normalized: MsgfluxTrace = []
        for agent, turn_record in raw_trace:
            normalized.append(
                (
                    agent_names.get(id(agent), getattr(agent, "name", "unknown")),
                    turn_record,
                )
            )
        return normalized

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],  # noqa: ARG002
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ret: dict[str, list[dict[str, Any]]] = {}
        trajectories = eval_batch.trajectories or []

        for comp_name in components_to_update:
            items: list[dict[str, Any]] = []

            for data in trajectories:
                trace = data.get("trace", [])
                trace_instances = [
                    trace_item
                    for trace_item in trace
                    if trace_item[0] == comp_name
                ]

                if not trace_instances:
                    continue

                selected = self.rng.choice(trace_instances)
                _pred_name, turn_record = selected

                try:
                    feedback = self.feedback_map[comp_name](
                        predictor_output=turn_record.get("assistant_output"),
                        predictor_inputs=turn_record.get("inputs"),
                        module_inputs=data["example"],
                        module_outputs=data["prediction"],
                        captured_trace=trace,
                        pred_trace=[selected],
                    )
                except Exception:
                    logger.warning("GEPA feedback extraction failed.", exc_info=True)
                    feedback = {
                        "score": self.failure_score,
                        "feedback": (
                            f"This trajectory got a score of {self.failure_score}."
                        ),
                    }

                module_score = float(data.get("score", self.failure_score))
                if feedback["score"] != module_score and self.warn_on_score_mismatch:
                    logger.warning(
                        "GEPA metric returned a component score different from the "
                        "module-level score. msgflux follows DSPy's behavior and keeps "
                        "the module-level score for optimization."
                    )
                    self.warn_on_score_mismatch = False

                items.append(
                    {
                        "Inputs": _stringify_for_reflection(
                            turn_record.get("inputs", "")
                        ),
                        "Generated Outputs": _stringify_for_reflection(
                            turn_record.get("assistant_output", "")
                        ),
                        "Feedback": feedback["feedback"],
                    }
                )

            if items:
                ret[comp_name] = items

        if not ret:
            raise RuntimeError("No valid reflective examples found for GEPA.")

        return ret
