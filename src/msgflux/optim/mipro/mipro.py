"""MIPROv2 — Multi-prompt Instruction Proposal Optimization.

Combines instruction generation, bootstrapped demos, and Optuna-based
Bayesian search to find the best prompt configuration per agent.

Three phases:
    1. Bootstrap few-shot demo candidate sets via BootstrapFewShot.
    2. Propose instruction candidates via a prompt model.
    3. Search (instruction_idx, demo_idx) per agent with Optuna TPESampler.

Optuna is an optional dependency — imported at runtime.

Reference: ``dspy/teleprompt/mipro_optimizer_v2.py`` (853 lines).
"""

import copy
import logging
import random
from collections import defaultdict
from typing import Any, Callable, List, Literal, Optional, Sequence

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.evaluate import Evaluate
from msgflux.optim.teleprompter import Teleprompter
from msgflux.optim.utils import (
    create_minibatch,
    create_n_fewshot_demo_sets,
    eval_candidate_program,
    get_program_with_highest_avg_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROPOSE_INSTRUCTION = """\
You are an instruction optimizer for an AI agent.

Here is the agent's current configuration:
{agent_info}

Here are some example inputs from the training data:
{data_summary}

{demo_context}

Your task is to propose a single, improved system prompt for this agent \
that will lead it to perform the task well. The prompt should be concise, \
clear, and actionable.

Respond with ONLY the improved system prompt, nothing else."""

_PROPOSE_WITH_TIP = """\
You are an instruction optimizer for an AI agent.

Here is the agent's current configuration:
{agent_info}

Here are some example inputs from the training data:
{data_summary}

{demo_context}

Prompting tip: {tip}

Your task is to propose a single, improved system prompt for this agent \
that will lead it to perform the task well. The prompt should be concise, \
clear, and actionable.

Respond with ONLY the improved system prompt, nothing else."""

_PROMPTING_TIPS = [
    "",
    "Break the task into clear, sequential steps.",
    "Include a concrete example of ideal input/output behavior.",
    "Specify the format of the expected output explicitly.",
    "Use role-playing to set the agent's persona and expertise.",
    "Add constraints to avoid common failure modes.",
    "Be specific about edge cases and how to handle them.",
]

_DATASET_SUMMARY_PROMPT = """\
You are analyzing a task dataset for prompt optimization.

Representative training examples:
{examples}

Write a concise summary of the task, the input patterns, the expected outputs,
and any important ambiguities or edge cases.

Return only the summary."""

_PROGRAM_DESCRIPTION_PROMPT = """\
You are analyzing a msgflux program composed of one or more agents.

Program outline:
{program_outline}

Write a concise description of the overall task this program is solving and how
the agents appear to divide the work.

Return only the description."""

_MODULE_DESCRIPTION_PROMPT = """\
You are analyzing one component inside a msgflux program.

Program overview:
{program_description}

Component name:
{component_name}

Current component configuration:
{agent_info}

Describe this component's role in the broader program and what it should do well.

Return only the description."""

_INSTRUCTION_PROPOSAL_PROMPT = """\
You are optimizing one agent prompt inside a msgflux program.

Dataset summary:
{dataset_summary}

Program overview:
{program_description}

Component name:
{component_name}

Component role:
{module_description}

Current prompt:
{basic_instruction}

Representative task demos:
{task_demos}

{tip_block}Write one improved prompt for this component. Keep it specific,
grounded in the task, and focused on reliable task success.

Return only the improved prompt."""


def _import_optuna():
    try:
        import optuna  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                "MIPROv2 requires 'optuna'. "
                "Install with `pip install optuna`."
            ) from exc
        raise
    return optuna


class MIPROv2(Teleprompter):
    """Multi-prompt Instruction Proposal Optimization v2.

    Args:
        metric: ``metric(example, prediction) -> float | bool``
        prompt_model: Model/Agent for generating instruction candidates.
            Must be callable: ``prompt_model(message) -> str``.
        auto: Preset mode — ``"light"``, ``"medium"``, or ``"heavy"``.
        num_candidates: Number of instruction/demo candidates.
        num_trials: Number of Optuna trials (ignored when *auto* is set).
        max_bootstrapped_demos: Max bootstrapped demos per candidate set.
        max_labeled_demos: Max labeled demos per candidate set.
        num_threads: Threads for evaluation.
        seed: Random seed for reproducibility.
        init_temperature: Temperature for instruction generation.
        minibatch_size: Size of minibatch for evaluation.
        minibatch_full_eval_steps: Full eval every N minibatch steps.

    Example::

        optimizer = MIPROv2(
            metric=exact_match,
            prompt_model=prompt_fn,
            auto="light",
        )
        compiled = optimizer.compile(student, trainset=examples)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        prompt_model: Any = None,
        auto: Optional[Literal["light", "medium", "heavy"]] = "light",
        num_candidates: Optional[int] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        num_threads: Optional[int] = None,
        seed: int = 9,
        init_temperature: float = 1.0,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        teacher_settings: Optional[dict[str, Any]] = None,
        max_errors: Optional[int] = None,
        metric_threshold: Optional[float] = None,
        verbose: bool = False,
    ):
        if auto is not None and auto not in AUTO_RUN_SETTINGS:
            raise ValueError(
                f"Invalid auto mode: {auto}. "
                f"Must be one of {set(AUTO_RUN_SETTINGS)}."
            )
        self.metric = metric
        self.prompt_model = prompt_model
        self.auto = auto
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.num_candidates = num_candidates
        self.num_trials = num_trials
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_threads = num_threads
        self.seed = seed
        self.init_temperature = init_temperature
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps
        self.teacher_settings = teacher_settings or {}
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.verbose = verbose
        self.prompt_model_total_calls = 0

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        valset: Optional[List[Example]] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: Optional[int] = None,
        minibatch_full_eval_steps: Optional[int] = None,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Optimize via bootstrap + instruction proposal + Optuna search."""
        seed = seed or self.seed
        rng = random.Random(seed)  # noqa: S311
        resolved_num_trials = num_trials or self.num_trials or 30
        resolved_minibatch_size = minibatch_size or self.minibatch_size
        resolved_full_eval_steps = (
            minibatch_full_eval_steps or self.minibatch_full_eval_steps
        )
        effective_max_bootstrapped_demos = (
            self.max_bootstrapped_demos
            if max_bootstrapped_demos is None
            else max_bootstrapped_demos
        )
        effective_max_labeled_demos = (
            self.max_labeled_demos
            if max_labeled_demos is None
            else max_labeled_demos
        )
        zeroshot_opt = (
            effective_max_bootstrapped_demos == 0
            and effective_max_labeled_demos == 0
        )

        # Split train/val
        trainset, valset = self._split_datasets(trainset, valset, rng)

        # Resolve hyperparameters from auto mode
        (
            num_instruct_candidates,
            num_fewshot_candidates,
            resolved_num_trials,
            use_minibatch,
            valset,
        ) = self._resolve_hyperparams(
            student,
            valset,
            num_trials=resolved_num_trials,
            minibatch=minibatch,
            zeroshot_opt=zeroshot_opt,
        )
        if resolved_minibatch_size > len(valset):
            raise ValueError(
                "Minibatch size cannot exceed the size of the valset. "
                f"Valset size: {len(valset)}."
            )

        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads or 1,
        )

        # Phase 1: Bootstrap demo candidate sets
        demo_candidates = self._bootstrap_demo_sets(
            program,
            trainset,
            teacher,
            num_fewshot_candidates,
            seed,
            zeroshot_opt=zeroshot_opt,
            max_bootstrapped_demos=effective_max_bootstrapped_demos,
            max_labeled_demos=effective_max_labeled_demos,
        )

        # Phase 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program,
            trainset,
            demo_candidates,
            num_instruct_candidates,
            rng,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            data_aware=data_aware_proposer,
            tip_aware=tip_aware_proposer,
            fewshot_aware=fewshot_aware_proposer,
        )

        if zeroshot_opt:
            demo_candidates = None

        # Phase 3: Optuna search
        best_program = self._optuna_search(
            program,
            instruction_candidates,
            demo_candidates,
            evaluate,
            valset,
            resolved_num_trials,
            use_minibatch,
            seed,
            rng,
            minibatch_size=resolved_minibatch_size,
            minibatch_full_eval_steps=resolved_full_eval_steps,
        )

        return best_program

    # ------------------------------------------------------------------
    # Phase 1: Bootstrap demo sets
    # ------------------------------------------------------------------

    def _bootstrap_demo_sets(
        self,
        program: Module,
        trainset: list[Example],
        teacher: Optional[Module],
        num_sets: int,
        seed: int,
        *,
        zeroshot_opt: bool,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
    ) -> dict[str, list[list[Example]]]:
        """Create DSPy-style demo candidate sets per agent."""
        logger.info("Phase 1: Bootstrapping %d demo candidate sets.", num_sets)
        return create_n_fewshot_demo_sets(
            student=program,
            num_candidate_sets=num_sets,
            trainset=trainset,
            max_labeled_demos=(
                LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT
                if zeroshot_opt
                else max_labeled_demos
            ),
            max_bootstrapped_demos=(
                BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT
                if zeroshot_opt
                else max_bootstrapped_demos
            ),
            metric=self.metric,
            teacher_settings=self.teacher_settings,
            max_errors=self.max_errors,
            metric_threshold=self.metric_threshold,
            teacher=teacher,
            seed=seed,
            rng=random.Random(seed),  # noqa: S311
        )

    # ------------------------------------------------------------------
    # Phase 2: Propose instructions
    # ------------------------------------------------------------------

    def _propose_instructions(
        self,
        program: Module,
        trainset: list[Example],
        demo_candidates: dict[str, list[list[Example]]],
        num_candidates: int,
        rng: random.Random,
        *,
        view_data_batch_size: int,
        program_aware: bool,
        data_aware: bool,
        tip_aware: bool,
        fewshot_aware: bool,
    ) -> dict[str, list[str]]:
        """Generate instruction candidates per agent."""
        logger.info("Phase 2: Proposing %d instructions per agent.", num_candidates)

        instruction_candidates: dict[str, list[str]] = {}

        proposer = None
        if self.prompt_model is not None:
            proposer = _GroundedPromptProposer(
                prompt_model=self.prompt_model,
                program=program,
                trainset=trainset,
                view_data_batch_size=view_data_batch_size,
                program_aware=program_aware,
                use_dataset_summary=data_aware,
                use_task_demos=fewshot_aware,
                num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
                use_tip=tip_aware,
                set_tip_randomly=tip_aware,
                verbose=self.verbose,
                rng=rng,
                init_temperature=self.init_temperature,
            )
        proposed_instruction_sets = (
            proposer.propose_instructions_for_program(
                program=program,
                demo_candidates=demo_candidates,
                num_candidates=max(num_candidates - 1, 0),
            )
            if proposer is not None
            else {}
        )
        if proposer is not None:
            self.prompt_model_total_calls = proposer.prompt_model_total_calls

        for name, agent in program.named_agents():
            component_name = _agent_key(name, agent)
            original = agent.get_system_prompt() or _get_agent_info(agent)
            candidates = [original]
            candidates.extend(proposed_instruction_sets.get(component_name, []))
            instruction_candidates[component_name] = candidates
            logger.info(
                "Agent %s: %d instruction candidates.",
                component_name,
                len(candidates),
            )

        return instruction_candidates

    # ------------------------------------------------------------------
    # Phase 3: Optuna search
    # ------------------------------------------------------------------

    def _optuna_search(
        self,
        program: Module,
        instruction_candidates: dict[str, list[str]],
        demo_candidates: Optional[dict[str, list[list[Example]]]],
        evaluate: Evaluate,
        valset: list[Example],
        num_trials: int,
        use_minibatch: bool,  # noqa: FBT001
        seed: int,
        rng: random.Random,
        *,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
    ) -> Module:
        """Use Optuna to search instruction/demo combinations."""
        optuna = _import_optuna()

        logger.info(
            "Phase 3: Optuna search (%d trials).", num_trials,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True,
        )
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
        )

        agent_names = [
            _agent_key(name, agent) for name, agent in program.named_agents()
        ]

        # Evaluate default program
        default_score = eval_candidate_program(
            len(valset), valset, program, evaluate, rng,
        ).score
        logger.info("Default program score: %.4f", default_score)

        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data: list[dict] = [
            {
                "score": default_score,
                "program": program.deepcopy(),
                "full_eval": True,
            },
        ]
        param_score_dict: dict[
            str,
            list[tuple[float, Module, list[tuple[str, int]]]],
        ] = defaultdict(list)
        fully_evaled_param_combos: set[str] = set()
        batch_size = minibatch_size if use_minibatch else len(valset)

        def objective(trial: Any) -> float:
            nonlocal best_score, best_program, total_eval_calls

            candidate = program.deepcopy()
            chosen_params, raw_chosen_params = self._apply_trial_configuration(
                candidate,
                instruction_candidates,
                demo_candidates,
                trial,
            )

            # Evaluate
            score = eval_candidate_program(
                batch_size, valset, candidate, evaluate, rng,
            ).score
            total_eval_calls += batch_size

            if not use_minibatch and score > best_score:
                best_score = score
                best_program = candidate.deepcopy()

            score_data.append(
                {
                    "score": score,
                    "program": candidate.deepcopy(),
                    "full_eval": batch_size >= len(valset),
                },
            )
            categorical_key = ",".join(map(str, chosen_params))
            param_score_dict[categorical_key].append(
                (score, candidate.deepcopy(), raw_chosen_params)
            )

            best_score, best_program, total_eval_calls = (
                self._maybe_run_periodic_full_eval(
                    trial_number=trial.number + 1,
                    num_trials=num_trials,
                    use_minibatch=use_minibatch,
                    minibatch_full_eval_steps=minibatch_full_eval_steps,
                    param_score_dict=param_score_dict,
                    fully_evaled_param_combos=fully_evaled_param_combos,
                    evaluate=evaluate,
                    valset=valset,
                    rng=rng,
                    score_data=score_data,
                    best_score=best_score,
                    best_program=best_program,
                    total_eval_calls=total_eval_calls,
                )
            )

            logger.info(
                "Trial %d: score=%.4f (best=%.4f)",
                trial.number + 1, score, best_score,
            )
            return score

        # Add default as baseline trial
        default_params = {}
        distributions = {}
        cat_dist = optuna.distributions.CategoricalDistribution
        for name in agent_names:
            instrs = instruction_candidates.get(name, [])
            if instrs:
                key = f"{name}_instruction"
                default_params[key] = 0
                distributions[key] = cat_dist(list(range(len(instrs))))
            demos = [] if demo_candidates is None else demo_candidates.get(name, [])
            if demos:
                key = f"{name}_demos"
                default_params[key] = 0
                distributions[key] = cat_dist(list(range(len(demos))))

        baseline_trial = optuna.trial.create_trial(
            params=default_params,
            distributions=distributions,
            value=default_score,
        )
        study.add_trial(baseline_trial)

        # Run optimization
        study.optimize(objective, n_trials=num_trials)

        # If minibatch was used, do final full eval on top candidates
        if use_minibatch:
            best_score, best_program = self._final_full_eval(
                score_data, evaluate, valset, rng,
                best_score, best_program,
            )

        best_program.prompt_model_total_calls = self.prompt_model_total_calls
        best_program.compile_(
            optimizer="MIPROv2",
            score=best_score,
            total_calls=total_eval_calls,
            prompt_model_calls=self.prompt_model_total_calls,
        )

        # Attach metadata
        sorted_data = sorted(
            score_data, key=lambda x: x["score"], reverse=True,
        )
        best_program.candidate_programs = sorted_data
        best_program.mb_candidate_programs = [
            row for row in sorted_data if not row.get("full_eval", False)
        ]

        return best_program

    def _apply_trial_configuration(
        self,
        candidate: Module,
        instruction_candidates: dict[str, list[str]],
        demo_candidates: Optional[dict[str, list[list[Example]]]],
        trial: Any,
    ) -> tuple[list[int], list[tuple[str, int]]]:
        chosen_params: list[int] = []
        raw_chosen_params: list[tuple[str, int]] = []

        for name, agent in candidate.named_agents():
            component_name = _agent_key(name, agent)
            instrs = instruction_candidates.get(component_name, [])
            if instrs:
                idx = trial.suggest_categorical(
                    f"{component_name}_instruction",
                    list(range(len(instrs))),
                )
                agent.optimized_system_prompt.data = instrs[idx]
                chosen_params.append(idx)
                raw_chosen_params.append((f"{component_name}_instruction", idx))

            demos = [] if demo_candidates is None else demo_candidates.get(
                component_name, []
            )
            if demos:
                demo_idx = trial.suggest_categorical(
                    f"{component_name}_demos",
                    list(range(len(demos))),
                )
                agent.optimized_examples = list(demos[demo_idx])
                chosen_params.append(demo_idx)
                raw_chosen_params.append((f"{component_name}_demos", demo_idx))

        return chosen_params, raw_chosen_params

    def _maybe_run_periodic_full_eval(
        self,
        *,
        trial_number: int,
        num_trials: int,
        use_minibatch: bool,
        minibatch_full_eval_steps: int,
        param_score_dict: dict[str, list[tuple[float, Module, list[tuple[str, int]]]]],
        fully_evaled_param_combos: set[str],
        evaluate: Evaluate,
        valset: list[Example],
        rng: random.Random,
        score_data: list[dict[str, Any]],
        best_score: float,
        best_program: Module,
        total_eval_calls: int,
    ) -> tuple[float, Module, int]:
        should_run = use_minibatch and (
            trial_number % minibatch_full_eval_steps == 0
            or trial_number == num_trials
        )
        if not should_run:
            return best_score, best_program, total_eval_calls

        full_eval_program, _mean, key, _params = get_program_with_highest_avg_score(
            param_score_dict,
            fully_evaled_param_combos,
        )
        if key in fully_evaled_param_combos:
            return best_score, best_program, total_eval_calls

        full_eval_score = eval_candidate_program(
            len(valset),
            valset,
            full_eval_program,
            evaluate,
            rng,
        ).score
        total_eval_calls += len(valset)
        fully_evaled_param_combos.add(key)
        score_data.append(
            {
                "score": full_eval_score,
                "program": full_eval_program.deepcopy(),
                "full_eval": True,
            }
        )
        if full_eval_score > best_score:
            best_score = full_eval_score
            best_program = full_eval_program.deepcopy()

        return best_score, best_program, total_eval_calls

    def _final_full_eval(
        self,
        score_data: list[dict],
        evaluate: Evaluate,
        valset: list[Example],
        rng: random.Random,
        best_score: float,
        best_program: Module,
    ) -> tuple[float, Module]:
        """Full evaluation of top candidates after minibatch search."""
        top_candidates = [
            row
            for row in sorted(
                score_data, key=lambda x: x["score"], reverse=True,
            )
            if not row.get("full_eval", False)
        ][:5]

        for entry in top_candidates:
            prog = entry["program"]
            score = eval_candidate_program(
                len(valset), valset, prog, evaluate, rng,
            ).score
            if score > best_score:
                best_score = score
                best_program = prog.deepcopy()

        logger.info("Final full eval best: %.4f", best_score)
        return best_score, best_program

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_datasets(
        self,
        trainset: list[Example],
        valset: Optional[list[Example]],
        rng: random.Random,  # noqa: ARG002
    ) -> tuple[list[Example], list[Example]]:
        """Split trainset into train/val if no valset provided."""
        if not trainset:
            raise ValueError("Trainset cannot be empty.")

        if valset is not None:
            return trainset, valset

        if len(trainset) < 2:
            raise ValueError(
                "Trainset must have at least 2 examples."
            )

        val_size = min(1000, max(1, int(len(trainset) * 0.80)))
        cutoff = len(trainset) - val_size
        return trainset[:cutoff], trainset[cutoff:]

    def _resolve_hyperparams(
        self,
        student: Module,
        valset: list[Example],
        *,
        num_trials: int,
        minibatch: bool,
        zeroshot_opt: bool,
    ) -> tuple[int, int, int, bool, list[Example]]:
        """Resolve candidate counts, num_trials, and minibatching."""
        if self.auto is not None:
            settings = AUTO_RUN_SETTINGS[self.auto]
            num_agents = len(list(student.named_agents()))
            num_vars = max(num_agents * (1 if zeroshot_opt else 2), 1)
            import math  # noqa: PLC0415
            resolved_num_trials = int(
                max(2 * num_vars * math.log2(max(settings["n"], 2)),
                    1.5 * settings["n"])
            )
            resolved_valset = valset
            if len(valset) > settings["val_size"]:
                resolved_valset = create_minibatch(
                    valset,
                    settings["val_size"],
                    random.Random(self.seed),  # noqa: S311
                )
            use_minibatch = len(resolved_valset) > MIN_MINIBATCH_SIZE
            num_instruct_candidates = (
                settings["n"] if zeroshot_opt else max(int(settings["n"] * 0.5), 1)
            )
            num_fewshot_candidates = settings["n"]
            return (
                num_instruct_candidates,
                num_fewshot_candidates,
                resolved_num_trials,
                use_minibatch,
                resolved_valset,
            )

        num_instruct_candidates = (
            self.num_instruct_candidates or self.num_candidates or 6
        )
        num_fewshot_candidates = (
            self.num_fewshot_candidates or self.num_candidates or 6
        )
        use_minibatch = minibatch and len(valset) > MIN_MINIBATCH_SIZE
        return (
            num_instruct_candidates,
            num_fewshot_candidates,
            num_trials,
            use_minibatch,
            valset,
        )

    @staticmethod
    def _summarize_data(
        trainset: list[Example], rng: random.Random,
    ) -> str:
        """Create a brief summary of training data."""
        sample_size = min(5, len(trainset))
        samples = rng.sample(trainset, sample_size)
        lines = []
        for i, ex in enumerate(samples, 1):
            lines.append(f"  Example {i}: Input={ex.inputs}, Output={ex.labels}")
        return "\n".join(lines)


class _GroundedPromptProposer:
    """Lightweight msgflux adaptation of DSPy's GroundedProposer."""

    def __init__(
        self,
        *,
        prompt_model: Any,
        program: Module,
        trainset: list[Example],
        view_data_batch_size: int = 10,
        program_aware: bool = True,
        use_dataset_summary: bool = True,
        use_task_demos: bool = True,
        num_demos_in_context: int = 3,
        use_tip: bool = True,
        set_tip_randomly: bool = True,
        verbose: bool = False,
        rng: random.Random | None = None,
        init_temperature: float = 1.0,
    ):
        self.prompt_model = prompt_model
        self.program = program
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.num_demos_in_context = num_demos_in_context
        self.use_tip = use_tip
        self.set_tip_randomly = set_tip_randomly
        self.verbose = verbose
        self.rng = rng or random.Random(0)  # noqa: S311
        self.init_temperature = init_temperature
        self.prompt_model_total_calls = 0
        self.program_outline = _render_program_outline(program)
        self.dataset_summary = (
            self._create_dataset_summary(trainset, view_data_batch_size)
            if use_dataset_summary
            else "Not provided."
        )
        self.program_description = (
            self._create_program_description()
            if program_aware
            else self.program_outline
        )
        self.module_descriptions = {
            _agent_key(name, agent): self._create_module_description(
                _agent_key(name, agent),
                agent,
            )
            for name, agent in program.named_agents()
        }

    def propose_instructions_for_program(
        self,
        *,
        program: Module,
        demo_candidates: dict[str, list[list[Example]]] | None,
        num_candidates: int,
    ) -> dict[str, list[str]]:
        proposed: dict[str, list[str]] = {}

        for name, agent in program.named_agents():
            component_name = _agent_key(name, agent)
            component_proposals: list[str] = []
            component_demo_candidates = (demo_candidates or {}).get(component_name, [])
            proposal_iterations = self._get_num_instruction_candidates(
                component_demo_candidates,
                num_candidates,
            )
            for demo_set_idx in range(proposal_iterations):
                prompt = self._build_instruction_prompt(
                    component_name=component_name,
                    agent=agent,
                    demo_candidates=component_demo_candidates,
                    demo_set_idx=demo_set_idx,
                )
                component_proposals.append(self._call_prompt_model(prompt))
            proposed[component_name] = component_proposals

        return proposed

    def _get_num_instruction_candidates(
        self,
        demo_candidates: list[list[Example]],
        num_candidates: int,
    ) -> int:
        if not demo_candidates:
            return num_candidates
        return min(num_candidates, max(len(demo_candidates), 1))

    def _create_dataset_summary(
        self,
        trainset: list[Example],
        view_data_batch_size: int,
    ) -> str:
        sample_size = min(view_data_batch_size, len(trainset))
        sample_examples = self.rng.sample(trainset, sample_size)
        example_blob = _summarize_examples_for_prompt(sample_examples)
        prompt = _DATASET_SUMMARY_PROMPT.format(examples=example_blob)
        try:
            return self._call_prompt_model(prompt, temperature=0.2)
        except Exception:
            logger.warning(
                "Falling back to deterministic dataset summary.",
                exc_info=True,
            )
            return example_blob

    def _create_program_description(self) -> str:
        prompt = _PROGRAM_DESCRIPTION_PROMPT.format(
            program_outline=self.program_outline
        )
        try:
            return self._call_prompt_model(prompt, temperature=0.2)
        except Exception:
            logger.warning(
                "Falling back to deterministic program summary.",
                exc_info=True,
            )
            return self.program_outline

    def _create_module_description(self, component_name: str, agent: Any) -> str:
        agent_info = _get_agent_info(agent)
        prompt = _MODULE_DESCRIPTION_PROMPT.format(
            program_description=self.program_description,
            component_name=component_name,
            agent_info=agent_info,
        )
        try:
            return self._call_prompt_model(prompt, temperature=0.2)
        except Exception:
            logger.warning(
                "Falling back to deterministic module summary.",
                exc_info=True,
            )
            return agent_info

    def _build_instruction_prompt(
        self,
        *,
        component_name: str,
        agent: Any,
        demo_candidates: list[list[Example]],
        demo_set_idx: int,
    ) -> str:
        task_demos = self._build_task_demos(demo_candidates, demo_set_idx)
        tip = self._choose_tip()
        tip_block = f"Prompting tip: {tip}\n\n" if tip else ""
        return _INSTRUCTION_PROPOSAL_PROMPT.format(
            dataset_summary=self.dataset_summary,
            program_description=self.program_description,
            component_name=component_name,
            module_description=self.module_descriptions.get(
                component_name,
                _get_agent_info(agent),
            ),
            basic_instruction=agent.get_system_prompt() or _get_agent_info(agent),
            task_demos=task_demos,
            tip_block=tip_block,
        )

    def _build_task_demos(
        self,
        demo_candidates: list[list[Example]],
        demo_set_idx: int,
    ) -> str:
        if not self.use_task_demos or not demo_candidates or demo_set_idx == 0:
            return "No task demos provided."

        current_set = demo_candidates[demo_set_idx:demo_set_idx + 1]
        adjacent_sets = (
            current_set
            + demo_candidates[demo_set_idx + 1:]
            + demo_candidates[:demo_set_idx]
        )
        gathered: list[str] = []
        for demo_set in adjacent_sets:
            for demo in demo_set:
                gathered.append(f"Input: {demo.inputs}\nOutput: {demo.labels}")
                if len(gathered) >= self.num_demos_in_context:
                    return "\n\n".join(gathered)

        return "\n\n".join(gathered) if gathered else "No task demos provided."

    def _choose_tip(self) -> str | None:
        if not self.use_tip:
            return None
        if self.set_tip_randomly:
            tip = self.rng.choice(_PROMPTING_TIPS)
            return tip or None
        return _PROMPTING_TIPS[1]

    def _call_prompt_model(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
    ) -> str:
        callable_model = self.prompt_model
        if hasattr(self.prompt_model, "sampling_run_params"):
            callable_model = copy.deepcopy(self.prompt_model)
            sampling_run_params = dict(
                getattr(callable_model, "sampling_run_params", {})
            )
            sampling_run_params["temperature"] = (
                self.init_temperature if temperature is None else temperature
            )
            callable_model.sampling_run_params = sampling_run_params

        raw_output = callable_model(prompt)
        self.prompt_model_total_calls += 1
        text = _normalize_model_output(raw_output).strip()
        if self.verbose:
            logger.info("Grounded proposer output: %s", text[:200])
        return text


def _get_agent_info(agent: Any) -> str:
    """Extract agent configuration as text."""
    parts = []
    if hasattr(agent, "system_message") and agent.system_message.data:
        parts.append(f"System message: {agent.system_message.data}")
    if hasattr(agent, "instructions") and agent.instructions.data:
        parts.append(f"Instructions: {agent.instructions.data}")
    return "\n".join(parts) if parts else "No configuration provided."


def _agent_key(name: str, agent: Any) -> str:
    if getattr(agent, "name", None):
        return str(agent.name)
    return str(name)


def _render_program_outline(program: Module) -> str:
    blocks = []
    for path, agent in program.named_agents():
        component_name = _agent_key(path, agent)
        blocks.append(
            "\n".join(
                [
                    f"Component: {component_name}",
                    _get_agent_info(agent),
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "No agents found."


def _summarize_examples_for_prompt(examples: Sequence[Example]) -> str:
    lines = []
    for idx, example in enumerate(examples, 1):
        lines.append(
            f"Example {idx}\nInput: {example.inputs}\nOutput: {example.labels}"
        )
    return "\n\n".join(lines)


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
