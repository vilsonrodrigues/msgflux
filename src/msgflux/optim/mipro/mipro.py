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

import logging
import random
from typing import Any, Callable, List, Literal, Optional

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.evaluate import Evaluate
from msgflux.optim.teleprompter import Teleprompter
from msgflux.optim.utils import eval_candidate_program

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

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
    "Break the task into clear, sequential steps.",
    "Include a concrete example of ideal input/output behavior.",
    "Specify the format of the expected output explicitly.",
    "Use role-playing to set the agent's persona and expertise.",
    "Add constraints to avoid common failure modes.",
    "Be specific about edge cases and how to handle them.",
]


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
    ):
        if auto is not None and auto not in AUTO_RUN_SETTINGS:
            raise ValueError(
                f"Invalid auto mode: {auto}. "
                f"Must be one of {set(AUTO_RUN_SETTINGS)}."
            )
        self.metric = metric
        self.prompt_model = prompt_model
        self.auto = auto
        self.num_candidates = num_candidates
        self.num_trials = num_trials
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_threads = num_threads
        self.seed = seed
        self.init_temperature = init_temperature
        self.minibatch_size = minibatch_size
        self.minibatch_full_eval_steps = minibatch_full_eval_steps

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        valset: Optional[List[Example]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        """Optimize via bootstrap + instruction proposal + Optuna search."""
        seed = seed or self.seed
        rng = random.Random(seed)  # noqa: S311

        # Split train/val
        trainset, valset = self._split_datasets(trainset, valset, rng)

        # Resolve hyperparameters from auto mode
        num_candidates, num_trials, use_minibatch = (
            self._resolve_hyperparams(student, valset)
        )

        program = student.deepcopy()
        evaluate = Evaluate(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads or 1,
        )

        # Phase 1: Bootstrap demo candidate sets
        demo_candidates = self._bootstrap_demo_sets(
            program, trainset, teacher, num_candidates, seed,
        )

        # Phase 2: Propose instruction candidates
        instruction_candidates = self._propose_instructions(
            program, trainset, demo_candidates, num_candidates, rng,
        )

        # Phase 3: Optuna search
        best_program = self._optuna_search(
            program, instruction_candidates, demo_candidates,
            evaluate, valset, num_trials, use_minibatch, seed, rng,
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
        seed: int,  # noqa: ARG002
    ) -> dict[str, list[list[Example]]]:
        """Create N demo candidate sets per agent via BootstrapFewShot."""
        from msgflux.optim.bootstrap import BootstrapFewShot  # noqa: PLC0415

        logger.info("Phase 1: Bootstrapping %d demo candidate sets.", num_sets)

        agent_names = [name for name, _ in program.named_agents()]
        demo_candidates: dict[str, list[list[Example]]] = {
            name: [] for name in agent_names
        }

        for i in range(num_sets):
            try:
                bs = BootstrapFewShot(
                    metric=self.metric,
                    max_bootstrapped_demos=self.max_bootstrapped_demos,
                    max_labeled_demos=self.max_labeled_demos,
                )
                compiled = bs.compile(
                    program, trainset=trainset, teacher=teacher,
                )
                for name, agent in compiled.named_agents():
                    if name in demo_candidates:
                        demo_candidates[name].append(list(agent.optimized_examples))
            except Exception:
                logger.warning(
                    "Bootstrap set %d failed", i + 1, exc_info=True,
                )
                # Fallback: empty demos
                for name in agent_names:
                    demo_candidates[name].append([])

            logger.info("Bootstrapped set %d/%d.", i + 1, num_sets)

        return demo_candidates

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
    ) -> dict[str, list[str]]:
        """Generate instruction candidates per agent."""
        logger.info("Phase 2: Proposing %d instructions per agent.", num_candidates)

        instruction_candidates: dict[str, list[str]] = {}

        data_summary = self._summarize_data(trainset, rng)

        for name, agent in program.named_agents():
            agent_info = _get_agent_info(agent)

            # Original instruction as candidate 0
            original = agent.get_system_prompt() or agent_info
            candidates = [original]

            if self.prompt_model is None:
                # Without prompt_model, just repeat original
                instruction_candidates[name] = candidates
                continue

            # Generate additional candidates
            demos = demo_candidates.get(name, [[]])
            for i in range(num_candidates - 1):
                demo_context = ""
                if demos:
                    demo_set = demos[i % len(demos)]
                    if demo_set:
                        demo_context = "Example demonstrations:\n"
                        for d in demo_set[:3]:
                            demo_context += (
                                f"  Input: {d.inputs}\n  Output: {d.labels}\n"
                            )

                tip = rng.choice(_PROMPTING_TIPS)
                prompt = _PROPOSE_WITH_TIP.format(
                    agent_info=agent_info,
                    data_summary=data_summary,
                    demo_context=demo_context,
                    tip=tip,
                )

                try:
                    result = self.prompt_model(prompt)
                    text = result if isinstance(result, str) else str(result)
                    candidates.append(text.strip())
                except Exception:
                    logger.warning(
                        "Failed to generate instruction %d for %s",
                        i + 1, name, exc_info=True,
                    )

            instruction_candidates[name] = candidates
            logger.info(
                "Agent %s: %d instruction candidates.",
                name, len(candidates),
            )

        return instruction_candidates

    # ------------------------------------------------------------------
    # Phase 3: Optuna search
    # ------------------------------------------------------------------

    def _optuna_search(
        self,
        program: Module,
        instruction_candidates: dict[str, list[str]],
        demo_candidates: dict[str, list[list[Example]]],
        evaluate: Evaluate,
        valset: list[Example],
        num_trials: int,
        use_minibatch: bool,  # noqa: FBT001
        seed: int,
        rng: random.Random,
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

        agent_names = [name for name, _ in program.named_agents()]

        # Evaluate default program
        default_score = eval_candidate_program(
            len(valset), valset, program, evaluate, rng,
        ).score
        logger.info("Default program score: %.4f", default_score)

        best_score = default_score
        best_program = program.deepcopy()
        score_data: list[dict] = [
            {"score": default_score, "program": program.deepcopy()},
        ]
        batch_size = (
            self.minibatch_size if use_minibatch else len(valset)
        )

        def objective(trial: Any) -> float:
            nonlocal best_score, best_program

            candidate = program.deepcopy()

            # Select instruction + demos per agent
            for name, agent in candidate.named_agents():
                instrs = instruction_candidates.get(name, [])
                if instrs:
                    idx = trial.suggest_categorical(
                        f"{name}_instruction",
                        list(range(len(instrs))),
                    )
                    agent.optimized_system_prompt.data = instrs[idx]

                demos = demo_candidates.get(name, [])
                if demos:
                    d_idx = trial.suggest_categorical(
                        f"{name}_demos",
                        list(range(len(demos))),
                    )
                    agent.optimized_examples = list(demos[d_idx])

            # Evaluate
            score = eval_candidate_program(
                batch_size, valset, candidate, evaluate, rng,
            ).score

            if not use_minibatch and score > best_score:
                best_score = score
                best_program = candidate.deepcopy()

            score_data.append(
                {"score": score, "program": candidate},
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
            demos = demo_candidates.get(name, [])
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

        best_program.compile_(
            optimizer="MIPROv2",
            score=best_score,
            total_calls=num_trials,
        )

        # Attach metadata
        sorted_data = sorted(
            score_data, key=lambda x: x["score"], reverse=True,
        )
        best_program.candidate_programs = sorted_data

        return best_program

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
        top_candidates = sorted(
            score_data, key=lambda x: x["score"], reverse=True,
        )[:5]

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
    ) -> tuple[int, int, bool]:
        """Resolve num_candidates, num_trials, use_minibatch."""
        if self.auto is not None:
            settings = AUTO_RUN_SETTINGS[self.auto]
            num_candidates = settings["n"]
            num_agents = len(list(student.named_agents()))
            num_vars = max(num_agents * 2, 1)
            import math  # noqa: PLC0415
            num_trials = int(
                max(2 * num_vars * math.log2(max(num_candidates, 2)),
                    1.5 * num_candidates)
            )
            use_minibatch = len(valset) > 50
            return num_candidates, num_trials, use_minibatch

        num_candidates = self.num_candidates or 6
        num_trials = self.num_trials or 30
        use_minibatch = len(valset) > 50
        return num_candidates, num_trials, use_minibatch

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


def _get_agent_info(agent: Any) -> str:
    """Extract agent configuration as text."""
    parts = []
    if hasattr(agent, "system_message") and agent.system_message.data:
        parts.append(f"System message: {agent.system_message.data}")
    if hasattr(agent, "instructions") and agent.instructions.data:
        parts.append(f"Instructions: {agent.instructions.data}")
    return "\n".join(parts) if parts else "No configuration provided."
