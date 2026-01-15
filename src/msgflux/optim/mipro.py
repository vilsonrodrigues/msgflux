"""MIPROv2 - Multi-prompt Instruction Proposal Optimization.

MIPROv2 optimizes multiple prompt components simultaneously using
Bayesian optimization to find the best combination of instructions
and demonstrations.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer


# Template for instruction proposal
INSTRUCTION_PROPOSAL_TEMPLATE = """You are an expert prompt engineer. Generate a clear and effective instruction for an AI system.

## Task Context
{task_context}

## Requirements
The instruction should:
1. Be specific and actionable
2. Include any necessary constraints
3. Guide the AI toward the correct output format

## Example Inputs and Outputs
{examples}

Generate a single, comprehensive instruction:"""


@dataclass
class PromptCandidate:
    """A candidate prompt configuration."""

    instruction: str
    demos: List[Example] = field(default_factory=list)
    score: float = 0.0
    evaluated: bool = False

    def __hash__(self):
        return hash((self.instruction, tuple(self.demos)))


@dataclass
class MiproTrial:
    """A single trial in the optimization process."""

    instruction_idx: int
    demo_indices: List[int]
    score: float
    candidate: PromptCandidate


class MIPROv2(Optimizer):
    """Multi-prompt Instruction Proposal Optimization v2.

    MIPROv2 optimizes prompt configurations by:
    1. Generating multiple instruction candidates
    2. Selecting demonstration examples
    3. Using surrogate model to predict performance
    4. Iteratively refining based on evaluations

    Example:
        >>> from msgflux.optim import MIPROv2
        >>> from msgflux.evaluate.metrics import exact_match
        >>>
        >>> optimizer = MIPROv2(
        ...     agent.parameters(),
        ...     metric=exact_match,
        ...     prompt_model=generator_model,
        ...     num_candidates=10,
        ...     num_trials=50,
        ...     seed=42,
        ... )
        >>> optimizer.step(trainset, valset)

    Args:
        params: Iterable of Parameters to optimize.
        metric: Metric function for evaluation.
        prompt_model: Module to use for generating instructions.
        num_candidates: Number of instruction candidates to generate.
        num_demos: Number of demonstrations per candidate.
        num_trials: Number of optimization trials.
        init_temperature: Initial temperature for sampling.
        task_context: Description of the task.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        metric: Callable[[Example, Any], float],
        prompt_model: Optional[Module] = None,
        num_candidates: int = 10,
        num_demos: int = 4,
        num_trials: int = 50,
        init_temperature: float = 1.0,
        task_context: Optional[str] = None,
        seed: int = 0,
    ):
        defaults = dict(
            num_candidates=num_candidates,
            num_demos=num_demos,
            num_trials=num_trials,
            init_temperature=init_temperature,
            seed=seed,
        )
        super().__init__(params, defaults)

        self.metric = metric
        self.prompt_model = prompt_model
        self.num_candidates = num_candidates
        self.num_demos = num_demos
        self.num_trials = num_trials
        self.init_temperature = init_temperature
        self.task_context = task_context or "Complete the given task accurately."
        self.seed = seed
        self.rng = random.Random(seed)

        # Track optimization state
        self._instruction_candidates: List[str] = []
        self._demo_pool: List[Example] = []
        self._trials: List[MiproTrial] = []
        self._best_candidate: Optional[PromptCandidate] = None
        self._best_score: float = 0.0

        # Simple surrogate model: average score per instruction/demo combination
        self._instruction_scores: Dict[int, List[float]] = {}
        self._demo_scores: Dict[int, List[float]] = {}

    def step(
        self,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        *,
        teacher: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Perform one MIPROv2 optimization step.

        Args:
            trainset: Training examples for demonstrations.
            valset: Validation examples for evaluation.
            teacher: Module to evaluate.
            closure: Closure for loss computation.

        Returns:
            The loss value if closure is provided, None otherwise.
        """
        self._step_count += 1

        if valset is None:
            valset = trainset

        # Initialize demonstration pool
        self._demo_pool = trainset.copy()

        # Generate instruction candidates
        if not self._instruction_candidates:
            self._instruction_candidates = self._generate_instruction_candidates(
                trainset
            )

        # Run optimization trials
        for trial_idx in range(self.num_trials):
            # Sample instruction and demos based on surrogate predictions
            instruction_idx = self._sample_instruction()
            demo_indices = self._sample_demos()

            # Create candidate
            candidate = PromptCandidate(
                instruction=self._instruction_candidates[instruction_idx],
                demos=[self._demo_pool[i] for i in demo_indices if i < len(self._demo_pool)],
            )

            # Evaluate candidate
            if teacher is not None:
                score = self._evaluate_candidate(candidate, valset, teacher)
            else:
                score = 0.0

            candidate.score = score
            candidate.evaluated = True

            # Record trial
            trial = MiproTrial(
                instruction_idx=instruction_idx,
                demo_indices=demo_indices,
                score=score,
                candidate=candidate,
            )
            self._trials.append(trial)

            # Update surrogate model
            self._update_surrogate(trial)

            # Track best
            if score > self._best_score:
                self._best_score = score
                self._best_candidate = candidate

        # Apply best configuration to parameters
        if self._best_candidate is not None:
            self._apply_candidate(self._best_candidate)

        if closure is not None:
            return closure()
        return None

    def _generate_instruction_candidates(
        self, trainset: List[Example]
    ) -> List[str]:
        """Generate instruction candidates using the prompt model."""
        candidates = []

        # Format example inputs/outputs
        sample_examples = self.rng.sample(
            trainset, min(5, len(trainset))
        )
        examples_str = "\n".join(
            f"Input: {ex.inputs}\nOutput: {ex.labels}"
            for ex in sample_examples
        )

        for i in range(self.num_candidates):
            if self.prompt_model is not None:
                prompt = INSTRUCTION_PROPOSAL_TEMPLATE.format(
                    task_context=self.task_context,
                    examples=examples_str,
                )
                try:
                    response = self.prompt_model(prompt)
                    instruction = str(response).strip()
                    if instruction:
                        candidates.append(instruction)
                except Exception:
                    pass

            # Fallback: generate simple variations
            if len(candidates) <= i:
                base = f"Complete the task based on the given input. Variation {i+1}."
                candidates.append(base)

        return candidates

    def _sample_instruction(self) -> int:
        """Sample an instruction index using Thompson sampling."""
        if not self._instruction_scores:
            # Random sampling initially
            return self.rng.randint(0, len(self._instruction_candidates) - 1)

        # Calculate expected value for each instruction
        scores = []
        for i in range(len(self._instruction_candidates)):
            if i in self._instruction_scores and self._instruction_scores[i]:
                # Use mean + exploration bonus
                mean = sum(self._instruction_scores[i]) / len(self._instruction_scores[i])
                bonus = 1.0 / (1 + len(self._instruction_scores[i]))
                scores.append(mean + self.init_temperature * bonus)
            else:
                # High value for unexplored
                scores.append(1.0)

        # Softmax sampling
        total = sum(s for s in scores)
        probs = [s / total for s in scores] if total > 0 else [1.0 / len(scores)] * len(scores)

        r = self.rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i

        return len(self._instruction_candidates) - 1

    def _sample_demos(self) -> List[int]:
        """Sample demonstration indices using Thompson sampling."""
        if not self._demo_pool:
            return []

        if not self._demo_scores:
            # Random sampling initially
            indices = list(range(len(self._demo_pool)))
            return self.rng.sample(indices, min(self.num_demos, len(indices)))

        # Calculate expected value for each demo
        scores = []
        for i in range(len(self._demo_pool)):
            if i in self._demo_scores and self._demo_scores[i]:
                mean = sum(self._demo_scores[i]) / len(self._demo_scores[i])
                bonus = 1.0 / (1 + len(self._demo_scores[i]))
                scores.append(mean + self.init_temperature * bonus)
            else:
                scores.append(1.0)

        # Sample top-k with some randomness
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1] + self.rng.random() * 0.1, reverse=True)

        return [idx for idx, _ in indexed_scores[: self.num_demos]]

    def _evaluate_candidate(
        self,
        candidate: PromptCandidate,
        valset: List[Example],
        teacher: Module,
    ) -> float:
        """Evaluate a candidate configuration."""
        # Apply candidate temporarily
        instructions_param = None
        examples_param = None
        original_instructions = None
        original_examples = None

        for group in self.param_groups:
            for param in group["params"]:
                if param.spec == PromptSpec.INSTRUCTIONS:
                    instructions_param = param
                    original_instructions = param.data
                elif param.spec == PromptSpec.EXAMPLES:
                    examples_param = param
                    original_examples = param.data

        try:
            # Apply candidate
            if instructions_param and instructions_param.requires_grad:
                instructions_param.data = candidate.instruction

            if examples_param and examples_param.requires_grad and candidate.demos:
                examples_param.data = self._format_demos(candidate.demos)

            # Evaluate
            total_score = 0.0
            for example in valset:
                try:
                    prediction = teacher(example.inputs)
                    score = self.metric(example, prediction)
                    total_score += score
                except Exception:
                    pass

            return total_score / len(valset) if valset else 0.0

        finally:
            # Restore original values
            if instructions_param:
                instructions_param.data = original_instructions
            if examples_param:
                examples_param.data = original_examples

    def _format_demos(self, demos: List[Example]) -> str:
        """Format demonstrations as a string."""
        formatted = []
        for i, demo in enumerate(demos, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {demo.inputs}")
            formatted.append(f"Output: {demo.labels}")
            formatted.append("")
        return "\n".join(formatted)

    def _update_surrogate(self, trial: MiproTrial) -> None:
        """Update surrogate model with trial results."""
        # Update instruction scores
        if trial.instruction_idx not in self._instruction_scores:
            self._instruction_scores[trial.instruction_idx] = []
        self._instruction_scores[trial.instruction_idx].append(trial.score)

        # Update demo scores
        for demo_idx in trial.demo_indices:
            if demo_idx not in self._demo_scores:
                self._demo_scores[demo_idx] = []
            self._demo_scores[demo_idx].append(trial.score)

    def _apply_candidate(self, candidate: PromptCandidate) -> None:
        """Apply the best candidate to parameters."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.spec == PromptSpec.INSTRUCTIONS and param.requires_grad:
                    param.data = candidate.instruction
                elif param.spec == PromptSpec.EXAMPLES and param.requires_grad:
                    if candidate.demos:
                        param.data = self._format_demos(candidate.demos)

    def get_best_candidate(self) -> Optional[PromptCandidate]:
        """Get the best candidate found."""
        return self._best_candidate

    def get_best_score(self) -> float:
        """Get the best score achieved."""
        return self._best_score

    def get_trials(self) -> List[MiproTrial]:
        """Get all optimization trials."""
        return self._trials

    def get_instruction_candidates(self) -> List[str]:
        """Get all generated instruction candidates."""
        return self._instruction_candidates

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dictionary."""
        state = super().state_dict()
        state.update(
            {
                "instruction_candidates": self._instruction_candidates,
                "best_score": self._best_score,
                "seed": self.seed,
            }
        )
        if self._best_candidate:
            state["best_instruction"] = self._best_candidate.instruction
            state["best_demos"] = [
                {"inputs": d.inputs, "labels": d.labels}
                for d in self._best_candidate.demos
            ]
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state dictionary."""
        super().load_state_dict(state)
        self._instruction_candidates = state.get("instruction_candidates", [])
        self._best_score = state.get("best_score", 0.0)
        self.seed = state.get("seed", self.seed)
        self.rng = random.Random(self.seed)

        if "best_instruction" in state:
            demos = [
                Example(inputs=d["inputs"], labels=d["labels"])
                for d in state.get("best_demos", [])
            ]
            self._best_candidate = PromptCandidate(
                instruction=state["best_instruction"],
                demos=demos,
                score=self._best_score,
            )

    # =========================================================================
    # Async Methods
    # =========================================================================

    async def astep(
        self,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        *,
        teacher: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
        max_concurrency: Optional[int] = None,
    ) -> Optional[float]:
        """Perform one MIPROv2 optimization step asynchronously.

        This method runs optimization trials concurrently, which can significantly
        speed up optimization when using async-capable modules.

        Args:
            trainset: Training examples for demonstrations.
            valset: Validation examples for evaluation.
            teacher: Module to evaluate (must support acall or aforward).
            closure: Closure for loss computation.
            max_concurrency: Maximum concurrent evaluations. If None, runs
                all trials concurrently.

        Returns:
            The loss value if closure is provided, None otherwise.

        Example:
            >>> result = await optimizer.astep(trainset, valset, teacher=agent)
        """
        self._step_count += 1

        if valset is None:
            valset = trainset

        # Initialize demonstration pool
        self._demo_pool = trainset.copy()

        # Generate instruction candidates (sync operation)
        if not self._instruction_candidates:
            self._instruction_candidates = self._generate_instruction_candidates(
                trainset
            )

        # Run optimization trials asynchronously
        if max_concurrency:
            await self._arun_trials_with_semaphore(
                valset, teacher, max_concurrency
            )
        else:
            await self._arun_trials_concurrent(valset, teacher)

        # Apply best configuration to parameters
        if self._best_candidate is not None:
            self._apply_candidate(self._best_candidate)

        if closure is not None:
            return closure()
        return None

    async def _arun_trials_concurrent(
        self,
        valset: List[Example],
        teacher: Optional[Module],
    ) -> None:
        """Run all trials concurrently without limit."""
        tasks = [
            self._arun_trial(valset, teacher)
            for _ in range(self.num_trials)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _arun_trials_with_semaphore(
        self,
        valset: List[Example],
        teacher: Optional[Module],
        max_concurrency: int,
    ) -> None:
        """Run trials with limited concurrency using semaphore."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_trial():
            async with semaphore:
                return await self._arun_trial(valset, teacher)

        tasks = [bounded_trial() for _ in range(self.num_trials)]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _arun_trial(
        self,
        valset: List[Example],
        teacher: Optional[Module],
    ) -> MiproTrial:
        """Run a single optimization trial asynchronously."""
        # Sample instruction and demos based on surrogate predictions
        instruction_idx = self._sample_instruction()
        demo_indices = self._sample_demos()

        # Create candidate
        candidate = PromptCandidate(
            instruction=self._instruction_candidates[instruction_idx],
            demos=[self._demo_pool[i] for i in demo_indices if i < len(self._demo_pool)],
        )

        # Evaluate candidate asynchronously
        if teacher is not None:
            score = await self._aevaluate_candidate(candidate, valset, teacher)
        else:
            score = 0.0

        candidate.score = score
        candidate.evaluated = True

        # Record trial
        trial = MiproTrial(
            instruction_idx=instruction_idx,
            demo_indices=demo_indices,
            score=score,
            candidate=candidate,
        )
        self._trials.append(trial)

        # Update surrogate model
        self._update_surrogate(trial)

        # Track best
        if score > self._best_score:
            self._best_score = score
            self._best_candidate = candidate

        return trial

    async def _aevaluate_candidate(
        self,
        candidate: PromptCandidate,
        valset: List[Example],
        teacher: Module,
    ) -> float:
        """Evaluate a candidate configuration asynchronously."""
        # Apply candidate temporarily
        instructions_param = None
        examples_param = None
        original_instructions = None
        original_examples = None

        for group in self.param_groups:
            for param in group["params"]:
                if param.spec == PromptSpec.INSTRUCTIONS:
                    instructions_param = param
                    original_instructions = param.data
                elif param.spec == PromptSpec.EXAMPLES:
                    examples_param = param
                    original_examples = param.data

        try:
            # Apply candidate
            if instructions_param and instructions_param.requires_grad:
                instructions_param.data = candidate.instruction

            if examples_param and examples_param.requires_grad and candidate.demos:
                examples_param.data = self._format_demos(candidate.demos)

            # Evaluate all examples concurrently
            tasks = [
                self._aevaluate_example(teacher, example)
                for example in valset
            ]
            scores = await asyncio.gather(*tasks, return_exceptions=True)

            # Sum valid scores
            total_score = 0.0
            for score in scores:
                if isinstance(score, (int, float)):
                    total_score += score

            return total_score / len(valset) if valset else 0.0

        finally:
            # Restore original values
            if instructions_param:
                instructions_param.data = original_instructions
            if examples_param:
                examples_param.data = original_examples

    async def _aevaluate_example(
        self,
        teacher: Module,
        example: Example,
    ) -> float:
        """Evaluate a single example asynchronously."""
        try:
            # Run the module asynchronously
            if hasattr(teacher, "acall"):
                prediction = await teacher.acall(example.inputs)
            elif hasattr(teacher, "aforward"):
                prediction = await teacher.aforward(example.inputs)
            else:
                # Fallback to sync in executor
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None, teacher, example.inputs
                )

            # Score with metric (sync operation)
            return self.metric(example, prediction)

        except Exception:
            return 0.0
