"""SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer.

This module provides the SIMBA optimizer for prompt optimization, which uses
the LLM to analyze its own performance and generate improvement rules through
mini-batch stochastic gradient ascent with self-reflection.
"""

import asyncio
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.logger import init_logger
from msgflux.optim.optimizer import Optimizer
from msgflux.optim.progress import OptimProgress, TrialInfo
from msgflux.optim.simba.utils import (
    ExecutionResult,
    append_a_demo,
    append_a_rule,
    prepare_models_for_resampling,
    wrap_program,
)

logger = init_logger(__name__)


@dataclass
class SIMBACandidate:
    """Represents a candidate program in SIMBA optimization."""

    program: Module
    score: float = 0.0
    step: int = 0


@dataclass
class SIMBATrialLog:
    """Log entry for a SIMBA optimization trial."""

    batch_idx: int
    batch_scores: List[float] = field(default_factory=list)
    baseline_score: float = 0.0
    candidate_scores: List[float] = field(default_factory=list)
    best_candidate_score: float = 0.0
    strategy_used: str = ""
    train_score: Optional[float] = None


@dataclass
class SIMBAResult:
    """Result of SIMBA optimization."""

    program: Module
    score: float
    candidate_programs: List[SIMBACandidate]
    trial_logs: Dict[int, SIMBATrialLog]


class SIMBA(Optimizer):
    """SIMBA (Stochastic Introspective Mini-Batch Ascent) optimizer.

    SIMBA is an optimizer that uses the LLM to analyze its own performance
    and generate improvement rules. It samples mini-batches, identifies
    challenging examples with high output variability, then either creates
    self-reflective rules or adds successful examples as demonstrations.

    The algorithm works by:
    1. Sampling mini-batches from the training set
    2. Running multiple program variants to identify examples with high variance
    3. Applying strategies (add demo or generate rule) to improve performance
    4. Maintaining a pool of candidate programs with softmax sampling

    Args:
        params: An iterable of Parameters to optimize.
        metric: Function that takes (example, prediction) and returns a score.
        bsize: Mini-batch size for optimization. Defaults to 32.
        num_candidates: Number of candidate programs per iteration. Defaults to 6.
        max_steps: Number of optimization steps. Defaults to 8.
        max_demos: Maximum demos per predictor before dropping some. Defaults to 4.
        prompt_model: Model for generating reflective rules. If None, uses
            the globally configured model.
        teacher_settings: Optional settings for teacher model.
        demo_input_field_maxlen: Max characters in demo input fields. Defaults to 100000.
        temperature_for_sampling: Temperature for program sampling. Defaults to 0.2.
        temperature_for_candidates: Temperature for candidate selection. Defaults to 0.2.
        verbose: If True, display detailed progress during optimization.
        seed: Random seed for reproducibility.

    Example:
        >>> agent = Agent(name="qa", model=model)
        >>> optimizer = SIMBA(
        ...     agent.parameters(),
        ...     metric=lambda ex, pred: float(ex.labels.lower() in pred.lower()),
        ...     bsize=32,
        ...     max_steps=8,
        ... )
        >>> result = optimizer.step(trainset=train_examples, student=agent)
        >>> optimized_agent = result.program
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        metric: Callable[[Example, Any], float],
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        max_demos: int = 4,
        prompt_model: Optional[Module] = None,
        teacher_settings: Optional[Dict[str, Any]] = None,
        demo_input_field_maxlen: int = 100_000,
        temperature_for_sampling: float = 0.2,
        temperature_for_candidates: float = 0.2,
        verbose: bool = False,
        seed: int = 0,
    ):
        defaults = dict(
            bsize=bsize,
            num_candidates=num_candidates,
            max_steps=max_steps,
            max_demos=max_demos,
        )
        super().__init__(params, defaults)

        self.metric = metric
        self.bsize = bsize
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.prompt_model = prompt_model
        self.teacher_settings = teacher_settings
        self.demo_input_field_maxlen = demo_input_field_maxlen
        self.temperature_for_sampling = temperature_for_sampling
        self.temperature_for_candidates = temperature_for_candidates
        self.verbose = verbose
        self.seed = seed

        # Initialize RNG
        self.rng = random.Random(seed)

        # Setup strategies
        if self.max_demos > 0:
            self.strategies = [append_a_demo(demo_input_field_maxlen), append_a_rule]
        else:
            self.strategies = [append_a_rule]

        # Progress tracking
        self._progress = OptimProgress(verbose=verbose)

        # Internal state
        self._programs: List[Module] = []
        self._program_scores: Dict[int, List[float]] = {}
        self._next_program_idx: int = 0
        self._trial_logs: Dict[int, SIMBATrialLog] = {}
        self._winning_programs: List[Module] = []

    def step(
        self,
        trainset: List[Example],
        student: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
    ) -> SIMBAResult:
        """Execute SIMBA optimization step.

        Args:
            trainset: Training examples for optimization.
            student: The module to optimize.
            closure: Optional closure for computing loss.

        Returns:
            SIMBAResult with the optimized program and metadata.

        Raises:
            AssertionError: If trainset is smaller than bsize.
        """
        if closure is not None:
            closure()

        assert len(trainset) >= self.bsize, (
            f"Trainset too small: {len(trainset)} < {self.bsize}"
        )

        # Reset state
        self._reset_state()

        # Start progress tracking
        self._progress.start(
            "SIMBA",
            bsize=self.bsize,
            num_candidates=self.num_candidates,
            max_steps=self.max_steps,
            max_demos=self.max_demos,
            trainset_size=len(trainset),
        )

        # Initialize baseline program
        if student is None:
            raise ValueError("student module is required for SIMBA optimization")

        student = self._deepcopy_program(student)
        student._simba_idx = 0
        self._programs.append(student)
        self._program_scores[0] = []
        self._winning_programs = [student]

        # Shuffle data
        data_indices = list(range(len(trainset)))
        self.rng.shuffle(data_indices)
        instance_idx = 0

        best_overall_score = 0.0

        # Main optimization loop
        for batch_idx in range(self.max_steps):
            self._trial_logs[batch_idx] = SIMBATrialLog(batch_idx=batch_idx)

            self._progress.step(
                "MINI-BATCH OPTIMIZATION",
                batch_idx + 1,
                self.max_steps,
                f"Processing batch of {self.bsize} examples",
            )

            # Step 1: Get next batch
            if instance_idx + self.bsize > len(trainset):
                self.rng.shuffle(data_indices)
                instance_idx = 0

            batch_indices = data_indices[instance_idx:instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize

            # Step 2: Sample trajectories
            self._progress.substep(
                f"Sampling {self.bsize} x {self.num_candidates} trajectories..."
            )
            outputs = self._sample_trajectories(batch)

            # Step 3: Sort buckets by variability
            buckets = self._create_buckets(batch, outputs)

            # Compute baseline score
            all_scores = [o.score for o in outputs]
            baseline_score = sum(all_scores) / len(all_scores)
            self._trial_logs[batch_idx].baseline_score = baseline_score
            self._trial_logs[batch_idx].batch_scores = all_scores

            self._progress.metric("Baseline", baseline_score, 1.0)

            # Step 4: Build candidate programs
            self._progress.substep("Building candidate programs...")
            system_candidates = self._build_candidates(
                batch, buckets, outputs, batch_idx
            )

            # Step 5: Evaluate candidates
            self._progress.substep(f"Evaluating {len(system_candidates)} candidates...")
            candidate_scores = self._evaluate_candidates(system_candidates, batch)
            self._trial_logs[batch_idx].candidate_scores = candidate_scores

            if candidate_scores:
                best_score = max(candidate_scores)
                self._trial_logs[batch_idx].best_candidate_score = best_score
                is_best = best_score > best_overall_score
                if is_best:
                    best_overall_score = best_score

                self._progress.trial(TrialInfo(
                    trial_num=batch_idx + 1,
                    total_trials=self.max_steps,
                    score=best_score,
                    best_score=best_overall_score,
                    is_best=is_best,
                ))

            # Step 6: Select best and register candidates
            if candidate_scores:
                best_idx = candidate_scores.index(max(candidate_scores))
                best_program = self._deepcopy_program(system_candidates[best_idx])
                self._winning_programs.append(best_program)

            # Register all candidates
            for idx, cand in enumerate(system_candidates):
                self._register_program(cand, [candidate_scores[idx]] if candidate_scores else [])

        # Final validation
        self._progress.step(
            "FINAL VALIDATION",
            self.max_steps + 1,
            self.max_steps + 1,
            f"Evaluating candidates on full trainset ({len(trainset)} examples)",
        )
        result = self._final_validation(trainset)

        # Finish progress
        self._progress.finish(
            best_score=result.score,
            summary={
                "candidates_evaluated": len(result.candidate_programs),
                "total_programs_generated": len(self._programs),
                "optimization_steps": self.max_steps,
            },
        )

        self._step_count += 1
        return result

    async def astep(
        self,
        trainset: List[Example],
        student: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
        *,
        max_concurrency: Optional[int] = None,
    ) -> SIMBAResult:
        """Execute SIMBA optimization step asynchronously.

        Args:
            trainset: Training examples for optimization.
            student: The module to optimize.
            closure: Optional closure for computing loss.
            max_concurrency: Maximum concurrent evaluations.

        Returns:
            SIMBAResult with the optimized program and metadata.
        """
        if closure is not None:
            closure()

        assert len(trainset) >= self.bsize, (
            f"Trainset too small: {len(trainset)} < {self.bsize}"
        )

        # Reset state
        self._reset_state()

        if student is None:
            raise ValueError("student module is required for SIMBA optimization")

        student = self._deepcopy_program(student)
        student._simba_idx = 0
        self._programs.append(student)
        self._program_scores[0] = []
        self._winning_programs = [student]

        # Shuffle data
        data_indices = list(range(len(trainset)))
        self.rng.shuffle(data_indices)
        instance_idx = 0

        # Main optimization loop
        for batch_idx in range(self.max_steps):
            self._trial_logs[batch_idx] = SIMBATrialLog(batch_idx=batch_idx)

            logger.info(f"Starting async batch {batch_idx + 1} of {self.max_steps}")

            # Get next batch
            if instance_idx + self.bsize > len(trainset):
                self.rng.shuffle(data_indices)
                instance_idx = 0

            batch_indices = data_indices[instance_idx:instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize

            # Sample trajectories asynchronously
            outputs = await self._asample_trajectories(batch, max_concurrency)

            # Create buckets
            buckets = self._create_buckets(batch, outputs)

            # Compute baseline
            all_scores = [o.score for o in outputs]
            baseline_score = sum(all_scores) / len(all_scores)
            self._trial_logs[batch_idx].baseline_score = baseline_score
            self._trial_logs[batch_idx].batch_scores = all_scores

            # Build candidates
            system_candidates = self._build_candidates(
                batch, buckets, outputs, batch_idx
            )

            # Evaluate candidates asynchronously
            candidate_scores = await self._aevaluate_candidates(
                system_candidates, batch, max_concurrency
            )
            self._trial_logs[batch_idx].candidate_scores = candidate_scores

            if candidate_scores:
                best_idx = candidate_scores.index(max(candidate_scores))
                best_program = self._deepcopy_program(system_candidates[best_idx])
                self._winning_programs.append(best_program)

            for idx, cand in enumerate(system_candidates):
                self._register_program(cand, [candidate_scores[idx]] if candidate_scores else [])

        # Final validation
        result = await self._afinal_validation(trainset, max_concurrency)

        self._step_count += 1
        return result

    def _reset_state(self) -> None:
        """Reset internal state for a new optimization run."""
        self._programs = []
        self._program_scores = {}
        self._next_program_idx = 0
        self._trial_logs = {}
        self._winning_programs = []

    def _deepcopy_program(self, program: Module) -> Module:
        """Create a deep copy of a program."""
        try:
            return deepcopy(program)
        except Exception:
            # Fallback for modules that don't support deepcopy
            return program

    def _calc_average_score(self, prog_idx: int) -> float:
        """Calculate average score for a program."""
        scores = self._program_scores.get(prog_idx, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _top_k_plus_baseline(self, k: int) -> List[int]:
        """Get top k program indices plus baseline."""
        scored = sorted(
            self._programs,
            key=lambda p: self._calc_average_score(getattr(p, "_simba_idx", 0)),
            reverse=True,
        )
        top_k = [getattr(p, "_simba_idx", 0) for p in scored[:k]]

        # Ensure baseline (0) is included
        if 0 not in top_k and top_k:
            top_k[-1] = 0

        return list(dict.fromkeys(top_k))

    def _softmax_sample(self, program_idxs: List[int], temperature: float) -> int:
        """Sample a program index using softmax weighting."""
        if not program_idxs:
            raise ValueError("No programs available for sampling")

        scores = [self._calc_average_score(idx) for idx in program_idxs]
        exps = [math.exp(s / temperature) for s in scores]
        sum_exps = sum(exps)

        if sum_exps <= 0:
            return self.rng.choice(program_idxs)

        probs = [val / sum_exps for val in exps]
        return self.rng.choices(program_idxs, weights=probs, k=1)[0]

    def _register_program(self, prog: Module, scores: List[float]) -> None:
        """Register a new program in the pool."""
        self._next_program_idx += 1
        prog._simba_idx = self._next_program_idx
        self._programs.append(prog)
        self._program_scores[self._next_program_idx] = scores

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data (pure Python implementation)."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        k = (n - 1) * percentile / 100.0
        f = int(k)
        c = f + 1 if f + 1 < n else f
        if f == c:
            return float(sorted_data[f])
        return float(sorted_data[f] * (c - k) + sorted_data[c] * (k - f))

    def _poisson(self, lam: float) -> int:
        """Generate Poisson-distributed random number (Knuth's algorithm)."""
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= self.rng.random()
        return k - 1

    def _sample_trajectories(
        self,
        batch: List[Example],
    ) -> List[ExecutionResult]:
        """Sample trajectories for a batch of examples."""
        outputs: List[ExecutionResult] = []
        models = prepare_models_for_resampling(
            self._programs[0], self.num_candidates, self.teacher_settings
        )
        top_programs = self._top_k_plus_baseline(self.num_candidates)

        logger.info(
            f"Sampling trajectories: {self.bsize} examples x {self.num_candidates} candidates"
        )

        predictor2name: Dict[int, str] = {}

        for model in models:
            for example in batch:
                # Select program via softmax
                prog_idx = self._softmax_sample(top_programs, self.temperature_for_sampling)
                candidate = self._deepcopy_program(self._programs[prog_idx])

                # Set model if provided
                if model is not None and hasattr(candidate, "set_model"):
                    candidate.set_model(model)

                # Track predictor names
                for name, mod in candidate.named_modules():
                    predictor2name[id(mod)] = name

                # Execute wrapped program
                wrapped = wrap_program(candidate, self.metric)
                result = wrapped(example)
                outputs.append(result)

        return outputs

    async def _asample_trajectories(
        self,
        batch: List[Example],
        max_concurrency: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """Sample trajectories asynchronously."""
        models = prepare_models_for_resampling(
            self._programs[0], self.num_candidates, self.teacher_settings
        )
        top_programs = self._top_k_plus_baseline(self.num_candidates)

        tasks = []
        for model in models:
            for example in batch:
                prog_idx = self._softmax_sample(top_programs, self.temperature_for_sampling)
                candidate = self._deepcopy_program(self._programs[prog_idx])

                if model is not None and hasattr(candidate, "set_model"):
                    candidate.set_model(model)

                tasks.append((candidate, example))

        if max_concurrency:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def bounded_execute(prog, ex):
                async with semaphore:
                    return await self._aexecute_one(prog, ex)

            results = await asyncio.gather(
                *[bounded_execute(p, e) for p, e in tasks],
                return_exceptions=True,
            )
        else:
            results = await asyncio.gather(
                *[self._aexecute_one(p, e) for p, e in tasks],
                return_exceptions=True,
            )

        outputs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Trajectory sampling error: {result}")
                outputs.append(ExecutionResult(
                    prediction=None,
                    trace=[],
                    score=0.0,
                    example=tasks[i][1],
                    output_metadata={},
                ))
            else:
                outputs.append(result)

        return outputs

    async def _aexecute_one(
        self,
        program: Module,
        example: Example,
    ) -> ExecutionResult:
        """Execute a single program asynchronously."""
        wrapped = wrap_program(program, self.metric)

        # Check for async execution capability
        if hasattr(program, "acall"):
            prediction = await program.acall(example.inputs)
            score = self.metric(example, prediction)
            return ExecutionResult(
                prediction=prediction,
                trace=[],
                score=score,
                example=example,
                output_metadata={},
            )
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, wrapped, example)

    def _create_buckets(
        self,
        batch: List[Example],
        outputs: List[ExecutionResult],
    ) -> List[Tuple[List[ExecutionResult], Tuple[float, float, float]]]:
        """Create and sort buckets by score variability."""
        buckets = []

        for idx in range(len(batch)):
            # Gather results for this example
            bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
            bucket.sort(key=lambda x: x.score, reverse=True)

            if not bucket:
                continue

            max_score = bucket[0].score
            min_score = bucket[-1].score
            avg_score = sum(x.score for x in bucket) / len(bucket)

            max_to_min_gap = max_score - min_score
            max_to_avg_gap = max_score - avg_score

            buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))

        # Sort by variability metrics
        buckets.sort(key=lambda x: x[1], reverse=True)
        return buckets

    def _build_candidates(
        self,
        batch: List[Example],
        buckets: List[Tuple[List[ExecutionResult], Tuple[float, float, float]]],
        outputs: List[ExecutionResult],
        batch_idx: int,
    ) -> List[Module]:
        """Build candidate programs by applying strategies."""
        system_candidates: List[Module] = []

        # Compute batch percentiles
        all_scores = [o.score for o in outputs]
        batch_10p = self._percentile(all_scores, 10)
        batch_90p = self._percentile(all_scores, 90)

        # Build predictor name mappings
        predictor2name: Dict[int, str] = {}
        for prog in self._programs[:1]:  # Use first program as reference
            for name, mod in prog.named_modules():
                predictor2name[id(mod)] = name

        for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
            max_to_min_gap, max_score, max_to_avg_gap = bucket_stats

            logger.debug(
                f"Bucket {bucket_idx + 1}: max={max_score:.3f}, "
                f"gap={max_to_min_gap:.3f}, avg_gap={max_to_avg_gap:.3f}"
            )

            # Select source program
            src_idx = self._softmax_sample(
                self._top_k_plus_baseline(self.num_candidates),
                self.temperature_for_candidates,
            )
            system_candidate = self._deepcopy_program(self._programs[src_idx])

            # Build name2predictor for this candidate
            name2predictor: Dict[str, Any] = {}
            for name, mod in system_candidate.named_modules():
                name2predictor[name] = mod
                predictor2name[id(mod)] = name

            # Drop some demos randomly
            self._drop_demos(system_candidate)

            # Choose and apply strategy
            strategy = self.rng.choice(self.strategies)
            strategy_name = getattr(strategy, "__name__", "unknown")
            self._trial_logs[batch_idx].strategy_used = strategy_name

            logger.info(f"Applying strategy: {strategy_name}")

            try:
                strategy(
                    bucket,
                    system_candidate,
                    predictor2name=predictor2name,
                    name2predictor=name2predictor,
                    batch_10p_score=batch_10p,
                    batch_90p_score=batch_90p,
                    prompt_model=self.prompt_model,
                )
            except Exception as e:
                logger.error(f"Strategy failed: {e}")
                continue

            system_candidates.append(system_candidate)

            if len(system_candidates) >= self.num_candidates + 1:
                break

        return system_candidates

    def _drop_demos(self, system: Module) -> None:
        """Randomly drop some demos from predictors."""
        max_demos_tmp = self.max_demos if self.max_demos > 0 else 3
        num_demos_list = []

        # Collect demo counts
        for name, mod in system.named_modules():
            if hasattr(mod, "demos"):
                num_demos_list.append(len(mod.demos))
            elif hasattr(mod, "_demos"):
                num_demos_list.append(len(mod._demos))

        if not num_demos_list:
            return

        num_demos = max(num_demos_list)
        num_to_drop = max(
            self._poisson(num_demos / max_demos_tmp),
            int(num_demos >= max_demos_tmp),
        )
        num_to_drop = min(num_to_drop, num_demos)

        if num_to_drop == 0:
            return

        indices_to_drop = set(self.rng.sample(range(num_demos), min(num_to_drop, num_demos)))

        # Drop demos
        for name, mod in system.named_modules():
            if hasattr(mod, "demos"):
                mod.demos = [d for i, d in enumerate(mod.demos) if i not in indices_to_drop]
            elif hasattr(mod, "_demos"):
                mod._demos = [d for i, d in enumerate(mod._demos) if i not in indices_to_drop]

    def _evaluate_candidates(
        self,
        candidates: List[Module],
        batch: List[Example],
    ) -> List[float]:
        """Evaluate candidate programs on a batch."""
        if not candidates:
            return []

        logger.info(f"Evaluating {len(candidates)} candidates on {len(batch)} examples")

        scores: List[float] = []

        for candidate in candidates:
            wrapped = wrap_program(candidate, self.metric)
            candidate_scores = []

            for example in batch:
                result = wrapped(example)
                candidate_scores.append(result.score)

            avg_score = sum(candidate_scores) / len(candidate_scores)
            scores.append(avg_score)

        return scores

    async def _aevaluate_candidates(
        self,
        candidates: List[Module],
        batch: List[Example],
        max_concurrency: Optional[int] = None,
    ) -> List[float]:
        """Evaluate candidates asynchronously."""
        if not candidates:
            return []

        tasks = [(c, e) for c in candidates for e in batch]

        if max_concurrency:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def bounded_eval(prog, ex):
                async with semaphore:
                    return await self._aexecute_one(prog, ex)

            results = await asyncio.gather(
                *[bounded_eval(p, e) for p, e in tasks],
                return_exceptions=True,
            )
        else:
            results = await asyncio.gather(
                *[self._aexecute_one(p, e) for p, e in tasks],
                return_exceptions=True,
            )

        # Compute average scores per candidate
        scores = []
        batch_size = len(batch)

        for idx in range(len(candidates)):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            candidate_results = results[start:end]

            candidate_scores = []
            for r in candidate_results:
                if isinstance(r, Exception):
                    candidate_scores.append(0.0)
                else:
                    candidate_scores.append(r.score)

            avg = sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0.0
            scores.append(avg)

        return scores

    def _final_validation(self, trainset: List[Example]) -> SIMBAResult:
        """Final validation on full trainset."""
        M = len(self._winning_programs) - 1
        N = self.num_candidates + 1

        if M < 1:
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]

        program_idxs = list(dict.fromkeys(program_idxs))
        candidate_programs = [
            self._deepcopy_program(self._winning_programs[i])
            for i in program_idxs
        ]

        logger.info(
            f"Final validation: {len(candidate_programs)} programs on {len(trainset)} examples"
        )

        scores = []
        for prog in candidate_programs:
            wrapped = wrap_program(prog, self.metric)
            prog_scores = [wrapped(ex).score for ex in trainset]
            avg = sum(prog_scores) / len(prog_scores) if prog_scores else 0.0
            scores.append(avg)

        # Update trial logs with final scores
        for idx, score in enumerate(scores[1:], start=0):
            if idx in self._trial_logs:
                self._trial_logs[idx].train_score = score

        # Find best program
        best_idx = scores.index(max(scores)) if scores else 0
        best_program = self._deepcopy_program(candidate_programs[best_idx])
        best_score = scores[best_idx] if scores else 0.0

        logger.info(f"Best program: index {best_idx}, score {best_score:.4f}")

        # Build result
        candidates = [
            SIMBACandidate(program=p, score=s)
            for p, s in zip(candidate_programs, scores)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        return SIMBAResult(
            program=best_program,
            score=best_score,
            candidate_programs=candidates,
            trial_logs=self._trial_logs,
        )

    async def _afinal_validation(
        self,
        trainset: List[Example],
        max_concurrency: Optional[int] = None,
    ) -> SIMBAResult:
        """Final validation asynchronously."""
        M = len(self._winning_programs) - 1
        N = self.num_candidates + 1

        if M < 1:
            program_idxs = [0] * N
        else:
            program_idxs = [round(i * M / (N - 1)) for i in range(N)]

        program_idxs = list(dict.fromkeys(program_idxs))
        candidate_programs = [
            self._deepcopy_program(self._winning_programs[i])
            for i in program_idxs
        ]

        scores = await self._aevaluate_candidates(
            candidate_programs, trainset, max_concurrency
        )

        for idx, score in enumerate(scores[1:], start=0):
            if idx in self._trial_logs:
                self._trial_logs[idx].train_score = score

        best_idx = scores.index(max(scores)) if scores else 0
        best_program = self._deepcopy_program(candidate_programs[best_idx])
        best_score = scores[best_idx] if scores else 0.0

        candidates = [
            SIMBACandidate(program=p, score=s)
            for p, s in zip(candidate_programs, scores)
        ]
        candidates.sort(key=lambda c: c.score, reverse=True)

        return SIMBAResult(
            program=best_program,
            score=best_score,
            candidate_programs=candidates,
            trial_logs=self._trial_logs,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state as a dict."""
        state = super().state_dict()
        state.update({
            "seed": self.seed,
            "next_program_idx": self._next_program_idx,
            "trial_logs": {
                k: {
                    "batch_idx": v.batch_idx,
                    "baseline_score": v.baseline_score,
                    "best_candidate_score": v.best_candidate_score,
                    "strategy_used": v.strategy_used,
                    "train_score": v.train_score,
                }
                for k, v in self._trial_logs.items()
            },
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        super().load_state_dict(state_dict)

        if "seed" in state_dict:
            self.seed = state_dict["seed"]
            self.rng = random.Random(self.seed)

        if "next_program_idx" in state_dict:
            self._next_program_idx = state_dict["next_program_idx"]
