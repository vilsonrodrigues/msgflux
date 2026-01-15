"""GEPA - Genetic Evolutionary Prompt Algorithm.

This optimizer uses genetic algorithms to evolve prompts through
selection, crossover, and mutation operations.
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


# Template for mutation
MUTATION_TEMPLATE = """You are an expert prompt engineer. Your task is to modify an instruction to potentially improve it.

## Original Instruction
{instruction}

## Mutation Type: {mutation_type}

Apply the mutation type to create a new version of the instruction:
- "rephrase": Reword while keeping the same meaning
- "expand": Add more detail or clarification
- "simplify": Make more concise and direct
- "focus": Emphasize a specific aspect
- "generalize": Make more broadly applicable

Generate the mutated instruction (output only the new instruction, no explanation):"""


# Template for crossover
CROSSOVER_TEMPLATE = """You are an expert prompt engineer. Your task is to combine two instructions into a new, improved instruction.

## Instruction A
{instruction_a}

## Instruction B
{instruction_b}

Combine the best aspects of both instructions into a single, coherent instruction.
Generate the combined instruction (output only the new instruction, no explanation):"""


@dataclass
class Individual:
    """An individual in the genetic population."""

    instruction: str
    demos: List[Example] = field(default_factory=list)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.instruction)


@dataclass
class GEPAStats:
    """Statistics for GEPA optimization."""

    generation: int
    best_fitness: float
    avg_fitness: float
    population_size: int
    num_mutations: int
    num_crossovers: int


class GEPA(Optimizer):
    """Genetic Evolutionary Prompt Algorithm.

    GEPA evolves prompts through:
    1. Population initialization
    2. Fitness evaluation
    3. Selection (tournament or roulette)
    4. Crossover (combining instructions)
    5. Mutation (modifying instructions)

    Example:
        >>> from msgflux.optim import GEPA
        >>> from msgflux.evaluate.metrics import exact_match
        >>>
        >>> optimizer = GEPA(
        ...     agent.parameters(),
        ...     metric=exact_match,
        ...     prompt_model=generator_model,
        ...     population_size=20,
        ...     num_generations=10,
        ...     seed=42,
        ... )
        >>> optimizer.step(trainset, valset)

    Args:
        params: Iterable of Parameters to optimize.
        metric: Metric function for evaluation.
        prompt_model: Module for mutation and crossover operations.
        population_size: Size of the population.
        num_generations: Number of evolution generations.
        mutation_rate: Probability of mutation.
        crossover_rate: Probability of crossover.
        tournament_size: Size of tournament for selection.
        elite_size: Number of top individuals to preserve.
        seed: Random seed for reproducibility.
    """

    MUTATION_TYPES = ["rephrase", "expand", "simplify", "focus", "generalize"]

    def __init__(
        self,
        params: Iterable[Parameter],
        *,
        metric: Callable[[Example, Any], float],
        prompt_model: Optional[Module] = None,
        population_size: int = 20,
        num_generations: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elite_size: int = 2,
        seed: int = 0,
    ):
        defaults = dict(
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            tournament_size=tournament_size,
            elite_size=elite_size,
            seed=seed,
        )
        super().__init__(params, defaults)

        self.metric = metric
        self.prompt_model = prompt_model
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.seed = seed
        self.rng = random.Random(seed)

        # Track evolution state
        self._population: List[Individual] = []
        self._current_generation: int = 0
        self._best_individual: Optional[Individual] = None
        self._best_fitness: float = 0.0
        self._stats_history: List[GEPAStats] = []

    def step(
        self,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        *,
        teacher: Optional[Module] = None,
        closure: Optional[Callable[[], float]] = None,
    ) -> Optional[float]:
        """Perform one GEPA optimization step.

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

        # Initialize population if needed
        if not self._population:
            self._initialize_population(trainset)

        # Evolve for specified generations
        for gen in range(self.num_generations):
            self._current_generation += 1

            # Evaluate fitness
            self._evaluate_population(valset, teacher)

            # Record statistics
            self._record_stats()

            # Selection, crossover, and mutation
            new_population = self._evolve_population(trainset)
            self._population = new_population

        # Apply best individual to parameters
        if self._best_individual is not None:
            self._apply_individual(self._best_individual)

        if closure is not None:
            return closure()
        return None

    def _initialize_population(self, trainset: List[Example]) -> None:
        """Initialize the population with diverse individuals."""
        # Get current instruction as base
        base_instruction = ""
        for group in self.param_groups:
            for param in group["params"]:
                if param.spec == PromptSpec.INSTRUCTIONS:
                    base_instruction = param.data or ""
                    break

        # Create initial population
        self._population = []

        # Add base instruction
        if base_instruction:
            self._population.append(
                Individual(
                    instruction=base_instruction,
                    demos=self._sample_demos(trainset),
                    generation=0,
                )
            )

        # Generate variations
        variations = [
            "Complete the task step by step.",
            "Analyze the input carefully and provide the correct output.",
            "Follow the examples and apply the same pattern.",
            "Think through the problem and give a precise answer.",
            "Process the input according to the given format.",
        ]

        for i, var in enumerate(variations):
            if len(self._population) >= self.population_size:
                break
            self._population.append(
                Individual(
                    instruction=var,
                    demos=self._sample_demos(trainset),
                    generation=0,
                )
            )

        # Fill remaining with mutations if prompt model available
        while len(self._population) < self.population_size:
            if self._population and self.prompt_model is not None:
                parent = self.rng.choice(self._population)
                mutated = self._mutate(parent)
                mutated.generation = 0
                self._population.append(mutated)
            else:
                # Generate simple variation
                idx = len(self._population)
                self._population.append(
                    Individual(
                        instruction=f"Complete the task correctly. Approach {idx}.",
                        demos=self._sample_demos(trainset),
                        generation=0,
                    )
                )

    def _sample_demos(self, trainset: List[Example], k: int = 4) -> List[Example]:
        """Sample k demonstrations from trainset."""
        if not trainset:
            return []
        return self.rng.sample(trainset, min(k, len(trainset)))

    def _evaluate_population(
        self, valset: List[Example], teacher: Optional[Module]
    ) -> None:
        """Evaluate fitness of all individuals."""
        for individual in self._population:
            if teacher is not None:
                individual.fitness = self._evaluate_individual(
                    individual, valset, teacher
                )

            # Track best
            if individual.fitness > self._best_fitness:
                self._best_fitness = individual.fitness
                self._best_individual = individual

    def _evaluate_individual(
        self,
        individual: Individual,
        valset: List[Example],
        teacher: Module,
    ) -> float:
        """Evaluate a single individual."""
        # Apply individual temporarily
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
            # Apply individual
            if instructions_param and instructions_param.requires_grad:
                instructions_param.data = individual.instruction

            if examples_param and examples_param.requires_grad and individual.demos:
                examples_param.data = self._format_demos(individual.demos)

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

    def _evolve_population(self, trainset: List[Example]) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation."""
        new_population: List[Individual] = []
        num_mutations = 0
        num_crossovers = 0

        # Elitism: preserve best individuals
        sorted_pop = sorted(self._population, key=lambda x: x.fitness, reverse=True)
        for i in range(min(self.elite_size, len(sorted_pop))):
            elite = Individual(
                instruction=sorted_pop[i].instruction,
                demos=sorted_pop[i].demos.copy(),
                fitness=sorted_pop[i].fitness,
                generation=self._current_generation,
                parent_ids=[id(sorted_pop[i])],
            )
            new_population.append(elite)

        # Fill rest of population
        while len(new_population) < self.population_size:
            if self.rng.random() < self.crossover_rate and len(self._population) >= 2:
                # Crossover
                parent_a = self._select_parent()
                parent_b = self._select_parent()
                offspring = self._crossover(parent_a, parent_b, trainset)
                new_population.append(offspring)
                num_crossovers += 1
            else:
                # Mutation
                parent = self._select_parent()
                offspring = self._mutate(parent)
                offspring.demos = self._sample_demos(trainset)
                offspring.generation = self._current_generation
                new_population.append(offspring)
                num_mutations += 1

        return new_population

    def _select_parent(self) -> Individual:
        """Select a parent using tournament selection."""
        if len(self._population) <= self.tournament_size:
            return max(self._population, key=lambda x: x.fitness)

        tournament = self.rng.sample(self._population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(
        self,
        parent_a: Individual,
        parent_b: Individual,
        trainset: List[Example],
    ) -> Individual:
        """Create offspring through crossover."""
        if self.prompt_model is not None:
            prompt = CROSSOVER_TEMPLATE.format(
                instruction_a=parent_a.instruction,
                instruction_b=parent_b.instruction,
            )
            try:
                response = self.prompt_model(prompt)
                new_instruction = str(response).strip()
            except Exception:
                # Fallback: simple concatenation
                new_instruction = f"{parent_a.instruction} Additionally, {parent_b.instruction.lower()}"
        else:
            # Simple crossover: take first half of A and second half of B
            words_a = parent_a.instruction.split()
            words_b = parent_b.instruction.split()
            mid_a = len(words_a) // 2
            mid_b = len(words_b) // 2
            new_instruction = " ".join(words_a[:mid_a] + words_b[mid_b:])

        # Combine demos from both parents
        combined_demos = parent_a.demos + parent_b.demos
        unique_demos = list({d.inputs: d for d in combined_demos}.values())
        selected_demos = self.rng.sample(unique_demos, min(4, len(unique_demos)))

        return Individual(
            instruction=new_instruction,
            demos=selected_demos,
            generation=self._current_generation,
            parent_ids=[id(parent_a), id(parent_b)],
        )

    def _mutate(self, parent: Individual) -> Individual:
        """Create offspring through mutation."""
        mutation_type = self.rng.choice(self.MUTATION_TYPES)

        if self.prompt_model is not None:
            prompt = MUTATION_TEMPLATE.format(
                instruction=parent.instruction,
                mutation_type=mutation_type,
            )
            try:
                response = self.prompt_model(prompt)
                new_instruction = str(response).strip()
            except Exception:
                new_instruction = parent.instruction + " (mutated)"
        else:
            # Simple mutations without LLM
            new_instruction = self._simple_mutate(parent.instruction, mutation_type)

        return Individual(
            instruction=new_instruction,
            demos=parent.demos.copy(),
            generation=self._current_generation,
            parent_ids=[id(parent)],
        )

    def _simple_mutate(self, instruction: str, mutation_type: str) -> str:
        """Perform simple mutations without LLM."""
        if mutation_type == "rephrase":
            # Swap some words
            words = instruction.split()
            if len(words) > 2:
                i, j = self.rng.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
            return " ".join(words)

        elif mutation_type == "expand":
            return instruction + " Be thorough and precise."

        elif mutation_type == "simplify":
            words = instruction.split()
            if len(words) > 3:
                return " ".join(words[: len(words) * 2 // 3])
            return instruction

        elif mutation_type == "focus":
            return "Focus carefully: " + instruction

        elif mutation_type == "generalize":
            return instruction.replace("the ", "any ").replace("this ", "each ")

        return instruction

    def _record_stats(self) -> None:
        """Record statistics for the current generation."""
        if not self._population:
            return

        fitnesses = [ind.fitness for ind in self._population]
        stats = GEPAStats(
            generation=self._current_generation,
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            population_size=len(self._population),
            num_mutations=0,
            num_crossovers=0,
        )
        self._stats_history.append(stats)

    def _apply_individual(self, individual: Individual) -> None:
        """Apply the best individual to parameters."""
        for group in self.param_groups:
            for param in group["params"]:
                if param.spec == PromptSpec.INSTRUCTIONS and param.requires_grad:
                    param.data = individual.instruction
                elif param.spec == PromptSpec.EXAMPLES and param.requires_grad:
                    if individual.demos:
                        param.data = self._format_demos(individual.demos)

    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual found."""
        return self._best_individual

    def get_best_fitness(self) -> float:
        """Get the best fitness achieved."""
        return self._best_fitness

    def get_population(self) -> List[Individual]:
        """Get the current population."""
        return self._population

    def get_stats_history(self) -> List[GEPAStats]:
        """Get the statistics history."""
        return self._stats_history

    def get_current_generation(self) -> int:
        """Get the current generation number."""
        return self._current_generation

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dictionary."""
        state = super().state_dict()
        state.update(
            {
                "current_generation": self._current_generation,
                "best_fitness": self._best_fitness,
                "seed": self.seed,
            }
        )
        if self._best_individual:
            state["best_instruction"] = self._best_individual.instruction
            state["best_demos"] = [
                {"inputs": d.inputs, "labels": d.labels}
                for d in self._best_individual.demos
            ]
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state dictionary."""
        super().load_state_dict(state)
        self._current_generation = state.get("current_generation", 0)
        self._best_fitness = state.get("best_fitness", 0.0)
        self.seed = state.get("seed", self.seed)
        self.rng = random.Random(self.seed)

        if "best_instruction" in state:
            demos = [
                Example(inputs=d["inputs"], labels=d["labels"])
                for d in state.get("best_demos", [])
            ]
            self._best_individual = Individual(
                instruction=state["best_instruction"],
                demos=demos,
                fitness=self._best_fitness,
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
        """Perform one GEPA optimization step asynchronously.

        This method evaluates population fitness concurrently, which can
        significantly speed up optimization when using async-capable modules.

        Args:
            trainset: Training examples for demonstrations.
            valset: Validation examples for evaluation.
            teacher: Module to evaluate (must support acall or aforward).
            closure: Closure for loss computation.
            max_concurrency: Maximum concurrent evaluations per generation.
                If None, evaluates all individuals concurrently.

        Returns:
            The loss value if closure is provided, None otherwise.

        Example:
            >>> result = await optimizer.astep(trainset, valset, teacher=agent)
        """
        self._step_count += 1

        if valset is None:
            valset = trainset

        # Initialize population if needed
        if not self._population:
            self._initialize_population(trainset)

        # Evolve for specified generations
        for gen in range(self.num_generations):
            self._current_generation += 1

            # Evaluate fitness asynchronously
            await self._aevaluate_population(valset, teacher, max_concurrency)

            # Record statistics
            self._record_stats()

            # Selection, crossover, and mutation (sync operations)
            new_population = self._evolve_population(trainset)
            self._population = new_population

        # Apply best individual to parameters
        if self._best_individual is not None:
            self._apply_individual(self._best_individual)

        if closure is not None:
            return closure()
        return None

    async def _aevaluate_population(
        self,
        valset: List[Example],
        teacher: Optional[Module],
        max_concurrency: Optional[int] = None,
    ) -> None:
        """Evaluate fitness of all individuals asynchronously."""
        if teacher is None:
            return

        if max_concurrency:
            await self._aevaluate_population_with_semaphore(
                valset, teacher, max_concurrency
            )
        else:
            await self._aevaluate_population_concurrent(valset, teacher)

    async def _aevaluate_population_concurrent(
        self,
        valset: List[Example],
        teacher: Module,
    ) -> None:
        """Evaluate all individuals concurrently without limit."""
        tasks = [
            self._aevaluate_individual(individual, valset, teacher)
            for individual in self._population
        ]
        fitnesses = await asyncio.gather(*tasks, return_exceptions=True)

        # Update fitness and track best
        for individual, fitness in zip(self._population, fitnesses):
            if isinstance(fitness, (int, float)):
                individual.fitness = fitness
                if fitness > self._best_fitness:
                    self._best_fitness = fitness
                    self._best_individual = individual

    async def _aevaluate_population_with_semaphore(
        self,
        valset: List[Example],
        teacher: Module,
        max_concurrency: int,
    ) -> None:
        """Evaluate individuals with limited concurrency using semaphore."""
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_evaluate(individual: Individual) -> Tuple[Individual, float]:
            async with semaphore:
                fitness = await self._aevaluate_individual(individual, valset, teacher)
                return individual, fitness

        tasks = [bounded_evaluate(ind) for ind in self._population]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update fitness and track best
        for result in results:
            if isinstance(result, tuple):
                individual, fitness = result
                individual.fitness = fitness
                if fitness > self._best_fitness:
                    self._best_fitness = fitness
                    self._best_individual = individual

    async def _aevaluate_individual(
        self,
        individual: Individual,
        valset: List[Example],
        teacher: Module,
    ) -> float:
        """Evaluate a single individual asynchronously."""
        # Apply individual temporarily
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
            # Apply individual
            if instructions_param and instructions_param.requires_grad:
                instructions_param.data = individual.instruction

            if examples_param and examples_param.requires_grad and individual.demos:
                examples_param.data = self._format_demos(individual.demos)

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
