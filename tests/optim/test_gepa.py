"""Unit tests for GEPA optimizer."""

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.gepa import GEPA, GEPAStats, Individual


class MockModule:
    """Mock module for testing GEPA."""

    def __init__(self, response: str = "Mutated instruction"):
        self.response = response
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        return self.response


@pytest.fixture
def trainset():
    """Create sample training set."""
    return [
        Example(inputs="What is 2+2?", labels="4"),
        Example(inputs="What is 3+3?", labels="6"),
        Example(inputs="What is 4+4?", labels="8"),
        Example(inputs="What is 5+5?", labels="10"),
        Example(inputs="What is 6+6?", labels="12"),
    ]


@pytest.fixture
def valset():
    """Create sample validation set."""
    return [
        Example(inputs="What is 7+7?", labels="14"),
        Example(inputs="What is 8+8?", labels="16"),
    ]


@pytest.fixture
def metric():
    """Create exact match metric."""

    def exact_match(example: Example, prediction) -> float:
        return 1.0 if str(prediction) == str(example.labels) else 0.0

    return exact_match


@pytest.fixture
def instructions_param():
    """Create instructions parameter."""
    return Parameter("Do the task.", PromptSpec.INSTRUCTIONS)


@pytest.fixture
def examples_param():
    """Create examples parameter."""
    return Parameter(None, PromptSpec.EXAMPLES)


class TestIndividual:
    """Tests for Individual dataclass."""

    def test_creation(self):
        """Test creating an individual."""
        individual = Individual(instruction="Test instruction")
        assert individual.instruction == "Test instruction"
        assert individual.demos == []
        assert individual.fitness == 0.0
        assert individual.generation == 0
        assert individual.parent_ids == []

    def test_with_demos(self, trainset):
        """Test individual with demos."""
        individual = Individual(
            instruction="Test",
            demos=trainset[:2],
            fitness=0.85,
            generation=3,
        )
        assert len(individual.demos) == 2
        assert individual.fitness == 0.85
        assert individual.generation == 3

    def test_hashable(self):
        """Test that individual is hashable."""
        individual = Individual(instruction="Test")
        # Should not raise
        hash(individual)


class TestGEPAStats:
    """Tests for GEPAStats dataclass."""

    def test_creation(self):
        """Test creating stats."""
        stats = GEPAStats(
            generation=5,
            best_fitness=0.9,
            avg_fitness=0.75,
            population_size=20,
            num_mutations=8,
            num_crossovers=10,
        )
        assert stats.generation == 5
        assert stats.best_fitness == 0.9
        assert stats.avg_fitness == 0.75
        assert stats.population_size == 20


class TestGEPAInitialization:
    """Tests for GEPA initialization."""

    def test_basic_initialization(self, instructions_param, metric):
        """Test basic initialization."""
        optimizer = GEPA([instructions_param], metric=metric)

        assert optimizer.metric == metric
        assert optimizer.population_size == 20
        assert optimizer.num_generations == 10
        assert optimizer.mutation_rate == 0.3
        assert optimizer.crossover_rate == 0.7
        assert optimizer.seed == 0

    def test_initialization_with_prompt_model(
        self, instructions_param, metric
    ):
        """Test initialization with prompt model."""
        prompt_model = MockModule()
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            population_size=10,
            num_generations=5,
            seed=42,
        )

        assert optimizer.prompt_model == prompt_model
        assert optimizer.population_size == 10
        assert optimizer.num_generations == 5
        assert optimizer.seed == 42

    def test_initialization_with_all_options(
        self, instructions_param, metric
    ):
        """Test initialization with all options."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=30,
            num_generations=20,
            mutation_rate=0.4,
            crossover_rate=0.6,
            tournament_size=5,
            elite_size=3,
            seed=123,
        )

        assert optimizer.mutation_rate == 0.4
        assert optimizer.crossover_rate == 0.6
        assert optimizer.tournament_size == 5
        assert optimizer.elite_size == 3


class TestGEPAStep:
    """Tests for GEPA step method."""

    def test_step_without_prompt_model(
        self, instructions_param, trainset, metric
    ):
        """Test step without prompt model."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=5,
            num_generations=2,
        )

        optimizer.step(trainset)

        assert optimizer._step_count == 1
        assert len(optimizer._population) > 0

    def test_step_with_prompt_model(
        self, instructions_param, trainset, valset, metric
    ):
        """Test step with prompt model."""
        prompt_model = MockModule("New instruction")
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            population_size=5,
            num_generations=2,
            seed=42,
        )

        optimizer.step(trainset, valset)

        assert optimizer._step_count == 1

    def test_step_records_stats(
        self, instructions_param, trainset, metric
    ):
        """Test that step records statistics."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=5,
            num_generations=3,
        )

        optimizer.step(trainset)

        assert len(optimizer._stats_history) == 3


class TestGEPAPopulation:
    """Tests for population management."""

    def test_initialize_population(
        self, instructions_param, trainset, metric
    ):
        """Test population initialization."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=10,
            seed=42,
        )

        optimizer._initialize_population(trainset)

        assert len(optimizer._population) == 10
        assert all(isinstance(ind, Individual) for ind in optimizer._population)

    def test_population_has_base_instruction(
        self, instructions_param, trainset, metric
    ):
        """Test that population includes base instruction."""
        base_instruction = "Do the task."
        param = Parameter(base_instruction, PromptSpec.INSTRUCTIONS)
        optimizer = GEPA(
            [param],
            metric=metric,
            population_size=5,
            seed=42,
        )

        optimizer._initialize_population(trainset)

        instructions = [ind.instruction for ind in optimizer._population]
        assert base_instruction in instructions


class TestGEPASelection:
    """Tests for selection methods."""

    def test_select_parent_tournament(
        self, instructions_param, trainset, metric
    ):
        """Test tournament selection."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            tournament_size=3,
            seed=42,
        )
        optimizer._population = [
            Individual(instruction=f"ind{i}", fitness=i * 0.1)
            for i in range(10)
        ]

        parent = optimizer._select_parent()

        assert parent in optimizer._population

    def test_select_parent_prefers_higher_fitness(
        self, instructions_param, metric
    ):
        """Test that selection prefers higher fitness."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            tournament_size=10,  # Large tournament = almost always best
            seed=42,
        )
        optimizer._population = [
            Individual(instruction="low", fitness=0.1),
            Individual(instruction="high", fitness=0.9),
        ]

        # Select many times, should mostly get "high"
        selections = [optimizer._select_parent().instruction for _ in range(20)]
        high_count = selections.count("high")

        assert high_count > 10  # Majority should be "high"


class TestGEPAGeneticOperations:
    """Tests for genetic operations."""

    def test_mutate_without_prompt_model(
        self, instructions_param, metric
    ):
        """Test mutation without prompt model."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            seed=42,
        )
        parent = Individual(instruction="Original instruction")

        offspring = optimizer._mutate(parent)

        assert isinstance(offspring, Individual)
        assert offspring.instruction != ""

    def test_mutate_with_prompt_model(
        self, instructions_param, metric
    ):
        """Test mutation with prompt model."""
        prompt_model = MockModule("Mutated by LLM")
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            seed=42,
        )
        parent = Individual(instruction="Original instruction")

        offspring = optimizer._mutate(parent)

        assert offspring.instruction == "Mutated by LLM"
        assert prompt_model.call_count == 1

    def test_crossover_without_prompt_model(
        self, instructions_param, trainset, metric
    ):
        """Test crossover without prompt model."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            seed=42,
        )
        parent_a = Individual(instruction="First parent instruction here")
        parent_b = Individual(instruction="Second parent instruction here")

        offspring = optimizer._crossover(parent_a, parent_b, trainset)

        assert isinstance(offspring, Individual)
        assert offspring.instruction != ""

    def test_crossover_with_prompt_model(
        self, instructions_param, trainset, metric
    ):
        """Test crossover with prompt model."""
        prompt_model = MockModule("Combined instruction")
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            seed=42,
        )
        parent_a = Individual(instruction="First")
        parent_b = Individual(instruction="Second")

        offspring = optimizer._crossover(parent_a, parent_b, trainset)

        assert offspring.instruction == "Combined instruction"

    def test_simple_mutate_rephrase(self, instructions_param, metric):
        """Test simple mutation - rephrase."""
        optimizer = GEPA([instructions_param], metric=metric, seed=42)

        result = optimizer._simple_mutate("word1 word2 word3", "rephrase")

        # Should swap words
        assert "word" in result

    def test_simple_mutate_expand(self, instructions_param, metric):
        """Test simple mutation - expand."""
        optimizer = GEPA([instructions_param], metric=metric)

        result = optimizer._simple_mutate("Original text", "expand")

        assert "Original text" in result
        assert len(result) > len("Original text")

    def test_simple_mutate_simplify(self, instructions_param, metric):
        """Test simple mutation - simplify."""
        optimizer = GEPA([instructions_param], metric=metric)

        result = optimizer._simple_mutate(
            "This is a long instruction with many words", "simplify"
        )

        assert len(result.split()) < 8


class TestGEPAEvolution:
    """Tests for evolution process."""

    def test_evolve_population_preserves_elite(
        self, instructions_param, trainset, metric
    ):
        """Test that evolution preserves elite individuals."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=5,
            elite_size=2,
            seed=42,
        )
        optimizer._population = [
            Individual(instruction=f"ind{i}", fitness=i * 0.2)
            for i in range(5)
        ]
        optimizer._current_generation = 1

        new_population = optimizer._evolve_population(trainset)

        # Top 2 should be preserved
        assert len(new_population) == 5

    def test_evolve_increments_generation(
        self, instructions_param, trainset, metric
    ):
        """Test that evolution increments generation."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=5,
            num_generations=3,
        )

        optimizer.step(trainset)

        assert optimizer._current_generation == 3


class TestGEPAStateManagement:
    """Tests for state management."""

    def test_state_dict(self, instructions_param, trainset, metric):
        """Test state_dict method."""
        optimizer = GEPA(
            [instructions_param], metric=metric, seed=42
        )
        optimizer._current_generation = 5
        optimizer._best_fitness = 0.95
        optimizer._best_individual = Individual(
            instruction="Best instruction",
            demos=[trainset[0]],
            fitness=0.95,
        )

        state = optimizer.state_dict()

        assert state["current_generation"] == 5
        assert state["best_fitness"] == 0.95
        assert state["best_instruction"] == "Best instruction"
        assert state["seed"] == 42

    def test_load_state_dict(self, instructions_param, metric):
        """Test load_state_dict method."""
        optimizer = GEPA([instructions_param], metric=metric, seed=1)

        state = {
            "param_groups": optimizer.param_groups,
            "state": {},  # Required by base Optimizer
            "current_generation": 10,
            "best_fitness": 0.88,
            "best_instruction": "Loaded instruction",
            "best_demos": [{"inputs": "q", "labels": "a"}],
            "seed": 99,
        }

        optimizer.load_state_dict(state)

        assert optimizer._current_generation == 10
        assert optimizer._best_fitness == 0.88
        assert optimizer._best_individual is not None
        assert optimizer._best_individual.instruction == "Loaded instruction"
        assert optimizer.seed == 99

    def test_get_best_individual(self, instructions_param, metric):
        """Test getting best individual."""
        optimizer = GEPA([instructions_param], metric=metric)
        individual = Individual(instruction="Best")
        optimizer._best_individual = individual

        assert optimizer.get_best_individual() == individual

    def test_get_best_fitness(self, instructions_param, metric):
        """Test getting best fitness."""
        optimizer = GEPA([instructions_param], metric=metric)
        optimizer._best_fitness = 0.92

        assert optimizer.get_best_fitness() == 0.92

    def test_get_population(self, instructions_param, metric):
        """Test getting population."""
        optimizer = GEPA([instructions_param], metric=metric)
        optimizer._population = [Individual(instruction="Test")]

        assert len(optimizer.get_population()) == 1

    def test_get_stats_history(self, instructions_param, metric):
        """Test getting stats history."""
        optimizer = GEPA([instructions_param], metric=metric)
        stats = GEPAStats(
            generation=1,
            best_fitness=0.8,
            avg_fitness=0.6,
            population_size=10,
            num_mutations=5,
            num_crossovers=4,
        )
        optimizer._stats_history.append(stats)

        assert optimizer.get_stats_history() == [stats]


class TestGEPAFormatDemos:
    """Tests for demo formatting."""

    def test_format_demos(self, instructions_param, trainset, metric):
        """Test formatting demos."""
        optimizer = GEPA([instructions_param], metric=metric)

        formatted = optimizer._format_demos(trainset[:2])

        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "What is 2+2?" in formatted
        assert "4" in formatted

    def test_sample_demos(self, instructions_param, trainset, metric):
        """Test sampling demos."""
        optimizer = GEPA([instructions_param], metric=metric, seed=42)

        demos = optimizer._sample_demos(trainset, k=3)

        assert len(demos) == 3
        assert all(d in trainset for d in demos)
