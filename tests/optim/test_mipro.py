"""Unit tests for MIPROv2 optimizer."""

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.mipro import MIPROv2, MiproTrial, PromptCandidate


class MockModule:
    """Mock module for testing MIPROv2."""

    def __init__(self, response: str = "Generated instruction"):
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


class TestPromptCandidate:
    """Tests for PromptCandidate dataclass."""

    def test_creation(self):
        """Test creating a candidate."""
        candidate = PromptCandidate(instruction="Test instruction")
        assert candidate.instruction == "Test instruction"
        assert candidate.demos == []
        assert candidate.score == 0.0
        assert candidate.evaluated is False

    def test_with_demos(self, trainset):
        """Test candidate with demos."""
        candidate = PromptCandidate(
            instruction="Test", demos=trainset[:2], score=0.85
        )
        assert len(candidate.demos) == 2
        assert candidate.score == 0.85

    def test_hashable(self):
        """Test that candidate is hashable."""
        candidate = PromptCandidate(instruction="Test")
        # Should not raise
        hash(candidate)


class TestMiproTrial:
    """Tests for MiproTrial dataclass."""

    def test_creation(self):
        """Test creating a trial."""
        candidate = PromptCandidate(instruction="Test")
        trial = MiproTrial(
            instruction_idx=0,
            demo_indices=[1, 2, 3],
            score=0.9,
            candidate=candidate,
        )
        assert trial.instruction_idx == 0
        assert trial.demo_indices == [1, 2, 3]
        assert trial.score == 0.9


class TestMIPROv2Initialization:
    """Tests for MIPROv2 initialization."""

    def test_basic_initialization(self, instructions_param, metric):
        """Test basic initialization."""
        optimizer = MIPROv2([instructions_param], metric=metric)

        assert optimizer.metric == metric
        assert optimizer.num_candidates == 10
        assert optimizer.num_demos == 4
        assert optimizer.num_trials == 50
        assert optimizer.seed == 0

    def test_initialization_with_prompt_model(
        self, instructions_param, metric
    ):
        """Test initialization with prompt model."""
        prompt_model = MockModule()
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            num_candidates=5,
            num_trials=20,
            seed=42,
        )

        assert optimizer.prompt_model == prompt_model
        assert optimizer.num_candidates == 5
        assert optimizer.num_trials == 20
        assert optimizer.seed == 42

    def test_initialization_with_all_options(
        self, instructions_param, metric
    ):
        """Test initialization with all options."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=8,
            num_demos=6,
            num_trials=100,
            init_temperature=0.5,
            task_context="Math problems",
            seed=123,
        )

        assert optimizer.num_demos == 6
        assert optimizer.init_temperature == 0.5
        assert optimizer.task_context == "Math problems"


class TestMIPROv2Step:
    """Tests for MIPROv2 step method."""

    def test_step_without_prompt_model(
        self, instructions_param, trainset, metric
    ):
        """Test step without prompt model."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_trials=5,
        )

        optimizer.step(trainset)

        assert optimizer._step_count == 1
        assert len(optimizer._instruction_candidates) > 0

    def test_step_with_prompt_model(
        self, instructions_param, trainset, valset, metric
    ):
        """Test step with prompt model."""
        prompt_model = MockModule("New instruction")
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            num_candidates=3,
            num_trials=5,
            seed=42,
        )

        optimizer.step(trainset, valset)

        assert optimizer._step_count == 1
        assert prompt_model.call_count >= 1

    def test_step_records_trials(
        self, instructions_param, trainset, metric
    ):
        """Test that step records trials."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_trials=10,
        )

        optimizer.step(trainset)

        assert len(optimizer._trials) == 10


class TestMIPROv2Sampling:
    """Tests for sampling methods."""

    def test_sample_instruction_initial(
        self, instructions_param, trainset, metric
    ):
        """Test sampling instruction initially (random)."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=5,
            seed=42,
        )
        optimizer._instruction_candidates = ["i1", "i2", "i3", "i4", "i5"]

        idx = optimizer._sample_instruction()

        assert 0 <= idx < 5

    def test_sample_demos_initial(
        self, instructions_param, trainset, metric
    ):
        """Test sampling demos initially (random)."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_demos=3,
            seed=42,
        )
        optimizer._demo_pool = trainset

        indices = optimizer._sample_demos()

        assert len(indices) == 3
        assert all(0 <= i < len(trainset) for i in indices)

    def test_sample_instruction_with_history(
        self, instructions_param, metric
    ):
        """Test sampling with history updates surrogate."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=3,
            seed=42,
        )
        optimizer._instruction_candidates = ["i1", "i2", "i3"]

        # Add some history
        optimizer._instruction_scores[0] = [0.9, 0.85]
        optimizer._instruction_scores[1] = [0.3, 0.4]
        # Instruction 2 has no history

        # Sample multiple times - should favor idx 0 (higher scores)
        indices = [optimizer._sample_instruction() for _ in range(10)]

        # At least some should be 0 (highest scorer)
        assert 0 in indices or 2 in indices  # 2 is unexplored, gets bonus


class TestMIPROv2StateManagement:
    """Tests for state management."""

    def test_state_dict(self, instructions_param, trainset, metric):
        """Test state_dict method."""
        optimizer = MIPROv2(
            [instructions_param], metric=metric, seed=42
        )
        optimizer._instruction_candidates = ["inst1", "inst2"]
        optimizer._best_score = 0.95
        optimizer._best_candidate = PromptCandidate(
            instruction="Best instruction",
            demos=[trainset[0]],
            score=0.95,
        )

        state = optimizer.state_dict()

        assert state["instruction_candidates"] == ["inst1", "inst2"]
        assert state["best_score"] == 0.95
        assert state["best_instruction"] == "Best instruction"
        assert state["seed"] == 42

    def test_load_state_dict(self, instructions_param, metric):
        """Test load_state_dict method."""
        optimizer = MIPROv2([instructions_param], metric=metric, seed=1)

        state = {
            "param_groups": optimizer.param_groups,
            "state": {},  # Required by base Optimizer
            "instruction_candidates": ["loaded1", "loaded2"],
            "best_score": 0.88,
            "best_instruction": "Loaded instruction",
            "best_demos": [{"inputs": "q", "labels": "a"}],
            "seed": 99,
        }

        optimizer.load_state_dict(state)

        assert optimizer._instruction_candidates == ["loaded1", "loaded2"]
        assert optimizer._best_score == 0.88
        assert optimizer._best_candidate is not None
        assert optimizer._best_candidate.instruction == "Loaded instruction"
        assert optimizer.seed == 99

    def test_get_best_candidate(self, instructions_param, metric):
        """Test getting best candidate."""
        optimizer = MIPROv2([instructions_param], metric=metric)
        candidate = PromptCandidate(instruction="Best")
        optimizer._best_candidate = candidate

        assert optimizer.get_best_candidate() == candidate

    def test_get_trials(self, instructions_param, metric):
        """Test getting trials."""
        optimizer = MIPROv2([instructions_param], metric=metric)
        trial = MiproTrial(
            instruction_idx=0,
            demo_indices=[0],
            score=0.5,
            candidate=PromptCandidate(instruction="Test"),
        )
        optimizer._trials.append(trial)

        assert optimizer.get_trials() == [trial]


class TestMIPROv2SurrogateUpdate:
    """Tests for surrogate model updates."""

    def test_update_surrogate(self, instructions_param, metric):
        """Test updating surrogate model."""
        optimizer = MIPROv2([instructions_param], metric=metric)

        trial = MiproTrial(
            instruction_idx=1,
            demo_indices=[0, 2, 4],
            score=0.8,
            candidate=PromptCandidate(instruction="Test"),
        )

        optimizer._update_surrogate(trial)

        assert 1 in optimizer._instruction_scores
        assert optimizer._instruction_scores[1] == [0.8]
        assert 0 in optimizer._demo_scores
        assert 2 in optimizer._demo_scores
        assert 4 in optimizer._demo_scores

    def test_update_surrogate_accumulates(self, instructions_param, metric):
        """Test that surrogate accumulates scores."""
        optimizer = MIPROv2([instructions_param], metric=metric)

        for score in [0.7, 0.8, 0.9]:
            trial = MiproTrial(
                instruction_idx=0,
                demo_indices=[0],
                score=score,
                candidate=PromptCandidate(instruction="Test"),
            )
            optimizer._update_surrogate(trial)

        assert optimizer._instruction_scores[0] == [0.7, 0.8, 0.9]


class TestMIPROv2FormatDemos:
    """Tests for demo formatting."""

    def test_format_demos(self, instructions_param, trainset, metric):
        """Test formatting demos."""
        optimizer = MIPROv2([instructions_param], metric=metric)

        formatted = optimizer._format_demos(trainset[:2])

        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "What is 2+2?" in formatted
        assert "4" in formatted

    def test_format_empty_demos(self, instructions_param, metric):
        """Test formatting empty demos."""
        optimizer = MIPROv2([instructions_param], metric=metric)

        formatted = optimizer._format_demos([])

        assert formatted == ""
