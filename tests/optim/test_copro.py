"""Unit tests for COPRO optimizer."""

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.copro import COPRO, CoproCandidate, CoproResult


class MockModule:
    """Mock module for testing COPRO."""

    def __init__(self, response: str = "default response"):
        self.response = response
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        # Simulate generating instruction candidates
        if "INSTRUCTION 1:" in prompt or "Generate" in prompt:
            return """INSTRUCTION 1:
Be clear and specific about the task.

INSTRUCTION 2:
Follow the examples carefully.

INSTRUCTION 3:
Think step by step before answering."""
        return self.response


@pytest.fixture
def trainset():
    """Create sample training set."""
    return [
        Example(inputs="What is 2+2?", labels="4"),
        Example(inputs="What is 3+3?", labels="6"),
        Example(inputs="What is 4+4?", labels="8"),
        Example(inputs="What is 5+5?", labels="10"),
    ]


@pytest.fixture
def valset():
    """Create sample validation set."""
    return [
        Example(inputs="What is 6+6?", labels="12"),
        Example(inputs="What is 7+7?", labels="14"),
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


class TestCoproCandidate:
    """Tests for CoproCandidate dataclass."""

    def test_creation(self):
        """Test creating a candidate."""
        candidate = CoproCandidate(instruction="Test instruction")
        assert candidate.instruction == "Test instruction"
        assert candidate.score == 0.0
        assert candidate.evaluated is False

    def test_with_score(self):
        """Test candidate with score."""
        candidate = CoproCandidate(
            instruction="Test", score=0.85, evaluated=True
        )
        assert candidate.score == 0.85
        assert candidate.evaluated is True


class TestCOPROInitialization:
    """Tests for COPRO initialization."""

    def test_basic_initialization(self, instructions_param, metric):
        """Test basic initialization."""
        optimizer = COPRO([instructions_param], metric=metric)

        assert optimizer.metric == metric
        assert optimizer.num_candidates == 5
        assert optimizer.seed == 0

    def test_initialization_with_prompt_model(
        self, instructions_param, metric
    ):
        """Test initialization with prompt model."""
        prompt_model = MockModule()
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            num_candidates=10,
            seed=42,
        )

        assert optimizer.prompt_model == prompt_model
        assert optimizer.num_candidates == 10
        assert optimizer.seed == 42

    def test_initialization_with_all_options(
        self, instructions_param, metric
    ):
        """Test initialization with all options."""
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            num_candidates=8,
            max_success_examples=10,
            max_failure_examples=10,
            task_description="Calculate math problems",
            breadth=3,
            depth=2,
            seed=123,
        )

        assert optimizer.max_success_examples == 10
        assert optimizer.max_failure_examples == 10
        assert optimizer.task_description == "Calculate math problems"
        assert optimizer.breadth == 3
        assert optimizer.depth == 2


class TestCOPROStep:
    """Tests for COPRO step method."""

    def test_step_without_prompt_model(
        self, instructions_param, trainset, metric
    ):
        """Test step without prompt model (no optimization)."""
        optimizer = COPRO([instructions_param], metric=metric)

        optimizer.step(trainset)

        assert optimizer._step_count == 1

    def test_step_with_prompt_model(
        self, instructions_param, trainset, valset, metric
    ):
        """Test step with prompt model."""
        prompt_model = MockModule()
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            prompt_model=prompt_model,
            num_candidates=3,
            seed=42,
        )

        optimizer.step(trainset, valset)

        assert optimizer._step_count == 1
        assert prompt_model.call_count >= 1

    def test_step_increments_count(self, instructions_param, trainset, metric):
        """Test that step increments step count."""
        optimizer = COPRO([instructions_param], metric=metric)

        optimizer.step(trainset)
        assert optimizer._step_count == 1

        optimizer.step(trainset)
        assert optimizer._step_count == 2


class TestCOPROCandidateGeneration:
    """Tests for candidate generation."""

    def test_parse_candidates(self, instructions_param, metric):
        """Test parsing candidates from model response."""
        optimizer = COPRO([instructions_param], metric=metric)

        response = """INSTRUCTION 1:
Be clear and specific.

INSTRUCTION 2:
Follow examples.

INSTRUCTION 3:
Think step by step."""

        candidates = optimizer._parse_candidates(response)

        assert len(candidates) == 3
        assert all(isinstance(c, CoproCandidate) for c in candidates)

    def test_format_examples(self, instructions_param, trainset, metric):
        """Test formatting examples for prompt."""
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            max_success_examples=2,
            seed=42,
        )

        examples_with_scores = [
            (trainset[0], "4", 1.0),
            (trainset[1], "6", 1.0),
        ]

        formatted = optimizer._format_examples(examples_with_scores, 2)

        assert "Example 1:" in formatted
        assert "Input:" in formatted
        assert "Expected:" in formatted


class TestCOPROStateManagement:
    """Tests for state management."""

    def test_state_dict(self, instructions_param, metric):
        """Test state_dict method."""
        optimizer = COPRO(
            [instructions_param], metric=metric, seed=42
        )
        optimizer._best_instruction = "Best instruction"
        optimizer._best_score = 0.95

        state = optimizer.state_dict()

        assert state["best_instruction"] == "Best instruction"
        assert state["best_score"] == 0.95
        assert state["seed"] == 42

    def test_load_state_dict(self, instructions_param, metric):
        """Test load_state_dict method."""
        optimizer = COPRO([instructions_param], metric=metric, seed=1)

        state = {
            "param_groups": optimizer.param_groups,
            "state": {},  # Required by base Optimizer
            "best_instruction": "Loaded instruction",
            "best_score": 0.88,
            "seed": 99,
        }

        optimizer.load_state_dict(state)

        assert optimizer._best_instruction == "Loaded instruction"
        assert optimizer._best_score == 0.88
        assert optimizer.seed == 99

    def test_get_best_instruction(self, instructions_param, metric):
        """Test getting best instruction."""
        optimizer = COPRO([instructions_param], metric=metric)
        optimizer._best_instruction = "Test instruction"

        assert optimizer.get_best_instruction() == "Test instruction"

    def test_get_best_score(self, instructions_param, metric):
        """Test getting best score."""
        optimizer = COPRO([instructions_param], metric=metric)
        optimizer._best_score = 0.92

        assert optimizer.get_best_score() == 0.92


class TestCOPROWithTeacher:
    """Tests for COPRO with teacher module."""

    def test_collect_examples_without_teacher(
        self, instructions_param, trainset, metric
    ):
        """Test collecting examples without teacher."""
        optimizer = COPRO([instructions_param], metric=metric)

        success, failure = optimizer._collect_examples(
            trainset, teacher=None, current_instruction=""
        )

        # Without teacher, all examples are treated as success
        assert len(success) == len(trainset)
        assert len(failure) == 0

    def test_collect_examples_with_teacher(
        self, instructions_param, trainset, metric
    ):
        """Test collecting examples with teacher."""

        class TeacherModule:
            def __call__(self, inputs):
                # Return correct answer for first two, wrong for rest
                if "2+2" in inputs:
                    return "4"
                elif "3+3" in inputs:
                    return "6"
                return "wrong"

        optimizer = COPRO([instructions_param], metric=metric)
        teacher = TeacherModule()

        success, failure = optimizer._collect_examples(
            trainset, teacher=teacher, current_instruction=""
        )

        assert len(success) == 2
        assert len(failure) == 2
