"""Unit tests for SIMBA optimizer."""

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.simba import SIMBA, SIMBACandidate, SIMBAResult, SIMBATrialLog
from msgflux.optim.simba.utils import (
    ExecutionResult,
    TrajectoryStep,
    append_a_demo,
    append_a_rule,
    prepare_models_for_resampling,
    recursive_mask,
    wrap_program,
)


class MockModule:
    """Mock module for testing SIMBA."""

    def __init__(self, response: str = "4"):
        self.response = response
        self.call_count = 0
        self._name = "MockModule"
        self._modules = {}
        self.demos = []
        self.instructions = "Test instructions"
        self._simba_idx = None

    def __call__(self, inputs) -> str:
        self.call_count += 1
        return self.response

    def named_modules(self):
        yield "", self
        for name, mod in self._modules.items():
            yield name, mod

    def register_forward_hook(self, hook):
        class Handle:
            def remove(self):
                pass

        return Handle()


class MockAsyncModule(MockModule):
    """Mock async module for testing."""

    async def acall(self, inputs) -> str:
        self.call_count += 1
        return self.response


@pytest.fixture
def trainset():
    """Create sample training set with enough examples for bsize."""
    examples = []
    for i in range(40):  # Need at least bsize (default 32)
        examples.append(
            Example(
                inputs=f"What is {i}+{i}?",
                labels=str(i * 2),
            )
        )
    return examples


@pytest.fixture
def small_trainset():
    """Create small training set for utility function tests."""
    return [
        Example(inputs="What is 2+2?", labels="4"),
        Example(inputs="What is 3+3?", labels="6"),
        Example(inputs="What is 4+4?", labels="8"),
    ]


@pytest.fixture
def metric():
    """Create exact match metric."""

    def exact_match(example: Example, prediction) -> float:
        if prediction is None:
            return 0.0
        return 1.0 if str(prediction) == str(example.labels) else 0.0

    return exact_match


@pytest.fixture
def instructions_param():
    """Create instructions parameter."""
    return Parameter("Solve the math problem.", PromptSpec.INSTRUCTIONS)


@pytest.fixture
def examples_param():
    """Create examples parameter."""
    return Parameter(None, PromptSpec.EXAMPLES)


# =============================================================================
# Tests for SIMBA Utility Functions
# =============================================================================


class TestTrajectoryStep:
    """Tests for TrajectoryStep dataclass."""

    def test_creation(self):
        """Test creating a trajectory step."""
        step = TrajectoryStep(
            module_name="test_module",
            inputs={"input": "test"},
            outputs={"output": "result"},
        )
        assert step.module_name == "test_module"
        assert step.inputs == {"input": "test"}
        assert step.outputs == {"output": "result"}


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_creation(self, small_trainset):
        """Test creating an execution result."""
        result = ExecutionResult(
            prediction="test",
            trace=[],
            score=0.5,
            example=small_trainset[0],
            output_metadata={},
        )
        assert result.prediction == "test"
        assert result.score == 0.5
        assert result.example == small_trainset[0]


class TestWrapProgram:
    """Tests for wrap_program function."""

    def test_wrap_basic(self, small_trainset, metric):
        """Test wrapping a basic program."""
        module = MockModule("4")
        wrapped = wrap_program(module, metric)

        result = wrapped(small_trainset[0])

        assert isinstance(result, ExecutionResult)
        assert result.prediction == "4"
        assert result.score == 1.0  # "4" matches "4"
        assert result.example == small_trainset[0]

    def test_wrap_with_wrong_answer(self, small_trainset, metric):
        """Test wrapping when prediction is wrong."""
        module = MockModule("wrong")
        wrapped = wrap_program(module, metric)

        result = wrapped(small_trainset[0])

        assert result.score == 0.0

    def test_wrap_with_exception(self, small_trainset, metric):
        """Test wrapping handles exceptions."""

        class FailingModule:
            def __call__(self, inputs):
                raise ValueError("Test error")

            def named_modules(self):
                return iter([])

        module = FailingModule()
        wrapped = wrap_program(module, metric)

        result = wrapped(small_trainset[0])

        assert result.prediction is None
        assert result.score == 0.0


class TestPrepareModelsForResampling:
    """Tests for prepare_models_for_resampling function."""

    def test_basic(self):
        """Test basic model preparation."""
        module = MockModule()
        models = prepare_models_for_resampling(module, 3)

        assert len(models) == 3

    def test_with_teacher_settings(self):
        """Test with teacher settings."""

        class ModuleWithModel(MockModule):
            def __init__(self):
                super().__init__()
                self.model = MockModule("base")

        module = ModuleWithModel()
        teacher_model = MockModule("teacher")
        models = prepare_models_for_resampling(
            module, 3, teacher_settings={"model": teacher_model}
        )

        assert len(models) == 3
        assert models[0] == teacher_model


class TestAppendADemo:
    """Tests for append_a_demo strategy."""

    def test_strategy_creation(self):
        """Test creating the strategy."""
        strategy = append_a_demo(100_000)
        assert callable(strategy)
        assert strategy.__name__ == "append_a_demo"

    def test_skip_low_score(self, small_trainset):
        """Test skipping when score is too low."""
        strategy = append_a_demo(100_000)
        module = MockModule()

        bucket = [
            ExecutionResult(
                prediction="wrong",
                trace=[],
                score=0.05,  # Below 10th percentile
                example=small_trainset[0],
                output_metadata={},
            )
        ]

        result = strategy(
            bucket,
            module,
            predictor2name={id(module): "mock"},
            name2predictor={"mock": module},
            batch_10p_score=0.1,
            batch_90p_score=0.9,
        )

        assert result is False  # Should skip


class TestAppendARule:
    """Tests for append_a_rule strategy."""

    def test_skip_when_no_contrast(self, small_trainset):
        """Test skipping when there's no contrast."""
        module = MockModule()

        bucket = [
            ExecutionResult(
                prediction="4",
                trace=[],
                score=0.05,  # Too low
                example=small_trainset[0],
                output_metadata={},
            ),
            ExecutionResult(
                prediction="wrong",
                trace=[],
                score=0.0,
                example=small_trainset[0],
                output_metadata={},
            ),
        ]

        result = append_a_rule(
            bucket,
            module,
            predictor2name={id(module): "mock"},
            name2predictor={"mock": module},
            batch_10p_score=0.1,
            batch_90p_score=0.9,
            prompt_model=None,
        )

        assert result is False

    def test_skip_without_prompt_model(self, small_trainset):
        """Test skipping when no prompt model provided."""
        module = MockModule()

        bucket = [
            ExecutionResult(
                prediction="4",
                trace=[],
                score=0.9,
                example=small_trainset[0],
                output_metadata={},
            ),
            ExecutionResult(
                prediction="wrong",
                trace=[],
                score=0.1,
                example=small_trainset[0],
                output_metadata={},
            ),
        ]

        result = append_a_rule(
            bucket,
            module,
            predictor2name={id(module): "mock"},
            name2predictor={"mock": module},
            batch_10p_score=0.05,
            batch_90p_score=0.95,
            prompt_model=None,
        )

        assert result is False


class TestRecursiveMask:
    """Tests for recursive_mask function."""

    def test_serializable_values(self):
        """Test that serializable values pass through."""
        assert recursive_mask({"a": 1, "b": "test"}) == {"a": 1, "b": "test"}
        assert recursive_mask([1, 2, 3]) == [1, 2, 3]
        assert recursive_mask("test") == "test"

    def test_non_serializable_values(self):
        """Test that non-serializable values are masked."""

        class CustomClass:
            pass

        result = recursive_mask({"a": CustomClass()})
        assert "non-serializable" in result["a"]


# =============================================================================
# Tests for SIMBA Data Classes
# =============================================================================


class TestSIMBACandidate:
    """Tests for SIMBACandidate dataclass."""

    def test_creation(self):
        """Test creating a candidate."""
        module = MockModule()
        candidate = SIMBACandidate(program=module, score=0.85, step=3)

        assert candidate.program == module
        assert candidate.score == 0.85
        assert candidate.step == 3

    def test_default_values(self):
        """Test default values."""
        module = MockModule()
        candidate = SIMBACandidate(program=module)

        assert candidate.score == 0.0
        assert candidate.step == 0


class TestSIMBATrialLog:
    """Tests for SIMBATrialLog dataclass."""

    def test_creation(self):
        """Test creating a trial log."""
        log = SIMBATrialLog(
            batch_idx=3,
            batch_scores=[0.5, 0.6, 0.7],
            baseline_score=0.6,
            candidate_scores=[0.65, 0.7],
            best_candidate_score=0.7,
            strategy_used="append_a_demo",
        )

        assert log.batch_idx == 3
        assert log.baseline_score == 0.6
        assert log.strategy_used == "append_a_demo"


class TestSIMBAResult:
    """Tests for SIMBAResult dataclass."""

    def test_creation(self):
        """Test creating a result."""
        module = MockModule()
        candidates = [SIMBACandidate(program=module, score=0.8)]
        trial_logs = {0: SIMBATrialLog(batch_idx=0)}

        result = SIMBAResult(
            program=module,
            score=0.85,
            candidate_programs=candidates,
            trial_logs=trial_logs,
        )

        assert result.program == module
        assert result.score == 0.85
        assert len(result.candidate_programs) == 1
        assert 0 in result.trial_logs


# =============================================================================
# Tests for SIMBA Optimizer
# =============================================================================


class TestSIMBAInitialization:
    """Tests for SIMBA initialization."""

    def test_basic_initialization(self, instructions_param, metric):
        """Test basic initialization."""
        optimizer = SIMBA([instructions_param], metric=metric)

        assert optimizer.metric == metric
        assert optimizer.bsize == 32
        assert optimizer.num_candidates == 6
        assert optimizer.max_steps == 8
        assert optimizer.max_demos == 4
        assert optimizer.seed == 0
        assert len(optimizer.strategies) == 2  # append_a_demo and append_a_rule

    def test_initialization_with_custom_params(self, instructions_param, metric):
        """Test initialization with custom parameters."""
        prompt_model = MockModule()
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=16,
            num_candidates=4,
            max_steps=5,
            max_demos=2,
            prompt_model=prompt_model,
            seed=42,
        )

        assert optimizer.bsize == 16
        assert optimizer.num_candidates == 4
        assert optimizer.max_steps == 5
        assert optimizer.max_demos == 2
        assert optimizer.prompt_model == prompt_model
        assert optimizer.seed == 42

    def test_initialization_no_demos(self, instructions_param, metric):
        """Test initialization with max_demos=0."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            max_demos=0,
        )

        # Should only have append_a_rule strategy
        assert len(optimizer.strategies) == 1


class TestSIMBAMethods:
    """Tests for SIMBA internal methods."""

    def test_reset_state(self, instructions_param, metric):
        """Test state reset."""
        optimizer = SIMBA([instructions_param], metric=metric)
        optimizer._programs = [MockModule()]
        optimizer._program_scores = {0: [0.5]}

        optimizer._reset_state()

        assert optimizer._programs == []
        assert optimizer._program_scores == {}
        assert optimizer._next_program_idx == 0

    def test_calc_average_score(self, instructions_param, metric):
        """Test average score calculation."""
        optimizer = SIMBA([instructions_param], metric=metric)
        optimizer._program_scores = {0: [0.5, 0.6, 0.7]}

        assert optimizer._calc_average_score(0) == 0.6
        assert optimizer._calc_average_score(999) == 0.0  # Non-existent

    def test_register_program(self, instructions_param, metric):
        """Test program registration."""
        optimizer = SIMBA([instructions_param], metric=metric)
        module = MockModule()

        optimizer._register_program(module, [0.5, 0.6])

        assert module._simba_idx == 1
        assert module in optimizer._programs
        assert optimizer._program_scores[1] == [0.5, 0.6]

    def test_drop_demos(self, instructions_param, metric):
        """Test demo dropping."""
        optimizer = SIMBA([instructions_param], metric=metric, max_demos=2)
        module = MockModule()
        module.demos = [
            Example(inputs="a", labels="b"),
            Example(inputs="c", labels="d"),
            Example(inputs="e", labels="f"),
            Example(inputs="g", labels="h"),
        ]

        optimizer._drop_demos(module)

        # Should have dropped some demos (stochastic)
        assert len(module.demos) <= 4


class TestSIMBAStep:
    """Tests for SIMBA step method."""

    def test_step_requires_student(self, trainset, instructions_param, metric):
        """Test that step requires student module."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
        )

        with pytest.raises(ValueError, match="student module is required"):
            optimizer.step(trainset)

    def test_step_requires_enough_data(self, instructions_param, metric):
        """Test that step requires trainset >= bsize."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
        )
        small_trainset = [Example(inputs="a", labels="b")]

        with pytest.raises(AssertionError, match="Trainset too small"):
            optimizer.step(small_trainset, student=MockModule())

    def test_step_basic(self, trainset, instructions_param, metric):
        """Test basic step execution."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
        )
        student = MockModule("4")

        result = optimizer.step(trainset, student=student)

        assert isinstance(result, SIMBAResult)
        assert result.program is not None
        assert isinstance(result.score, float)
        assert len(result.candidate_programs) > 0
        assert 0 in result.trial_logs

    def test_step_increments_count(self, trainset, instructions_param, metric):
        """Test that step increments step count."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
        )
        student = MockModule("4")

        assert optimizer.step_count == 0
        optimizer.step(trainset, student=student)
        assert optimizer.step_count == 1


class TestSIMBAStateDictionary:
    """Tests for SIMBA state dictionary."""

    def test_state_dict(self, instructions_param, metric):
        """Test state dict creation."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            seed=42,
        )

        state = optimizer.state_dict()

        assert "seed" in state
        assert state["seed"] == 42

    def test_load_state_dict(self, instructions_param, metric):
        """Test loading state dict."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            seed=0,
        )

        # Get a valid state dict first, then modify it
        state = optimizer.state_dict()
        state["seed"] = 123
        state["next_program_idx"] = 5
        optimizer.load_state_dict(state)

        assert optimizer.seed == 123
        assert optimizer._next_program_idx == 5


# =============================================================================
# Async Tests
# =============================================================================


class TestSIMBAAsync:
    """Tests for SIMBA async methods."""

    @pytest.mark.asyncio
    async def test_astep_basic(self, trainset, instructions_param, metric):
        """Test basic async step."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
        )
        student = MockAsyncModule("4")

        result = await optimizer.astep(trainset, student=student)

        assert isinstance(result, SIMBAResult)
        assert result.program is not None

    @pytest.mark.asyncio
    async def test_astep_with_concurrency(self, trainset, instructions_param, metric):
        """Test async step with concurrency limit."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
        )
        student = MockAsyncModule("4")

        result = await optimizer.astep(
            trainset, student=student, max_concurrency=5
        )

        assert isinstance(result, SIMBAResult)

    @pytest.mark.asyncio
    async def test_astep_requires_student(self, trainset, instructions_param, metric):
        """Test that astep requires student."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
        )

        with pytest.raises(ValueError, match="student module is required"):
            await optimizer.astep(trainset)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSIMBAIntegration:
    """Integration tests for SIMBA."""

    def test_multiple_steps(self, trainset, instructions_param, metric):
        """Test running multiple optimization steps."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=2,
            num_candidates=2,
        )
        student = MockModule("4")

        result = optimizer.step(trainset, student=student)

        assert result.score >= 0.0
        assert len(result.trial_logs) == 2  # Two steps

    def test_with_prompt_model(self, trainset, instructions_param, metric):
        """Test with prompt model for rule generation."""

        class PromptModelMock:
            def __call__(self, prompt: str) -> str:
                return '{"mock_module": "Test advice"}'

        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
            prompt_model=PromptModelMock(),
        )
        student = MockModule("4")

        result = optimizer.step(trainset, student=student)

        assert isinstance(result, SIMBAResult)

    def test_candidate_programs_sorted(self, trainset, instructions_param, metric):
        """Test that candidate programs are sorted by score."""
        optimizer = SIMBA(
            [instructions_param],
            metric=metric,
            bsize=32,
            max_steps=1,
            num_candidates=2,
        )
        student = MockModule("4")

        result = optimizer.step(trainset, student=student)

        if len(result.candidate_programs) > 1:
            scores = [c.score for c in result.candidate_programs]
            assert scores == sorted(scores, reverse=True)
