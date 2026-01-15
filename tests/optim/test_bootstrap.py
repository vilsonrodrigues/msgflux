"""Tests for msgflux.optim.bootstrap module."""

from unittest.mock import MagicMock, patch

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.bootstrap import BootstrapFewShot, BootstrapResult, Trace


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self, return_value="predicted answer"):
        super().__init__()
        self.return_value = return_value
        self._name = "MockModule"
        self.call_count = 0

    def forward(self, inputs):
        self.call_count += 1
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value


class TestTrace:
    """Test suite for Trace dataclass."""

    def test_trace_creation(self):
        """Test creating a Trace."""
        trace = Trace(
            module_name="TestModule",
            inputs={"question": "What is 2+2?"},
            outputs={"answer": "4"},
        )

        assert trace.module_name == "TestModule"
        assert trace.inputs == {"question": "What is 2+2?"}
        assert trace.outputs == {"answer": "4"}
        assert trace.augmented is True

    def test_trace_augmented_false(self):
        """Test Trace with augmented=False."""
        trace = Trace(
            module_name="Test",
            inputs={},
            outputs={},
            augmented=False,
        )

        assert trace.augmented is False


class TestBootstrapResult:
    """Test suite for BootstrapResult dataclass."""

    def test_bootstrap_result_success(self):
        """Test successful BootstrapResult."""
        example = Example(inputs="test", labels="answer")
        result = BootstrapResult(
            example=example,
            prediction="answer",
            traces=[],
            score=1.0,
            success=True,
        )

        assert result.success is True
        assert result.score == 1.0
        assert result.error is None

    def test_bootstrap_result_failure(self):
        """Test failed BootstrapResult."""
        example = Example(inputs="test", labels="answer")
        result = BootstrapResult(
            example=example,
            prediction=None,
            traces=[],
            score=0.0,
            success=False,
            error="Test error",
        )

        assert result.success is False
        assert result.error == "Test error"


class TestBootstrapFewShot:
    """Test suite for BootstrapFewShot optimizer."""

    @pytest.fixture
    def trainset(self):
        """Create sample training set."""
        return [
            Example(inputs="What is 2+2?", labels="4"),
            Example(inputs="What is 3+3?", labels="6"),
            Example(inputs="What is 4+4?", labels="8"),
            Example(inputs="What is 5+5?", labels="10"),
        ]

    @pytest.fixture
    def params(self):
        """Create sample parameters."""
        return [
            Parameter(data="You are a calculator.", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]

    @pytest.fixture
    def metric(self):
        """Create simple exact match metric."""

        def exact_match(example, prediction):
            return float(str(example.labels).lower() == str(prediction).lower())

        return exact_match

    @pytest.fixture
    def optimizer(self, params, metric):
        """Create BootstrapFewShot optimizer."""
        return BootstrapFewShot(
            params,
            metric=metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
            seed=42,
        )

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.max_bootstrapped_demos == 2
        assert optimizer.max_labeled_demos == 4
        assert optimizer.max_rounds == 1
        assert optimizer.max_errors == 10

    def test_step_without_teacher(self, optimizer, trainset):
        """Test step without teacher (only labeled demos)."""
        optimizer.step(trainset, teacher=None)

        assert optimizer._step_count == 1
        # Without teacher, no bootstrapping happens
        assert len(optimizer._bootstrapped_indices) == 0

    def test_step_with_teacher(self, optimizer, trainset, params):
        """Test step with teacher module."""
        teacher = MockModule(return_value="4")  # Will match first example

        optimizer.step(trainset, teacher=teacher)

        assert optimizer._step_count == 1
        # Teacher was called
        assert teacher.call_count > 0

    def test_step_updates_examples_param(self, optimizer, trainset, params):
        """Test that step updates examples parameter."""
        examples_param = params[1]
        teacher = MockModule(return_value="4")

        optimizer.step(trainset, teacher=teacher)

        # Examples parameter should be updated
        # (either with bootstrapped or labeled demos)
        assert examples_param.data is not None

    def test_metric_threshold(self, params, trainset, metric):
        """Test metric threshold filtering."""
        optimizer = BootstrapFewShot(
            params,
            metric=metric,
            metric_threshold=0.5,
            max_bootstrapped_demos=2,
        )

        # Teacher returns wrong answer - score below threshold
        teacher = MockModule(return_value="wrong")

        optimizer.step(trainset, teacher=teacher)

        # No examples should pass threshold
        assert len(optimizer._bootstrapped_indices) == 0

    def test_max_errors(self, params, trainset, metric):
        """Test max errors limit."""
        optimizer = BootstrapFewShot(
            params,
            metric=metric,
            max_errors=2,
        )

        # Teacher raises exception
        teacher = MockModule(return_value=Exception("Test error"))

        with pytest.raises(RuntimeError, match="Maximum errors"):
            optimizer.step(trainset, teacher=teacher)

    def test_get_bootstrap_results(self, optimizer, trainset):
        """Test getting bootstrap results."""
        teacher = MockModule(return_value="4")

        optimizer.step(trainset, teacher=teacher)

        results = optimizer.get_bootstrap_results()
        assert len(results) > 0
        assert all(isinstance(r, BootstrapResult) for r in results)

    def test_get_success_rate(self, optimizer, trainset):
        """Test calculating success rate."""
        teacher = MockModule(return_value="4")

        optimizer.step(trainset, teacher=teacher)

        rate = optimizer.get_success_rate()
        assert 0.0 <= rate <= 1.0

    def test_state_dict(self, optimizer, trainset):
        """Test state dictionary serialization."""
        teacher = MockModule(return_value="4")
        optimizer.step(trainset, teacher=teacher)

        state = optimizer.state_dict()

        assert "bootstrapped_indices" in state
        assert "error_count" in state
        assert "seed" in state

    def test_load_state_dict(self, params, trainset, metric):
        """Test loading state dictionary."""
        optimizer1 = BootstrapFewShot(params, metric=metric, seed=42)
        teacher = MockModule(return_value="4")
        optimizer1.step(trainset, teacher=teacher)
        state = optimizer1.state_dict()

        # Create new optimizer
        params2 = [
            Parameter(data="new", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]
        optimizer2 = BootstrapFewShot(params2, metric=metric, seed=1)
        optimizer2.load_state_dict(state)

        assert optimizer2.seed == 42
        assert optimizer2._bootstrapped_indices == optimizer1._bootstrapped_indices

    def test_multiple_rounds(self, params, trainset, metric):
        """Test multiple bootstrap rounds."""
        optimizer = BootstrapFewShot(
            params,
            metric=metric,
            max_rounds=3,
            max_bootstrapped_demos=4,
        )

        teacher = MockModule(return_value="4")
        optimizer.step(trainset, teacher=teacher)

        # Should attempt multiple rounds
        assert optimizer._step_count == 1

    def test_respects_requires_grad(self, trainset, metric):
        """Test that frozen params are not updated."""
        params = [
            Parameter(data="", spec=PromptSpec.EXAMPLES, requires_grad=False),
        ]

        optimizer = BootstrapFewShot(params, metric=metric)
        teacher = MockModule(return_value="4")

        optimizer.step(trainset, teacher=teacher)

        # Frozen param should not be updated
        assert params[0].data == ""

    def test_is_success_with_threshold(self, optimizer):
        """Test _is_success with threshold."""
        optimizer.metric_threshold = 0.5

        assert optimizer._is_success(0.6) is True
        assert optimizer._is_success(0.5) is True
        assert optimizer._is_success(0.4) is False

    def test_is_success_without_threshold(self, optimizer):
        """Test _is_success without threshold."""
        optimizer.metric_threshold = None

        assert optimizer._is_success(1.0) is True
        assert optimizer._is_success(0.0) is False

    def test_step_with_closure(self, optimizer, trainset):
        """Test step with closure function."""
        closure_called = [False]

        def closure():
            closure_called[0] = True
            return 0.7

        teacher = MockModule(return_value="4")
        loss = optimizer.step(trainset, teacher=teacher, closure=closure)

        assert closure_called[0] is True
        assert loss == 0.7

    def test_traces_to_examples(self, optimizer):
        """Test converting traces to examples."""
        optimizer._traces = {
            "Module1": [
                Trace(
                    module_name="Module1",
                    inputs={"input": "question"},
                    outputs={"output": "answer"},
                )
            ]
        }

        examples = optimizer._traces_to_examples()

        assert len(examples) == 1
        assert examples[0].inputs == "question"
        assert examples[0].labels == "answer"
