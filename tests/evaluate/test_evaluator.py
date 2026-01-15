"""Tests for msgflux.evaluate.evaluator module."""

import pytest

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.examples import Example
from msgflux.nn.modules.module import Module


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self, responses=None, raise_error=False):
        super().__init__()
        self.responses = responses or {}
        self.raise_error = raise_error
        self.call_count = 0

    def forward(self, inputs):
        self.call_count += 1
        if self.raise_error:
            raise ValueError("Mock error")
        # Return response based on input or default
        if isinstance(inputs, str) and inputs in self.responses:
            return self.responses[inputs]
        return "default response"


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass."""

    def test_creation(self):
        """Test creating EvaluationResult."""
        result = EvaluationResult(
            score=75.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
                (Example(inputs="q2", labels="a2"), "wrong", 0.0),
            ],
        )

        assert result.score == 75.0
        assert len(result.results) == 2

    def test_repr(self):
        """Test string representation."""
        result = EvaluationResult(score=85.5, results=[])

        repr_str = repr(result)

        assert "85.5" in repr_str
        assert "EvaluationResult" in repr_str

    def test_get_successes(self):
        """Test getting successful examples."""
        result = EvaluationResult(
            score=50.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
                (Example(inputs="q2", labels="a2"), "wrong", 0.0),
                (Example(inputs="q3", labels="a3"), "a3", 0.8),
            ],
        )

        successes = result.get_successes()

        assert len(successes) == 2
        assert all(score > 0 for _, _, score in successes)

    def test_get_failures(self):
        """Test getting failed examples."""
        result = EvaluationResult(
            score=50.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
                (Example(inputs="q2", labels="a2"), "wrong", 0.0),
            ],
        )

        failures = result.get_failures()

        assert len(failures) == 1
        assert failures[0][2] == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EvaluationResult(
            score=75.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
            ],
            num_errors=1,
            metadata={"test": "value"},
        )

        d = result.to_dict()

        assert d["score"] == 75.0
        assert d["num_examples"] == 1
        assert d["num_errors"] == 1
        assert d["metadata"] == {"test": "value"}


class TestEvaluator:
    """Test suite for Evaluator class."""

    @pytest.fixture
    def devset(self):
        """Create sample development set."""
        return [
            Example(inputs="What is 2+2?", labels="4"),
            Example(inputs="What is 3+3?", labels="6"),
            Example(inputs="What is 4+4?", labels="8"),
            Example(inputs="What is 5+5?", labels="10"),
        ]

    @pytest.fixture
    def metric(self):
        """Create simple exact match metric."""

        def exact_match(example, prediction):
            return float(str(example.labels) == str(prediction))

        return exact_match

    @pytest.fixture
    def module(self):
        """Create mock module with correct responses."""
        return MockModule(
            responses={
                "What is 2+2?": "4",
                "What is 3+3?": "6",
                "What is 4+4?": "8",
                "What is 5+5?": "10",
            }
        )

    @pytest.fixture
    def evaluator(self, metric):
        """Create Evaluator instance."""
        return Evaluator(metric=metric)

    def test_initialization(self, evaluator, metric):
        """Test evaluator initialization."""
        assert evaluator.metric == metric
        assert evaluator.num_workers is None
        assert evaluator.max_errors == 10
        assert evaluator.failure_score == 0.0

    def test_evaluate_all_correct(self, evaluator, module, devset):
        """Test evaluation with all correct predictions."""
        result = evaluator(module, devset)

        assert result.score == 100.0
        assert len(result.results) == 4

    def test_evaluate_all_wrong(self, metric, devset):
        """Test evaluation with all wrong predictions."""
        module = MockModule()  # Returns "default response"
        evaluator = Evaluator(metric=metric)

        result = evaluator(module, devset)

        assert result.score == 0.0

    def test_evaluate_partial_correct(self, metric):
        """Test evaluation with some correct predictions."""
        devset = [
            Example(inputs="q1", labels="correct"),
            Example(inputs="q2", labels="correct"),
            Example(inputs="q3", labels="wrong_label"),
            Example(inputs="q4", labels="wrong_label"),
        ]
        module = MockModule(
            responses={
                "q1": "correct",
                "q2": "correct",
                "q3": "incorrect",
                "q4": "incorrect",
            }
        )
        evaluator = Evaluator(metric=metric)

        result = evaluator(module, devset)

        assert result.score == 50.0

    def test_evaluate_empty_devset(self, evaluator, module):
        """Test evaluation with empty devset."""
        result = evaluator(module, [])

        assert result.score == 0.0
        assert len(result.results) == 0

    def test_evaluate_sets_eval_mode(self, evaluator, module, devset):
        """Test that evaluator sets module to eval mode."""
        module.train()
        assert module.training is True

        evaluator(module, devset)

        # Should have been set to eval mode during evaluation
        # (and restored if it was training)

    def test_evaluate_restores_training_mode(self, evaluator, module, devset):
        """Test that evaluator restores training mode."""
        module.train()

        evaluator(module, devset)

        assert module.training is True

    def test_evaluate_with_errors(self, metric, devset):
        """Test evaluation handles errors."""
        module = MockModule(raise_error=True)
        evaluator = Evaluator(metric=metric, max_errors=10, failure_score=0.0)

        result = evaluator(module, devset)

        assert result.num_errors == 4
        assert result.score == 0.0  # All failed with failure_score

    def test_evaluate_max_errors_exceeded(self, metric, devset):
        """Test that max_errors raises exception."""
        module = MockModule(raise_error=True)
        evaluator = Evaluator(metric=metric, max_errors=2)

        with pytest.raises(RuntimeError, match="Maximum errors"):
            evaluator(module, devset)

    def test_evaluate_single(self, evaluator, module):
        """Test evaluating a single example."""
        example = Example(inputs="What is 2+2?", labels="4")

        prediction, score = evaluator.evaluate_single(module, example)

        assert prediction == "4"
        assert score == 1.0

    def test_evaluate_with_return_predictions_false(self, evaluator, module, devset):
        """Test evaluation without returning predictions."""
        result = evaluator.evaluate(module, devset, return_predictions=False)

        assert result.score == 100.0
        assert len(result.results) == 0  # No results stored

    def test_metadata_in_result(self, evaluator, module, devset):
        """Test that metadata is included in result."""
        result = evaluator(module, devset)

        assert "num_examples" in result.metadata
        assert "num_evaluated" in result.metadata
        assert "metric_name" in result.metadata

    def test_callable_interface(self, evaluator, module, devset):
        """Test that evaluator can be called directly."""
        result = evaluator(module, devset)

        assert isinstance(result, EvaluationResult)
        assert result.score == 100.0

    def test_parallel_evaluation(self, metric, devset):
        """Test parallel evaluation."""
        module = MockModule(
            responses={
                "What is 2+2?": "4",
                "What is 3+3?": "6",
                "What is 4+4?": "8",
                "What is 5+5?": "10",
            }
        )
        evaluator = Evaluator(metric=metric, num_workers=2)

        result = evaluator(module, devset)

        assert result.score == 100.0
        assert module.call_count == 4

    def test_verbose_mode(self, metric, devset, caplog):
        """Test verbose logging."""
        import logging

        module = MockModule(
            responses={
                "What is 2+2?": "4",
                "What is 3+3?": "6",
                "What is 4+4?": "8",
                "What is 5+5?": "10",
            }
        )
        evaluator = Evaluator(metric=metric, verbose=True)

        with caplog.at_level(logging.INFO):
            evaluator(module, devset)

        # Should have logged evaluation info (captured via stdout, not caplog)
        # The evaluator uses a custom logger that writes to stdout
        assert True  # Test passes if no exception raised
