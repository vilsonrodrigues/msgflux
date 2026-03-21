"""Tests for the public msgflux.evaluate API."""

import logging

import pytest

from msgflux.evaluate import EvalResult, Evaluate, exact_match
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
        if isinstance(inputs, str) and inputs in self.responses:
            return self.responses[inputs]
        return "default response"


class KeywordModule(Module):
    """Mock module that expects keyword inputs."""

    def forward(self, *, question):
        return "4" if question == "What is 2+2?" else "6"


class TestEvalResult:
    def test_creation(self):
        result = EvalResult(
            score=75.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
                (Example(inputs="q2", labels="a2"), "wrong", 0.0),
            ],
            num_errors=1,
            metadata={"kind": "test"},
        )

        assert result.score == 75.0
        assert len(result.results) == 2
        assert result.num_errors == 1
        assert result.metadata == {"kind": "test"}

    def test_repr(self):
        result = EvalResult(score=85.5, results=[])
        assert "85.5" in repr(result)
        assert "EvalResult" in repr(result)

    def test_success_failure_helpers(self):
        result = EvalResult(
            score=50.0,
            results=[
                (Example(inputs="q1", labels="a1"), "a1", 1.0),
                (Example(inputs="q2", labels="a2"), "wrong", 0.0),
                (Example(inputs="q3", labels="a3"), "a3", 0.8),
            ],
        )

        assert len(result.get_successes()) == 2
        assert len(result.get_failures()) == 1

    def test_to_dict(self):
        result = EvalResult(
            score=75.0,
            results=[(Example(inputs="q1", labels="a1"), "a1", 1.0)],
            num_errors=1,
            metadata={"metric_name": "exact_match"},
        )

        payload = result.to_dict()

        assert payload["score"] == 75.0
        assert payload["num_examples"] == 1
        assert payload["num_errors"] == 1
        assert payload["metadata"] == {"metric_name": "exact_match"}


class TestEvaluateFacade:
    @pytest.fixture
    def devset(self):
        return [
            Example(inputs="What is 2+2?", labels="4"),
            Example(inputs="What is 3+3?", labels="6"),
            Example(inputs="What is 4+4?", labels="8"),
            Example(inputs="What is 5+5?", labels="10"),
        ]

    @pytest.fixture
    def module(self):
        return MockModule(
            responses={
                "What is 2+2?": "4",
                "What is 3+3?": "6",
                "What is 4+4?": "8",
                "What is 5+5?": "10",
            }
        )

    def test_evaluate_all_correct(self, module, devset):
        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(module)

        assert result.score == 100.0
        assert len(result.results) == 4
        assert result.num_errors == 0
        assert result.metadata["metric_name"] == "exact_match"

    def test_evaluate_all_wrong(self, devset):
        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(MockModule())
        assert result.score == 0.0

    def test_evaluate_partial_correct(self):
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
        evaluator = Evaluate(devset=devset, metric=exact_match)

        result = evaluator(module)

        assert result.score == 50.0

    def test_evaluate_empty_devset(self):
        evaluator = Evaluate(devset=[], metric=exact_match)
        result = evaluator(MockModule())

        assert result.score == 0.0
        assert len(result.results) == 0

    def test_evaluate_restores_training_mode(self, module, devset):
        evaluator = Evaluate(devset=devset, metric=exact_match)
        module.train()

        evaluator(module)

        assert module.training is True

    def test_evaluate_with_errors(self, devset):
        evaluator = Evaluate(
            devset=devset,
            metric=exact_match,
            max_errors=10,
            failure_score=0.0,
        )
        result = evaluator(MockModule(raise_error=True))

        assert result.num_errors == 4
        assert result.score == 0.0

    def test_evaluate_stops_after_max_errors(self, devset):
        evaluator = Evaluate(
            devset=devset,
            metric=exact_match,
            max_errors=2,
            failure_score=0.0,
        )
        result = evaluator(MockModule(raise_error=True))

        assert len(result.results) == 2
        assert result.num_errors == 2

    def test_evaluate_mapping_inputs(self):
        devset = [
            Example(inputs={"question": "What is 2+2?"}, labels="4"),
            Example(inputs={"question": "What is 3+3?"}, labels="6"),
        ]
        evaluator = Evaluate(devset=devset, metric=exact_match)

        result = evaluator(KeywordModule())

        assert result.score == 100.0

    def test_parallel_evaluation(self, devset, module):
        evaluator = Evaluate(devset=devset, metric=exact_match, num_threads=2)

        result = evaluator(module)

        assert result.score == 100.0
        assert module.call_count == 4

    def test_display_progress_path(self, devset, module, caplog):
        evaluator = Evaluate(
            devset=devset,
            metric=exact_match,
            display_progress=True,
        )

        with caplog.at_level(logging.INFO):
            result = evaluator(module)

        assert result.score == 100.0
