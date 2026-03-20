"""Tests for Evaluate class."""

import pytest

from msgflux.examples import Example
from msgflux.optim.evaluate import Evaluate, EvalResult


def exact_match(example, prediction):
    return 1.0 if str(prediction) == str(example.labels) else 0.0


class TestEvaluate:
    @pytest.fixture
    def devset(self):
        return [
            Example(inputs="2+2", labels="4"),
            Example(inputs="3+3", labels="6"),
            Example(inputs="1+1", labels="2"),
        ]

    def test_perfect_score(self, devset):
        def perfect_program(inputs):
            mapping = {"2+2": "4", "3+3": "6", "1+1": "2"}
            return mapping.get(inputs, "?")

        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(perfect_program)
        assert result.score == 1.0

    def test_zero_score(self, devset):
        def bad_program(inputs):
            return "wrong"

        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(bad_program)
        assert result.score == 0.0

    def test_partial_score(self, devset):
        def partial_program(inputs):
            return "4" if inputs == "2+2" else "wrong"

        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(partial_program)
        assert abs(result.score - 1.0 / 3.0) < 0.01

    def test_return_all_scores(self, devset):
        def program(inputs):
            return "4"

        evaluator = Evaluate(
            devset=devset, metric=exact_match, return_all_scores=True
        )
        result = evaluator(program)
        assert len(result.results) == 3

    def test_failure_score_on_error(self, devset):
        def failing_program(inputs):
            raise ValueError("boom")

        evaluator = Evaluate(
            devset=devset, metric=exact_match, failure_score=-1.0
        )
        result = evaluator(failing_program)
        assert result.score == -1.0

    def test_empty_devset(self):
        evaluator = Evaluate(devset=[], metric=exact_match)
        result = evaluator(lambda x: x)
        assert result.score == 0.0

    def test_custom_devset_override(self, devset):
        evaluator = Evaluate(devset=[], metric=exact_match)
        result = evaluator(lambda x: "4", devset=devset)
        assert result.score > 0

    def test_multithreaded(self, devset):
        def program(inputs):
            return "4" if inputs == "2+2" else "wrong"

        evaluator = Evaluate(
            devset=devset, metric=exact_match, num_threads=2
        )
        result = evaluator(program)
        assert isinstance(result, EvalResult)
