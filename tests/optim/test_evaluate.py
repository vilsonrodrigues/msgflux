"""Tests for Evaluate class."""

import csv
import json
import time

import pytest

from msgflux.examples import Example
from msgflux.optim import evaluate as evaluate_module
from msgflux.optim.evaluate import EvalResult, Evaluate


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
        assert result.score == 100.0

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
        assert abs(result.score - 33.33) < 0.01

    def test_results_are_always_returned(self, devset):
        def program(inputs):
            return "4"

        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(program)
        assert len(result.results) == 3

    def test_failure_score_on_error(self, devset):
        def failing_program(inputs):
            raise ValueError("boom")

        evaluator = Evaluate(
            devset=devset, metric=exact_match, failure_score=-1.0
        )
        result = evaluator(failing_program)
        assert result.score == -100.0

    def test_empty_devset(self):
        evaluator = Evaluate(devset=[], metric=exact_match)
        result = evaluator(lambda x: x)
        assert result.score == 0.0
        assert result.results == []

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

    def test_parallel_results_preserve_devset_order(self, devset):
        def program(inputs):
            delays = {"2+2": 0.03, "3+3": 0.01, "1+1": 0.02}
            time.sleep(delays[inputs])
            mapping = {"2+2": "4", "3+3": "6", "1+1": "2"}
            return mapping[inputs]

        evaluator = Evaluate(
            devset=devset,
            metric=exact_match,
            num_threads=3,
        )
        result = evaluator(program)

        assert [example.inputs for example, _, _ in result.results] == [
            example.inputs for example in devset
        ]

    def test_calls_program_with_mapping_inputs(self):
        devset = [
            Example(inputs={"question": "2+2"}, labels="4"),
            Example(inputs={"question": "3+3"}, labels="6"),
        ]

        def program(*, question):
            return "4" if question == "2+2" else "6"

        evaluator = Evaluate(devset=devset, metric=exact_match)
        result = evaluator(program)

        assert result.score == 100.0

    def test_calls_trace_aware_metric(self, devset):
        seen = []

        def metric(example, prediction, trace=None):
            seen.append(trace)
            return exact_match(example, prediction)

        evaluator = Evaluate(devset=devset, metric=metric)
        evaluator(lambda inputs: "4" if inputs == "2+2" else "wrong")

        assert seen == [None, None, None]

    def test_saves_results_as_csv_and_json(self, devset, tmp_path):
        csv_path = tmp_path / "results.csv"
        json_path = tmp_path / "results.json"

        def program(inputs):
            return "4" if inputs == "2+2" else "wrong"

        evaluator = Evaluate(
            devset=devset,
            metric=exact_match,
            save_as_csv=str(csv_path),
            save_as_json=str(json_path),
        )
        evaluator(program)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        assert rows[0]["inputs"] == "2+2"
        assert rows[0]["prediction"] == "4"
        assert rows[0]["exact_match"] == "1.0"
        assert payload[1]["labels"] == "6"
        assert payload[1]["exact_match"] == 0.0

    def test_call_overrides_save_paths(self, devset, tmp_path):
        csv_path = tmp_path / "override.csv"
        json_path = tmp_path / "override.json"

        evaluator = Evaluate(devset=devset, metric=exact_match)
        evaluator(
            lambda inputs: "4",
            save_as_csv=str(csv_path),
            save_as_json=str(json_path),
        )

        assert csv_path.exists()
        assert json_path.exists()

    def test_display_table_uses_dataframe_helpers(self, devset, monkeypatch):
        evaluator = Evaluate(devset=devset, metric=exact_match, display_table=2)
        constructed = []
        displayed = []

        monkeypatch.setattr(
            evaluate_module.importlib.util,
            "find_spec",
            lambda name: object() if name == "pandas" else None,
        )
        monkeypatch.setattr(
            evaluator,
            "_construct_result_table",
            lambda results, metric_name: constructed.append(
                (results, metric_name)
            )
            or "table",
        )
        monkeypatch.setattr(
            evaluator,
            "_display_result_table",
            lambda result_df, display_table: displayed.append(
                (result_df, display_table)
            ),
        )

        evaluator(lambda inputs: "4")

        assert constructed
        assert constructed[0][1] == "exact_match"
        assert displayed == [("table", 2)]
