"""Tests for ExperimentExporter."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from msgflux.optim.export import ExperimentExporter, ExperimentRecord, StepRecord


class TestStepRecord:
    def test_creation(self):
        record = StepRecord(step=10, score=0.85, best_score=0.85)
        assert record.step == 10
        assert record.score == 0.85
        assert record.best_score == 0.85
        assert record.timestamp is not None

    def test_with_metadata(self):
        record = StepRecord(step=1, score=0.5, metadata={"key": "value"})
        assert record.metadata == {"key": "value"}

    def test_optional_fields(self):
        record = StepRecord(step=1)
        assert record.score is None
        assert record.best_score is None
        assert record.metadata == {}


class TestExperimentRecord:
    def test_creation(self):
        record = ExperimentRecord(
            name="test-experiment",
            optimizer="MIPROv2",
            started_at="2024-01-01T00:00:00",
        )
        assert record.name == "test-experiment"
        assert record.optimizer == "MIPROv2"
        assert record.started_at == "2024-01-01T00:00:00"

    def test_default_values(self):
        record = ExperimentRecord(
            name="test",
            optimizer="test",
            started_at="2024-01-01T00:00:00",
        )
        assert record.finished_at is None
        assert record.final_score is None
        assert record.config == {}
        assert record.steps == []


class TestExperimentExporterInit:
    def test_default_initialization(self):
        exporter = ExperimentExporter()
        assert exporter.record.name.startswith("experiment_")
        assert exporter.record.optimizer == "unknown"
        assert exporter.record.config == {}

    def test_custom_initialization(self):
        exporter = ExperimentExporter(
            name="my-experiment",
            optimizer="MIPROv2",
            config={"param1": 10, "param2": "value"},
        )
        assert exporter.record.name == "my-experiment"
        assert exporter.record.optimizer == "MIPROv2"
        assert exporter.record.config == {"param1": 10, "param2": "value"}


class TestExperimentExporterSetMode:
    def test_set_max_mode(self):
        exporter = ExperimentExporter()
        exporter.set_mode("max")
        assert exporter._mode == "max"

    def test_set_min_mode(self):
        exporter = ExperimentExporter()
        exporter.set_mode("min")
        assert exporter._mode == "min"

    def test_invalid_mode_raises(self):
        exporter = ExperimentExporter()
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            exporter.set_mode("invalid")


class TestExperimentExporterLogStep:
    def test_log_step(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)

        assert len(exporter.record.steps) == 1
        assert exporter.record.steps[0].step == 1
        assert exporter.record.steps[0].score == 0.5

    def test_log_step_auto_tracks_best_score_max(self):
        exporter = ExperimentExporter()
        exporter.set_mode("max")

        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.8)  # Better
        exporter.log_step(3, score=0.6)  # Worse

        assert exporter.record.steps[0].best_score == 0.5
        assert exporter.record.steps[1].best_score == 0.8
        assert exporter.record.steps[2].best_score == 0.8

    def test_log_step_auto_tracks_best_score_min(self):
        exporter = ExperimentExporter()
        exporter.set_mode("min")

        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.3)  # Better (lower)
        exporter.log_step(3, score=0.4)  # Worse

        assert exporter.record.steps[0].best_score == 0.5
        assert exporter.record.steps[1].best_score == 0.3
        assert exporter.record.steps[2].best_score == 0.3

    def test_log_step_with_explicit_best_score(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5, best_score=0.9)

        assert exporter.record.steps[0].best_score == 0.9

    def test_log_step_with_metadata(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5, learning_rate=0.01, batch_size=32)

        assert exporter.record.steps[0].metadata["learning_rate"] == 0.01
        assert exporter.record.steps[0].metadata["batch_size"] == 32


class TestExperimentExporterLogTrial:
    def test_log_trial(self):
        exporter = ExperimentExporter()
        exporter.log_trial(1, score=0.7)

        assert len(exporter.record.steps) == 1
        assert exporter.record.steps[0].metadata["trial"] == 1
        assert exporter.record.steps[0].score == 0.7

    def test_log_trial_with_metadata(self):
        exporter = ExperimentExporter()
        exporter.log_trial(1, score=0.7, prompt_template="v1")

        assert exporter.record.steps[0].metadata["trial"] == 1
        assert exporter.record.steps[0].metadata["prompt_template"] == "v1"


class TestExperimentExporterLogGeneration:
    def test_log_generation(self):
        exporter = ExperimentExporter()
        exporter.log_generation(1, best_score=0.8, avg_score=0.65, population_size=20)

        assert len(exporter.record.steps) == 1
        assert exporter.record.steps[0].metadata["generation"] == 1
        assert exporter.record.steps[0].metadata["avg_score"] == 0.65
        assert exporter.record.steps[0].metadata["population_size"] == 20
        assert exporter.record.steps[0].score == 0.8

    def test_log_generation_without_optional_fields(self):
        exporter = ExperimentExporter()
        exporter.log_generation(1, best_score=0.8)

        assert "avg_score" not in exporter.record.steps[0].metadata
        assert "population_size" not in exporter.record.steps[0].metadata


class TestExperimentExporterFinish:
    def test_finish_sets_finished_at(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.finish()

        assert exporter.record.finished_at is not None

    def test_finish_with_explicit_final_score(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.finish(final_score=0.85)

        assert exporter.record.final_score == 0.85

    def test_finish_uses_best_score_as_final(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.9)
        exporter.log_step(3, score=0.7)
        exporter.finish()

        assert exporter.record.final_score == 0.9


class TestExperimentExporterSaveJson:
    def test_save_json_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter(name="test", optimizer="test")
            exporter.log_step(1, score=0.5)
            exporter.finish()

            path = Path(tmpdir) / "results.json"
            exporter.save_json(str(path))

            assert path.exists()

    def test_save_json_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter(name="test", optimizer="MIPROv2")
            exporter.log_step(1, score=0.5)
            exporter.finish(final_score=0.5)

            path = Path(tmpdir) / "results.json"
            exporter.save_json(str(path))

            with open(path) as f:
                data = json.load(f)

            assert data["name"] == "test"
            assert data["optimizer"] == "MIPROv2"
            assert len(data["steps"]) == 1
            assert data["final_score"] == 0.5

    def test_save_json_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter()
            exporter.log_step(1, score=0.5)

            path = Path(tmpdir) / "subdir" / "results.json"
            exporter.save_json(str(path))

            assert path.exists()


class TestExperimentExporterSaveCsv:
    def test_save_csv_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter()
            exporter.log_step(1, score=0.5)

            path = Path(tmpdir) / "results.csv"
            exporter.save_csv(str(path))

            assert path.exists()

    def test_save_csv_does_nothing_if_no_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter()

            path = Path(tmpdir) / "results.csv"
            exporter.save_csv(str(path))

            assert not path.exists()

    def test_save_csv_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter()
            exporter.log_step(1, score=0.5)
            exporter.log_step(2, score=0.7)

            path = Path(tmpdir) / "results.csv"
            exporter.save_csv(str(path))

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 3  # Header + 2 data rows
            assert "step" in lines[0]
            assert "score" in lines[0]
            assert "best_score" in lines[0]

    def test_save_csv_includes_metadata_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter()
            exporter.log_step(1, score=0.5, custom_field="value1")
            exporter.log_step(2, score=0.7, custom_field="value2")

            path = Path(tmpdir) / "results.csv"
            exporter.save_csv(str(path))

            with open(path) as f:
                content = f.read()

            assert "custom_field" in content
            assert "value1" in content
            assert "value2" in content


class TestExperimentExporterToDict:
    def test_to_dict(self):
        exporter = ExperimentExporter(name="test", optimizer="MIPROv2")
        exporter.log_step(1, score=0.5)

        data = exporter.to_dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["optimizer"] == "MIPROv2"
        assert len(data["steps"]) == 1


class TestExperimentExporterGetMethods:
    def test_get_scores(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.7)
        exporter.log_step(3)  # No score

        scores = exporter.get_scores()

        assert len(scores) == 2
        assert 0.5 in scores
        assert 0.7 in scores

    def test_get_best_scores(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.log_step(2, score=0.7)

        best_scores = exporter.get_best_scores()

        assert len(best_scores) == 2
        assert 0.5 in best_scores
        assert 0.7 in best_scores

    def test_get_steps(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.log_step(5, score=0.7)
        exporter.log_step(10, score=0.9)

        steps = exporter.get_steps()

        assert steps == [1, 5, 10]


class TestExperimentExporterProperties:
    def test_best_score_property(self):
        exporter = ExperimentExporter()
        assert exporter.best_score is None

        exporter.log_step(1, score=0.5)
        assert exporter.best_score == 0.5

        exporter.log_step(2, score=0.9)
        assert exporter.best_score == 0.9

        exporter.log_step(3, score=0.7)
        assert exporter.best_score == 0.9

    def test_num_steps_property(self):
        exporter = ExperimentExporter()
        assert exporter.num_steps == 0

        exporter.log_step(1, score=0.5)
        assert exporter.num_steps == 1

        exporter.log_step(2, score=0.7)
        assert exporter.num_steps == 2

    def test_duration_property_not_finished(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)

        assert exporter.duration is None

    def test_duration_property_finished(self):
        exporter = ExperimentExporter()
        exporter.log_step(1, score=0.5)
        exporter.finish()

        # Duration should be a positive number (or zero)
        assert exporter.duration is not None
        assert exporter.duration >= 0


class TestExperimentExporterSummary:
    def test_summary(self):
        exporter = ExperimentExporter(name="test", optimizer="MIPROv2")
        exporter.log_step(1, score=0.3)
        exporter.log_step(2, score=0.5)
        exporter.log_step(3, score=0.9)
        exporter.finish()

        summary = exporter.summary()

        assert summary["name"] == "test"
        assert summary["optimizer"] == "MIPROv2"
        assert summary["num_steps"] == 3
        assert summary["best_score"] == 0.9
        assert summary["min_score"] == 0.3
        assert summary["max_score"] == 0.9
        # Average of 0.3, 0.5, 0.9 = 0.567 (approximately)
        assert 0.56 < summary["avg_score"] < 0.58
        assert summary["duration_seconds"] is not None

    def test_summary_empty(self):
        exporter = ExperimentExporter()

        summary = exporter.summary()

        assert summary["num_steps"] == 0
        assert summary["avg_score"] is None
        assert summary["min_score"] is None
        assert summary["max_score"] is None

