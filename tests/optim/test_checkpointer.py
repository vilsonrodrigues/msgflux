"""Tests for Checkpointer."""

import json
import tempfile
from pathlib import Path

import pytest

from msgflux.optim.checkpointer import Checkpointer, CheckpointInfo


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self._state = {"value": 0}

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state_dict):
        self._state = state_dict.copy()


class TestCheckpointInfo:
    def test_creation(self):
        info = CheckpointInfo(
            path=Path("/tmp/test.pt"),
            step=10,
            score=0.85,
            timestamp="2024-01-01T00:00:00",
            metadata={"key": "value"},
        )
        assert info.step == 10
        assert info.score == 0.85
        assert info.metadata == {"key": "value"}


class TestCheckpointerInit:
    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            assert not checkpoint_dir.exists()

            Checkpointer(checkpoint_dir, verbose=False)

            assert checkpoint_dir.exists()

    def test_default_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)

            assert cp.save_every is None
            assert cp.keep_last == 5
            assert cp.save_best is True
            assert cp.mode == "max"

    def test_custom_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(
                tmpdir,
                save_every=10,
                keep_last=3,
                save_best=False,
                mode="min",
                verbose=False,
            )

            assert cp.save_every == 10
            assert cp.keep_last == 3
            assert cp.save_best is False
            assert cp.mode == "min"

    def test_invalid_mode_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
                Checkpointer(tmpdir, mode="invalid")


class TestCheckpointerSave:
    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()
            optimizer._state = {"value": 42}

            cp._current_step = 1
            info = cp.save(optimizer, score=0.85)

            assert info.path.exists()
            assert info.step == 1
            assert info.score == 0.85

    def test_save_increments_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)

            assert cp.current_step == 2

    def test_save_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_best=True, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.8, force_save=True)  # Best
            cp.step(optimizer, score=0.6, force_save=True)

            best_path = Path(tmpdir) / "best.pt"
            assert best_path.exists()
            assert cp.best_checkpoint is not None
            assert cp.best_checkpoint.score == 0.8


class TestCheckpointerLoad:
    def test_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()
            optimizer._state = {"value": 42}

            cp._current_step = 1
            info = cp.save(optimizer, score=0.85)

            loaded = cp.load(info.path)

            assert loaded["optimizer_state"]["value"] == 42
            assert loaded["score"] == 0.85
            assert loaded["step"] == 1

    def test_load_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            optimizer._state = {"value": 1}
            cp.step(optimizer, score=0.5, force_save=True)

            optimizer._state = {"value": 2}
            cp.step(optimizer, score=0.6, force_save=True)

            optimizer._state = {"value": 3}
            cp.step(optimizer, score=0.7, force_save=True)

            loaded = cp.load_latest()

            assert loaded is not None
            assert loaded["optimizer_state"]["value"] == 3
            assert loaded["step"] == 3

    def test_load_best(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_best=True, verbose=False)
            optimizer = MockOptimizer()

            optimizer._state = {"value": 1}
            cp.step(optimizer, score=0.5, force_save=True)

            optimizer._state = {"value": 2}
            cp.step(optimizer, score=0.9, force_save=True)  # Best

            optimizer._state = {"value": 3}
            cp.step(optimizer, score=0.6, force_save=True)

            loaded = cp.load_best()

            assert loaded is not None
            assert loaded["optimizer_state"]["value"] == 2

    def test_load_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            optimizer._state = {"value": 1}
            cp.step(optimizer, score=0.5, force_save=True)

            optimizer._state = {"value": 2}
            cp.step(optimizer, score=0.6, force_save=True)

            loaded = cp.load_step(1)

            assert loaded is not None
            assert loaded["optimizer_state"]["value"] == 1

    def test_load_latest_empty_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            assert cp.load_latest() is None

    def test_load_best_empty_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            assert cp.load_best() is None


class TestCheckpointerStep:
    def test_step_saves_at_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Disable save_best to test only save_every behavior
            cp = Checkpointer(tmpdir, save_every=3, save_best=False, verbose=False)
            optimizer = MockOptimizer()

            result1 = cp.step(optimizer, score=0.5)  # Step 1
            result2 = cp.step(optimizer, score=0.6)  # Step 2
            result3 = cp.step(optimizer, score=0.7)  # Step 3 - should save

            assert result1 is None
            assert result2 is None
            assert result3 is not None
            assert result3.step == 3

    def test_step_saves_best_regardless_of_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_every=10, save_best=True, verbose=False)
            optimizer = MockOptimizer()

            result = cp.step(optimizer, score=0.9)  # First is always best

            assert result is not None

    def test_force_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_every=100, verbose=False)
            optimizer = MockOptimizer()

            result = cp.step(optimizer, score=0.5, force_save=True)

            assert result is not None


class TestCheckpointerCleanup:
    def test_keeps_only_last_n_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, keep_last=2, save_best=False, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)
            cp.step(optimizer, score=0.7, force_save=True)
            cp.step(optimizer, score=0.8, force_save=True)

            checkpoints = cp.list_checkpoints()
            assert len(checkpoints) == 2

            # Should keep the most recent
            steps = [c.step for c in checkpoints]
            assert 4 in steps
            assert 3 in steps

    def test_keeps_best_even_if_old(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(
                tmpdir, keep_last=2, save_best=True, mode="max", verbose=False
            )
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.9, force_save=True)  # Best
            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)
            cp.step(optimizer, score=0.7, force_save=True)

            # Best checkpoint should still exist
            best_path = Path(tmpdir) / "best.pt"
            assert best_path.exists()


class TestCheckpointerMinMode:
    def test_lower_is_better(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_best=True, mode="min", verbose=False)
            optimizer = MockOptimizer()

            optimizer._state = {"value": 1}
            cp.step(optimizer, score=0.5, force_save=True)

            optimizer._state = {"value": 2}
            cp.step(optimizer, score=0.3, force_save=True)  # Better (lower)

            optimizer._state = {"value": 3}
            cp.step(optimizer, score=0.4, force_save=True)

            loaded = cp.load_best()
            assert loaded["optimizer_state"]["value"] == 2


class TestCheckpointerIndex:
    def test_index_file_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)

            index_path = Path(tmpdir) / "checkpoints.json"
            assert index_path.exists()

    def test_index_contains_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)

            index_path = Path(tmpdir) / "checkpoints.json"
            with open(index_path) as f:
                index = json.load(f)

            assert len(index["checkpoints"]) == 2
            assert index["current_step"] == 2

    def test_loads_index_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpointer and save
            cp1 = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            cp1.step(optimizer, score=0.5, force_save=True)
            cp1.step(optimizer, score=0.6, force_save=True)

            # Create new checkpointer and verify it loads state
            cp2 = Checkpointer(tmpdir, verbose=False)

            assert len(cp2.list_checkpoints()) == 2
            assert cp2.current_step == 2


class TestCheckpointerProperties:
    def test_latest_checkpoint_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, verbose=False)
            optimizer = MockOptimizer()

            assert cp.latest_checkpoint is None

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)

            assert cp.latest_checkpoint is not None
            assert cp.latest_checkpoint.step == 2

    def test_best_checkpoint_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_best=True, verbose=False)
            optimizer = MockOptimizer()

            assert cp.best_checkpoint is None

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.9, force_save=True)

            assert cp.best_checkpoint is not None
            assert cp.best_checkpoint.score == 0.9


class TestCheckpointerClear:
    def test_clear_removes_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = Checkpointer(tmpdir, save_best=True, verbose=False)
            optimizer = MockOptimizer()

            cp.step(optimizer, score=0.5, force_save=True)
            cp.step(optimizer, score=0.6, force_save=True)

            cp.clear()

            assert len(cp.list_checkpoints()) == 0
            assert cp.current_step == 0
            assert cp.best_checkpoint is None

            # Files should be deleted
            files = list(Path(tmpdir).glob("*.pt"))
            assert len(files) == 0
