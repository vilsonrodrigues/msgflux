"""Tests for Callbacks system."""

import tempfile
from pathlib import Path

import pytest

from msgflux.optim.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    ProgressCallback,
)


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self._step_count = 0
        self._state = {"value": 0}

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state_dict):
        self._state = state_dict.copy()


class TestCallback:
    def test_set_optimizer(self):
        class TestCallback(Callback):
            pass

        cb = TestCallback()
        optimizer = MockOptimizer()

        cb.set_optimizer(optimizer)

        assert cb.optimizer is optimizer

    def test_default_methods_return_false(self):
        class TestCallback(Callback):
            pass

        cb = TestCallback()

        assert cb.on_step_end(1) is False
        assert cb.on_trial_end(1, 0.5) is False
        assert cb.on_generation_end(1, 0.5) is False

    def test_default_methods_do_nothing(self):
        class TestCallback(Callback):
            pass

        cb = TestCallback()

        # Should not raise
        cb.on_optimization_begin()
        cb.on_optimization_end()
        cb.on_step_begin(1)
        cb.on_trial_begin(1)
        cb.on_batch_begin(1)
        cb.on_batch_end(1)
        cb.on_generation_begin(1)


class TestCallbackList:
    def test_empty_list(self):
        cl = CallbackList()
        assert len(cl) == 0

    def test_append(self):
        cl = CallbackList()
        cl.append(Callback())
        assert len(cl) == 1

    def test_extend(self):
        cl = CallbackList()
        cl.extend([Callback(), Callback()])
        assert len(cl) == 2

    def test_set_optimizer_on_all(self):
        class TrackedCallback(Callback):
            pass

        cb1 = TrackedCallback()
        cb2 = TrackedCallback()
        cl = CallbackList([cb1, cb2])
        optimizer = MockOptimizer()

        cl.set_optimizer(optimizer)

        assert cb1.optimizer is optimizer
        assert cb2.optimizer is optimizer

    def test_on_step_end_returns_true_if_any_returns_true(self):
        class StopCallback(Callback):
            def on_step_end(self, step, logs=None):
                return True

        class ContinueCallback(Callback):
            def on_step_end(self, step, logs=None):
                return False

        cl = CallbackList([ContinueCallback(), StopCallback()])

        result = cl.on_step_end(1)
        assert result is True

    def test_on_step_end_returns_false_if_none_returns_true(self):
        class ContinueCallback(Callback):
            def on_step_end(self, step, logs=None):
                return False

        cl = CallbackList([ContinueCallback(), ContinueCallback()])

        result = cl.on_step_end(1)
        assert result is False

    def test_logs_accumulate(self):
        class LoggingCallback(Callback):
            def on_step_end(self, step, logs=None):
                logs["step_callback"] = step
                return False

        cb = LoggingCallback()
        cl = CallbackList([cb])

        cl.on_step_end(5, {"initial": "value"})

        assert cl.logs["initial"] == "value"
        assert cl.logs["step_callback"] == 5

    def test_on_trial_end_passes_score(self):
        class ScoreCallback(Callback):
            def __init__(self):
                self.received_score = None

            def on_trial_end(self, trial, score, logs=None):
                self.received_score = score
                return False

        cb = ScoreCallback()
        cl = CallbackList([cb])

        cl.on_trial_end(1, 0.85)

        assert cb.received_score == 0.85

    def test_iteration(self):
        cb1 = Callback()
        cb2 = Callback()
        cl = CallbackList([cb1, cb2])

        callbacks = list(cl)
        assert len(callbacks) == 2
        assert cb1 in callbacks
        assert cb2 in callbacks


class TestEarlyStoppingCallback:
    def test_initialization(self):
        cb = EarlyStoppingCallback(patience=10, min_delta=0.01, mode="min")
        assert cb._early_stopping.patience == 10
        assert cb._early_stopping.min_delta == 0.01
        assert cb._early_stopping.mode == "min"

    def test_on_trial_end_returns_true_when_stopped(self):
        cb = EarlyStoppingCallback(patience=2, verbose=False)
        optimizer = MockOptimizer()
        cb.set_optimizer(optimizer)

        cb.on_trial_end(1, 0.5)
        cb.on_trial_end(2, 0.4)  # No improvement
        result = cb.on_trial_end(3, 0.3)  # No improvement, should stop

        assert result is True
        assert cb.stopped is True

    def test_on_step_end_with_score_in_logs(self):
        cb = EarlyStoppingCallback(patience=2, verbose=False)
        optimizer = MockOptimizer()
        cb.set_optimizer(optimizer)

        cb.on_step_end(1, {"score": 0.5})
        cb.on_step_end(2, {"score": 0.4})
        result = cb.on_step_end(3, {"score": 0.3})

        assert result is True

    def test_on_generation_end(self):
        cb = EarlyStoppingCallback(patience=2, verbose=False)
        optimizer = MockOptimizer()
        cb.set_optimizer(optimizer)

        cb.on_generation_end(1, 0.5)
        cb.on_generation_end(2, 0.4)
        result = cb.on_generation_end(3, 0.3)

        assert result is True

    def test_properties(self):
        cb = EarlyStoppingCallback(patience=5, verbose=False)
        optimizer = MockOptimizer()
        cb.set_optimizer(optimizer)

        cb.on_trial_end(1, 0.8)

        assert cb.best_score == 0.8
        assert cb.best_step == 1
        assert cb.stopped is False

    def test_reset(self):
        cb = EarlyStoppingCallback(patience=1, verbose=False)
        optimizer = MockOptimizer()
        cb.set_optimizer(optimizer)

        cb.on_trial_end(1, 0.5)
        cb.on_trial_end(2, 0.4)

        assert cb.stopped is True

        cb.reset()

        assert cb.stopped is False


class TestCheckpointCallback:
    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(tmpdir, save_every=5, keep_last=3)
            assert cb._checkpointer.save_every == 5
            assert cb._checkpointer.keep_last == 3

    def test_on_step_end_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(tmpdir, save_every=1, verbose=False)
            optimizer = MockOptimizer()
            cb.set_optimizer(optimizer)

            cb.on_step_end(1, {"score": 0.5})

            checkpoints = cb.checkpointer.list_checkpoints()
            assert len(checkpoints) >= 1

    def test_on_trial_end_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(tmpdir, save_every=1, verbose=False)
            optimizer = MockOptimizer()
            cb.set_optimizer(optimizer)

            cb.on_trial_end(1, 0.5)

            checkpoints = cb.checkpointer.list_checkpoints()
            assert len(checkpoints) >= 1

    def test_load_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = CheckpointCallback(tmpdir, save_every=1, verbose=False)
            optimizer = MockOptimizer()
            optimizer._state = {"value": 42}
            cb.set_optimizer(optimizer)

            cb.on_step_end(1, {"score": 0.9})

            loaded = cb.load_latest()
            assert loaded is not None
            assert loaded["optimizer_state"]["value"] == 42

            best = cb.load_best()
            assert best is not None


class TestProgressCallback:
    def test_initialization(self):
        cb = ProgressCallback(log_every=5, include_scores=True)
        assert cb.log_every == 5
        assert cb.include_scores is True

    def test_always_returns_false(self):
        cb = ProgressCallback()

        assert cb.on_step_end(1, {"score": 0.5}) is False
        assert cb.on_trial_end(1, 0.5) is False
        assert cb.on_generation_end(1, 0.5) is False


class TestHistoryCallback:
    def test_records_step_data(self):
        cb = HistoryCallback()

        cb.on_step_end(1, {"score": 0.5, "best_score": 0.5})
        cb.on_step_end(2, {"score": 0.6, "best_score": 0.6})

        history = cb.get_history()
        assert 1 in history["step"]
        assert 2 in history["step"]
        assert 0.5 in history["score"]
        assert 0.6 in history["score"]

    def test_records_trial_data(self):
        cb = HistoryCallback()

        cb.on_trial_end(1, 0.5, {"best_score": 0.5})
        cb.on_trial_end(2, 0.7, {"best_score": 0.7})

        history = cb.get_history()
        assert 1 in history["trial"]
        assert 2 in history["trial"]
        assert 0.5 in history["score"]
        assert 0.7 in history["score"]

    def test_records_generation_data(self):
        cb = HistoryCallback()

        cb.on_generation_end(1, 0.5, {"avg_score": 0.4})
        cb.on_generation_end(2, 0.7, {"avg_score": 0.6})

        history = cb.get_history()
        assert 1 in history["generation"]
        assert 2 in history["generation"]
        assert 0.5 in history["best_score"]
        assert 0.7 in history["best_score"]

    def test_get_scores(self):
        cb = HistoryCallback()

        cb.on_trial_end(1, 0.5)
        cb.on_trial_end(2, 0.7)
        cb.on_trial_end(3, 0.6)

        scores = cb.get_scores()
        assert len(scores) == 3
        assert 0.5 in scores
        assert 0.7 in scores
        assert 0.6 in scores

    def test_get_best_scores(self):
        cb = HistoryCallback()

        cb.on_trial_end(1, 0.5, {"best_score": 0.5})
        cb.on_trial_end(2, 0.7, {"best_score": 0.7})

        best_scores = cb.get_best_scores()
        assert 0.5 in best_scores
        assert 0.7 in best_scores

    def test_final_best_score(self):
        cb = HistoryCallback()

        cb.on_trial_end(1, 0.5, {"best_score": 0.5})
        cb.on_trial_end(2, 0.9, {"best_score": 0.9})
        cb.on_trial_end(3, 0.6, {"best_score": 0.9})

        assert cb.final_best_score == 0.9

    def test_final_best_score_empty(self):
        cb = HistoryCallback()
        assert cb.final_best_score is None

    def test_clear(self):
        cb = HistoryCallback()

        cb.on_trial_end(1, 0.5)
        cb.on_trial_end(2, 0.7)

        cb.clear()

        assert len(cb.get_scores()) == 0
        assert cb._step_count == 0
        assert cb._trial_count == 0

    def test_always_returns_false(self):
        cb = HistoryCallback()

        assert cb.on_step_end(1) is False
        assert cb.on_trial_end(1, 0.5) is False
        assert cb.on_generation_end(1, 0.5) is False


class TestCallbackListIntegration:
    def test_multiple_callbacks_work_together(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = HistoryCallback()
            checkpoint = CheckpointCallback(tmpdir, save_every=1, verbose=False)
            early_stop = EarlyStoppingCallback(patience=2, verbose=False)

            cl = CallbackList([history, checkpoint, early_stop])
            optimizer = MockOptimizer()
            cl.set_optimizer(optimizer)

            cl.on_optimization_begin()

            # Simulate optimization
            cl.on_trial_end(1, 0.5)
            cl.on_trial_end(2, 0.4)  # No improvement
            result = cl.on_trial_end(3, 0.3)  # Should stop

            cl.on_optimization_end()

            assert result is True  # Early stopping triggered
            assert len(history.get_scores()) == 3
            assert len(checkpoint.checkpointer.list_checkpoints()) >= 1
