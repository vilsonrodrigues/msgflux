"""Tests for EarlyStopping."""

import pytest

from msgflux.optim.early_stopping import EarlyStopping, EarlyStoppingState


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self._step_count = 0
        self._state = {"test": "value"}

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state_dict):
        self._state = state_dict.copy()


class TestEarlyStoppingState:
    def test_creation(self):
        state = EarlyStoppingState(
            best_score=0.5,
            best_step=10,
            best_state=None,
            steps_without_improvement=3,
            stopped=False,
            stop_reason=None,
        )
        assert state.best_score == 0.5
        assert state.best_step == 10
        assert state.steps_without_improvement == 3
        assert state.stopped is False


class TestEarlyStoppingInit:
    def test_default_initialization(self):
        es = EarlyStopping()
        assert es.patience == 5
        assert es.min_delta == 0.0
        assert es.mode == "max"
        assert es.restore_best is True
        assert es.verbose is True

    def test_custom_initialization(self):
        es = EarlyStopping(
            patience=10,
            min_delta=0.01,
            mode="min",
            restore_best=False,
            verbose=False,
        )
        assert es.patience == 10
        assert es.min_delta == 0.01
        assert es.mode == "min"
        assert es.restore_best is False
        assert es.verbose is False

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            EarlyStopping(mode="invalid")

    def test_initial_state_max_mode(self):
        es = EarlyStopping(mode="max")
        assert es._state.best_score == float("-inf")
        assert es._state.stopped is False

    def test_initial_state_min_mode(self):
        es = EarlyStopping(mode="min")
        assert es._state.best_score == float("inf")


class TestEarlyStoppingMaxMode:
    def test_improvement_updates_best(self):
        es = EarlyStopping(patience=3, mode="max", verbose=False)
        optimizer = MockOptimizer()

        result = es(0.5, optimizer, step=1)
        assert result is False
        assert es.best_score == 0.5
        assert es.best_step == 1

    def test_no_improvement_increments_counter(self):
        es = EarlyStopping(patience=3, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)  # No improvement

        assert es._state.steps_without_improvement == 1
        assert es.best_score == 0.5

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=3, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)
        es(0.3, optimizer, step=3)
        result = es(0.2, optimizer, step=4)

        assert result is True
        assert es.stopped is True
        assert es.stop_reason is not None
        assert "No improvement for 3 steps" in es.stop_reason

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=3, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)  # No improvement
        es(0.6, optimizer, step=3)  # Improvement!

        assert es._state.steps_without_improvement == 0
        assert es.best_score == 0.6
        assert es.best_step == 3

    def test_min_delta_affects_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.1, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.55, optimizer, step=2)  # Not enough improvement

        assert es._state.steps_without_improvement == 1
        assert es.best_score == 0.5

        es(0.65, optimizer, step=3)  # Enough improvement
        assert es._state.steps_without_improvement == 0
        assert es.best_score == 0.65


class TestEarlyStoppingMinMode:
    def test_lower_is_better(self):
        es = EarlyStopping(patience=3, mode="min", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        assert es.best_score == 0.5

        es(0.3, optimizer, step=2)  # Improvement (lower is better)
        assert es.best_score == 0.3
        assert es._state.steps_without_improvement == 0

        es(0.4, optimizer, step=3)  # No improvement
        assert es._state.steps_without_improvement == 1
        assert es.best_score == 0.3


class TestEarlyStoppingRestoreBest:
    def test_restores_best_state_on_stop(self):
        es = EarlyStopping(patience=2, restore_best=True, verbose=False)
        optimizer = MockOptimizer()

        # Best state at step 1
        optimizer._state = {"step": 1, "value": "best"}
        es(0.9, optimizer, step=1)

        # Worse states
        optimizer._state = {"step": 2, "value": "worse"}
        es(0.5, optimizer, step=2)

        optimizer._state = {"step": 3, "value": "worst"}
        es(0.3, optimizer, step=3)  # This triggers stop

        # Should have restored best state
        assert optimizer._state["step"] == 1
        assert optimizer._state["value"] == "best"

    def test_no_restore_when_disabled(self):
        es = EarlyStopping(patience=2, restore_best=False, verbose=False)
        optimizer = MockOptimizer()

        optimizer._state = {"step": 1}
        es(0.9, optimizer, step=1)

        optimizer._state = {"step": 2}
        es(0.5, optimizer, step=2)

        optimizer._state = {"step": 3}
        es(0.3, optimizer, step=3)

        # State should NOT be restored
        assert optimizer._state["step"] == 3


class TestEarlyStoppingProperties:
    def test_stopped_property(self):
        es = EarlyStopping(patience=1, verbose=False)
        optimizer = MockOptimizer()

        assert es.stopped is False
        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)

        assert es.stopped is True

    def test_best_score_property(self):
        es = EarlyStopping(verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        assert es.best_score == 0.5

        es(0.8, optimizer, step=2)
        assert es.best_score == 0.8

    def test_best_step_property(self):
        es = EarlyStopping(verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        assert es.best_step == 1

        es(0.8, optimizer, step=5)
        assert es.best_step == 5

    def test_stop_reason_property(self):
        es = EarlyStopping(patience=1, verbose=False)
        optimizer = MockOptimizer()

        assert es.stop_reason is None

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)

        assert es.stop_reason is not None
        assert "Best score: 0.5000" in es.stop_reason


class TestEarlyStoppingReset:
    def test_reset_clears_state(self):
        es = EarlyStopping(patience=2, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)
        es(0.3, optimizer, step=3)

        assert es.stopped is True

        es.reset()

        assert es.stopped is False
        assert es.best_score == float("-inf")
        assert es.best_step == 0
        assert es._state.steps_without_improvement == 0

    def test_reset_respects_mode(self):
        es = EarlyStopping(mode="min", verbose=False)
        es.reset()
        assert es.best_score == float("inf")


class TestEarlyStoppingStateDict:
    def test_state_dict_contains_all_fields(self):
        es = EarlyStopping(patience=5, min_delta=0.01, mode="max", verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)

        state = es.state_dict()

        assert "best_score" in state
        assert "best_step" in state
        assert "steps_without_improvement" in state
        assert "stopped" in state
        assert "patience" in state
        assert "min_delta" in state
        assert "mode" in state

        assert state["best_score"] == 0.5
        assert state["best_step"] == 1
        assert state["steps_without_improvement"] == 1

    def test_load_state_dict(self):
        es = EarlyStopping(verbose=False)

        state = {
            "best_score": 0.9,
            "best_step": 10,
            "steps_without_improvement": 2,
            "stopped": False,
            "stop_reason": None,
        }

        es.load_state_dict(state)

        assert es.best_score == 0.9
        assert es.best_step == 10
        assert es._state.steps_without_improvement == 2


class TestEarlyStoppingAlreadyStopped:
    def test_returns_true_if_already_stopped(self):
        es = EarlyStopping(patience=1, verbose=False)
        optimizer = MockOptimizer()

        es(0.5, optimizer, step=1)
        es(0.4, optimizer, step=2)  # Stops here

        # Subsequent calls should return True immediately
        assert es(0.9, optimizer, step=3) is True
        assert es(1.0, optimizer, step=4) is True


class TestEarlyStoppingWithOptimizerStepCount:
    def test_uses_optimizer_step_count_if_step_not_provided(self):
        es = EarlyStopping(verbose=False)
        optimizer = MockOptimizer()
        optimizer._step_count = 5

        es(0.5, optimizer)

        assert es.best_step == 5
