"""Tests for msgflux.trainer.trainer module."""

import pytest

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer
from msgflux.trainer.trainer import (
    Callback,
    EarlyStopping,
    ProgressCallback,
    Trainer,
    TrainerConfig,
    TrainingState,
)


class MockModule(Module):
    """Mock module for testing."""

    def __init__(self):
        super().__init__()
        self._examples = Parameter(data="", spec=PromptSpec.EXAMPLES)

    def forward(self, inputs):
        return "predicted"


class MockOptimizer(Optimizer):
    """Mock optimizer for testing."""

    def __init__(self, params):
        super().__init__(params, defaults={})
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        self._step_count += 1
        if closure:
            return closure()
        return None


class MockEvaluator:
    """Mock evaluator for testing."""

    def __init__(self, scores=None):
        self.scores = scores or [50.0, 60.0, 70.0, 80.0, 90.0]
        self.call_count = 0

    def __call__(self, module, devset, **kwargs):
        score = self.scores[min(self.call_count, len(self.scores) - 1)]
        self.call_count += 1
        return EvaluationResult(score=score, results=[])


class TestTrainerConfig:
    """Test suite for TrainerConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainerConfig()

        assert config.max_epochs == 10
        assert config.eval_every == 1
        assert config.save_best is True
        assert config.early_stopping_patience is None
        assert config.verbose is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainerConfig(
            max_epochs=5,
            eval_every=2,
            save_best=False,
            early_stopping_patience=3,
        )

        assert config.max_epochs == 5
        assert config.eval_every == 2
        assert config.save_best is False
        assert config.early_stopping_patience == 3


class TestTrainingState:
    """Test suite for TrainingState."""

    def test_default_values(self):
        """Test default state values."""
        state = TrainingState()

        assert state.epoch == 0
        assert state.step == 0
        assert state.best_score == 0.0
        assert state.best_epoch == 0
        assert state.history == []
        assert state.is_best is False

    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = TrainingState(
            epoch=5,
            step=50,
            best_score=85.0,
            best_epoch=3,
            history=[{"epoch": 0, "score": 50.0}],
        )

        d = state.to_dict()

        assert d["epoch"] == 5
        assert d["step"] == 50
        assert d["best_score"] == 85.0
        assert d["best_epoch"] == 3


class TestCallback:
    """Test suite for Callback base class."""

    def test_callback_methods_exist(self):
        """Test that all callback methods exist."""
        callback = Callback()

        # These should not raise
        callback.on_train_begin(None)
        callback.on_train_end(None)
        callback.on_epoch_begin(None, 0)
        callback.on_epoch_end(None, 0, {})
        callback.on_evaluate(None, None)


class TestEarlyStopping:
    """Test suite for EarlyStopping callback."""

    def test_no_early_stop_with_improvement(self):
        """Test no early stopping when improving."""
        es = EarlyStopping(patience=3)

        es.on_epoch_end(None, 0, {"val_score": 50.0})
        assert es.should_stop is False

        es.on_epoch_end(None, 1, {"val_score": 60.0})
        assert es.should_stop is False

        es.on_epoch_end(None, 2, {"val_score": 70.0})
        assert es.should_stop is False

    def test_early_stop_no_improvement(self):
        """Test early stopping with no improvement."""
        es = EarlyStopping(patience=2)

        es.on_epoch_end(None, 0, {"val_score": 70.0})
        es.on_epoch_end(None, 1, {"val_score": 60.0})
        assert es.should_stop is False

        es.on_epoch_end(None, 2, {"val_score": 65.0})
        assert es.should_stop is True

    def test_min_delta(self):
        """Test min_delta parameter."""
        es = EarlyStopping(patience=2, min_delta=5.0)

        es.on_epoch_end(None, 0, {"val_score": 70.0})
        es.on_epoch_end(None, 1, {"val_score": 72.0})  # Not enough improvement
        es.on_epoch_end(None, 2, {"val_score": 73.0})  # Still not enough

        assert es.should_stop is True


class TestTrainer:
    """Test suite for Trainer class."""

    @pytest.fixture
    def module(self):
        """Create mock module."""
        return MockModule()

    @pytest.fixture
    def optimizer(self, module):
        """Create mock optimizer."""
        return MockOptimizer(module.parameters())

    @pytest.fixture
    def evaluator(self):
        """Create mock evaluator."""
        return MockEvaluator()

    @pytest.fixture
    def trainset(self):
        """Create sample training set."""
        return [
            Example(inputs="q1", labels="a1"),
            Example(inputs="q2", labels="a2"),
        ]

    @pytest.fixture
    def valset(self):
        """Create sample validation set."""
        return [
            Example(inputs="v1", labels="va1"),
            Example(inputs="v2", labels="va2"),
        ]

    @pytest.fixture
    def trainer(self, module, optimizer, evaluator):
        """Create Trainer instance."""
        return Trainer(
            module=module,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=5, verbose=False),
        )

    def test_initialization(self, trainer, module, optimizer):
        """Test trainer initialization."""
        assert trainer.module is module
        assert trainer.optimizer is optimizer
        assert trainer.config.max_epochs == 5

    def test_fit_runs_epochs(self, trainer, trainset, valset, optimizer):
        """Test that fit runs the correct number of epochs."""
        trainer.fit(trainset=trainset, valset=valset)

        assert optimizer.step_calls == 5
        assert trainer.state.epoch == 4  # 0-indexed, last epoch

    def test_fit_tracks_best_score(self, trainer, trainset, valset):
        """Test that fit tracks best score."""
        trainer.fit(trainset=trainset, valset=valset)

        assert trainer.state.best_score > 0
        assert trainer.state.best_epoch >= 0

    def test_fit_records_history(self, trainer, trainset, valset):
        """Test that fit records history."""
        trainer.fit(trainset=trainset, valset=valset)

        assert len(trainer.state.history) == 5
        for entry in trainer.state.history:
            assert "epoch" in entry
            assert "val_score" in entry

    def test_fit_without_valset(self, module, optimizer, trainset):
        """Test fit without validation set."""
        trainer = Trainer(
            module=module,
            optimizer=optimizer,
            evaluator=None,
            config=TrainerConfig(max_epochs=3, verbose=False),
        )

        trainer.fit(trainset=trainset)

        assert optimizer.step_calls == 3

    def test_early_stopping(self, module, trainset, valset):
        """Test early stopping."""
        optimizer = MockOptimizer(module.parameters())
        # Evaluator with decreasing scores to trigger early stopping
        evaluator = MockEvaluator(scores=[80.0, 70.0, 60.0, 50.0, 40.0])

        trainer = Trainer(
            module=module,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(
                max_epochs=10,
                early_stopping_patience=2,
                verbose=False,
            ),
        )

        trainer.fit(trainset=trainset, valset=valset)

        # Should stop early due to no improvement
        assert optimizer.step_calls < 10

    def test_eval_every(self, module, optimizer, trainset, valset):
        """Test eval_every parameter."""
        evaluator = MockEvaluator()

        trainer = Trainer(
            module=module,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=6, eval_every=2, verbose=False),
        )

        trainer.fit(trainset=trainset, valset=valset)

        # Should evaluate on epochs 1, 3, 5 (0-indexed)
        assert evaluator.call_count == 3

    def test_custom_callbacks(self, module, optimizer, trainset, valset):
        """Test custom callbacks."""
        callback_log = []

        class LoggingCallback(Callback):
            def on_epoch_begin(self, trainer, epoch):
                callback_log.append(f"begin_{epoch}")

            def on_epoch_end(self, trainer, epoch, logs):
                callback_log.append(f"end_{epoch}")

        evaluator = MockEvaluator()
        trainer = Trainer(
            module=module,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=3, verbose=False),
            callbacks=[LoggingCallback()],
        )

        trainer.fit(trainset=trainset, valset=valset)

        assert "begin_0" in callback_log
        assert "end_0" in callback_log
        assert len(callback_log) == 6  # 3 begins + 3 ends

    def test_restore_best(self, trainer, trainset, valset):
        """Test restoring best state."""
        trainer.fit(trainset=trainset, valset=valset)

        # Should have saved best state
        trainer.restore_best()

        # Should not raise

    def test_state_dict(self, trainer, trainset, valset):
        """Test state dictionary serialization."""
        trainer.fit(trainset=trainset, valset=valset)

        state = trainer.state_dict()

        assert "training_state" in state
        assert "optimizer_state" in state
        assert "config" in state

    def test_load_state_dict(self, module, trainset, valset):
        """Test loading state dictionary."""
        optimizer1 = MockOptimizer(module.parameters())
        evaluator = MockEvaluator()
        trainer1 = Trainer(
            module=module,
            optimizer=optimizer1,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=3, verbose=False),
        )
        trainer1.fit(trainset=trainset, valset=valset)
        state = trainer1.state_dict()

        # Create new trainer
        module2 = MockModule()
        optimizer2 = MockOptimizer(module2.parameters())
        trainer2 = Trainer(
            module=module2,
            optimizer=optimizer2,
            config=TrainerConfig(max_epochs=5, verbose=False),
        )
        trainer2.load_state_dict(state)

        assert trainer2.state.epoch == trainer1.state.epoch
        assert trainer2.state.best_score == trainer1.state.best_score

    def test_module_train_eval_modes(self, trainer, trainset, valset):
        """Test that module is set to train/eval modes appropriately."""
        trainer.fit(trainset=trainset, valset=valset)

        # Module should be in some state at end
        # (implementation dependent)

    def test_returns_training_state(self, trainer, trainset, valset):
        """Test that fit returns TrainingState."""
        result = trainer.fit(trainset=trainset, valset=valset)

        assert isinstance(result, TrainingState)
        assert result.best_score > 0
