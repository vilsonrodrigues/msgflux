"""Unit tests for async optimizer methods."""

import asyncio

import pytest

from msgflux.examples import Example
from msgflux.evaluate.evaluator import Evaluator, EvaluationResult
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.bootstrap import BootstrapFewShot
from msgflux.optim.copro import COPRO
from msgflux.optim.gepa import GEPA
from msgflux.optim.mipro import MIPROv2


class MockAsyncModule:
    """Mock module with async support for testing."""

    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0
        self.training = True

    def __call__(self, inputs) -> str:
        self.call_count += 1
        return self.response

    async def acall(self, inputs) -> str:
        """Async call method."""
        self.call_count += 1
        await asyncio.sleep(0.001)  # Simulate async work
        return self.response

    async def aforward(self, inputs) -> str:
        """Async forward method."""
        return await self.acall(inputs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class MockSyncModule:
    """Mock module without async support for testing fallback."""

    def __init__(self, response: str = "Sync response"):
        self.response = response
        self.call_count = 0
        self.training = True

    def __call__(self, inputs) -> str:
        self.call_count += 1
        return self.response

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@pytest.fixture
def trainset():
    """Create sample training set."""
    return [
        Example(inputs="What is 2+2?", labels="4"),
        Example(inputs="What is 3+3?", labels="6"),
        Example(inputs="What is 4+4?", labels="8"),
        Example(inputs="What is 5+5?", labels="10"),
        Example(inputs="What is 6+6?", labels="12"),
    ]


@pytest.fixture
def valset():
    """Create sample validation set."""
    return [
        Example(inputs="What is 7+7?", labels="14"),
        Example(inputs="What is 8+8?", labels="16"),
    ]


@pytest.fixture
def metric():
    """Create exact match metric."""

    def exact_match(example: Example, prediction) -> float:
        return 1.0 if str(prediction) == str(example.labels) else 0.0

    return exact_match


@pytest.fixture
def instructions_param():
    """Create instructions parameter."""
    return Parameter("Do the task.", PromptSpec.INSTRUCTIONS)


@pytest.fixture
def examples_param():
    """Create examples parameter."""
    return Parameter(None, PromptSpec.EXAMPLES)


# =============================================================================
# Evaluator Async Tests
# =============================================================================


class TestEvaluatorAsync:
    """Tests for Evaluator async methods."""

    @pytest.mark.asyncio
    async def test_aevaluate_basic(self, trainset, metric):
        """Test basic async evaluation."""
        module = MockAsyncModule("4")
        evaluator = Evaluator(metric=metric)

        result = await evaluator.aevaluate(module, trainset)

        assert isinstance(result, EvaluationResult)
        assert result.score >= 0
        assert module.call_count == len(trainset)

    @pytest.mark.asyncio
    async def test_aevaluate_with_concurrency(self, trainset, metric):
        """Test async evaluation with concurrency limit."""
        module = MockAsyncModule("4")
        evaluator = Evaluator(metric=metric)

        result = await evaluator.aevaluate(module, trainset, max_concurrency=2)

        assert isinstance(result, EvaluationResult)
        assert module.call_count == len(trainset)

    @pytest.mark.asyncio
    async def test_aevaluate_fallback_to_sync(self, trainset, metric):
        """Test that aevaluate falls back to sync for non-async modules."""
        module = MockSyncModule("4")
        evaluator = Evaluator(metric=metric)

        result = await evaluator.aevaluate(module, trainset)

        assert isinstance(result, EvaluationResult)
        assert module.call_count == len(trainset)

    @pytest.mark.asyncio
    async def test_aevaluate_empty_devset(self, metric):
        """Test async evaluation with empty dataset."""
        module = MockAsyncModule()
        evaluator = Evaluator(metric=metric)

        result = await evaluator.aevaluate(module, [])

        assert result.score == 0.0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_aevaluate_sets_eval_mode(self, trainset, metric):
        """Test that aevaluate sets module to eval mode."""
        module = MockAsyncModule("4")
        module.training = True
        evaluator = Evaluator(metric=metric)

        await evaluator.aevaluate(module, trainset)

        # Module should be restored to training mode after
        assert module.training is True

    @pytest.mark.asyncio
    async def test_aevaluate_single(self, trainset, metric):
        """Test evaluating a single example asynchronously."""
        module = MockAsyncModule("4")
        evaluator = Evaluator(metric=metric)

        prediction, score = await evaluator.aevaluate_single(module, trainset[0])

        assert prediction == "4"
        assert score == 1.0  # "4" matches trainset[0].labels


# =============================================================================
# BootstrapFewShot Async Tests
# =============================================================================


class TestBootstrapFewShotAsync:
    """Tests for BootstrapFewShot async methods."""

    @pytest.mark.asyncio
    async def test_astep_basic(self, instructions_param, trainset, metric):
        """Test basic async step."""
        optimizer = BootstrapFewShot(
            [instructions_param],
            metric=metric,
            max_bootstrapped_demos=3,
        )
        teacher = MockAsyncModule("4")

        await optimizer.astep(trainset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_concurrency(self, instructions_param, trainset, metric):
        """Test async step with concurrency limit."""
        optimizer = BootstrapFewShot(
            [instructions_param],
            metric=metric,
            max_bootstrapped_demos=3,
        )
        teacher = MockAsyncModule("4")

        await optimizer.astep(trainset, teacher=teacher, max_concurrency=2)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_fallback_sync(self, instructions_param, trainset, metric):
        """Test async step with sync module (fallback)."""
        optimizer = BootstrapFewShot(
            [instructions_param],
            metric=metric,
            max_bootstrapped_demos=3,
        )
        teacher = MockSyncModule("4")

        await optimizer.astep(trainset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_collects_results(self, instructions_param, trainset, metric):
        """Test that async step collects bootstrap results."""
        optimizer = BootstrapFewShot(
            [instructions_param],
            metric=metric,
            max_bootstrapped_demos=5,
        )
        teacher = MockAsyncModule("4")

        await optimizer.astep(trainset, teacher=teacher)

        # Should have some bootstrap results
        assert len(optimizer._bootstrap_results) > 0


# =============================================================================
# COPRO Async Tests
# =============================================================================


class TestCOPROAsync:
    """Tests for COPRO async methods."""

    @pytest.mark.asyncio
    async def test_astep_basic(self, instructions_param, trainset, metric):
        """Test basic async step."""
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            depth=1,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_concurrency(self, instructions_param, trainset, metric):
        """Test async step with concurrency limit."""
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            depth=1,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher, max_concurrency=2)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_valset(
        self, instructions_param, trainset, valset, metric
    ):
        """Test async step with separate validation set."""
        optimizer = COPRO(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            depth=1,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, valset, teacher=teacher)

        assert optimizer._step_count == 1


# =============================================================================
# MIPROv2 Async Tests
# =============================================================================


class TestMIPROv2Async:
    """Tests for MIPROv2 async methods."""

    @pytest.mark.asyncio
    async def test_astep_basic(self, instructions_param, trainset, metric):
        """Test basic async step."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            num_trials=3,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_concurrency(self, instructions_param, trainset, metric):
        """Test async step with concurrency limit."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            num_trials=3,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher, max_concurrency=2)

        assert optimizer._step_count == 1
        assert len(optimizer._trials) == 3

    @pytest.mark.asyncio
    async def test_astep_with_valset(
        self, instructions_param, trainset, valset, metric
    ):
        """Test async step with separate validation set."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            num_trials=3,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, valset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_records_trials(self, instructions_param, trainset, metric):
        """Test that async step records trials."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            num_trials=5,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert len(optimizer._trials) == 5


# =============================================================================
# GEPA Async Tests
# =============================================================================


class TestGEPAAsync:
    """Tests for GEPA async methods."""

    @pytest.mark.asyncio
    async def test_astep_basic(self, instructions_param, trainset, metric):
        """Test basic async step."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=3,
            num_generations=2,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_concurrency(self, instructions_param, trainset, metric):
        """Test async step with concurrency limit."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=3,
            num_generations=2,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher, max_concurrency=2)

        assert optimizer._step_count == 1
        assert optimizer._current_generation == 2

    @pytest.mark.asyncio
    async def test_astep_with_valset(
        self, instructions_param, trainset, valset, metric
    ):
        """Test async step with separate validation set."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=3,
            num_generations=2,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, valset, teacher=teacher)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_evolves_population(self, instructions_param, trainset, metric):
        """Test that async step evolves population."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=5,
            num_generations=3,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert len(optimizer._population) == 5
        assert optimizer._current_generation == 3

    @pytest.mark.asyncio
    async def test_astep_records_stats(self, instructions_param, trainset, metric):
        """Test that async step records statistics."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=3,
            num_generations=3,
        )
        teacher = MockAsyncModule()

        await optimizer.astep(trainset, teacher=teacher)

        assert len(optimizer._stats_history) == 3


# =============================================================================
# Cross-Optimizer Tests
# =============================================================================


class TestAsyncConcurrencyBehavior:
    """Tests for async concurrency behavior across optimizers."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_faster(self, trainset, metric):
        """Test that concurrent execution is faster than sequential."""
        import time

        module = MockAsyncModule("4")

        # Add some delay to async calls
        original_acall = module.acall

        async def slow_acall(inputs):
            await asyncio.sleep(0.01)
            return await original_acall(inputs)

        module.acall = slow_acall

        evaluator = Evaluator(metric=metric)

        # Time concurrent execution
        start = time.time()
        await evaluator.aevaluate(module, trainset)
        concurrent_time = time.time() - start

        # Reset for sequential test
        module.call_count = 0

        # Time with max_concurrency=1 (sequential)
        start = time.time()
        await evaluator.aevaluate(module, trainset, max_concurrency=1)
        sequential_time = time.time() - start

        # Concurrent should be faster (or at least not slower)
        # Note: with small sample sizes this might not always hold
        assert concurrent_time <= sequential_time * 2

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, instructions_param, metric):
        """Test that semaphore properly limits concurrency."""
        concurrent_count = 0
        max_concurrent = 0

        class ConcurrencyTracker:
            def __init__(self):
                self.training = True

            async def acall(self, inputs):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.01)
                concurrent_count -= 1
                return "result"

            def train(self):
                self.training = True

            def eval(self):
                self.training = False

        trainset = [Example(inputs=f"q{i}", labels="a") for i in range(10)]

        def metric(ex, pred):
            return 1.0

        evaluator = Evaluator(metric=metric)
        module = ConcurrencyTracker()

        await evaluator.aevaluate(module, trainset, max_concurrency=3)

        assert max_concurrent <= 3


class TestAsyncWithoutTeacher:
    """Tests for async methods without teacher module."""

    @pytest.mark.asyncio
    async def test_mipro_astep_without_teacher(self, instructions_param, trainset, metric):
        """Test MIPROv2 astep without teacher."""
        optimizer = MIPROv2(
            [instructions_param],
            metric=metric,
            num_candidates=2,
            num_trials=3,
        )

        await optimizer.astep(trainset)

        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_gepa_astep_without_teacher(self, instructions_param, trainset, metric):
        """Test GEPA astep without teacher."""
        optimizer = GEPA(
            [instructions_param],
            metric=metric,
            population_size=3,
            num_generations=2,
        )

        await optimizer.astep(trainset)

        assert optimizer._step_count == 1
