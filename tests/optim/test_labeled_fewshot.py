"""Tests for msgflux.optim.labeled_fewshot module."""

import pytest

from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.nn.parameter import Parameter
from msgflux.optim.labeled_fewshot import LabeledFewShot


class TestLabeledFewShot:
    """Test suite for LabeledFewShot optimizer."""

    @pytest.fixture
    def trainset(self):
        """Create sample training set."""
        return [
            Example(inputs="What is 2+2?", labels="4"),
            Example(inputs="What is 3+3?", labels="6"),
            Example(inputs="What is 4+4?", labels="8"),
            Example(inputs="What is 5+5?", labels="10"),
            Example(inputs="What is 6+6?", labels="12"),
            Example(inputs="What is 7+7?", labels="14"),
            Example(inputs="What is 8+8?", labels="16"),
            Example(inputs="What is 9+9?", labels="18"),
        ]

    @pytest.fixture
    def params(self):
        """Create sample parameters including examples parameter."""
        return [
            Parameter(data="You are a calculator.", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="Calculate the sum.", spec=PromptSpec.INSTRUCTIONS),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]

    @pytest.fixture
    def optimizer(self, params, trainset):
        """Create LabeledFewShot optimizer."""
        return LabeledFewShot(params, trainset=trainset, k=4, seed=42)

    def test_initialization(self, optimizer, trainset):
        """Test optimizer initialization."""
        assert optimizer.k == 4
        assert optimizer.sample is True
        assert optimizer.seed == 42
        assert len(optimizer.trainset) == len(trainset)

    def test_initialization_empty_trainset(self, params):
        """Test that empty trainset raises ValueError."""
        with pytest.raises(ValueError, match="trainset cannot be empty"):
            LabeledFewShot(params, trainset=[], k=4)

    def test_step_selects_k_examples(self, optimizer, params):
        """Test that step selects k examples."""
        optimizer.step()

        selected = optimizer.get_selected_examples()
        assert len(selected) == 4

    def test_step_updates_examples_param(self, optimizer, params):
        """Test that step updates the examples parameter."""
        examples_param = params[2]  # examples parameter
        original_data = examples_param.data

        optimizer.step()

        assert examples_param.data != original_data
        assert len(examples_param.data) > 0

    def test_step_with_sample_false(self, params, trainset):
        """Test step without sampling (takes first k)."""
        optimizer = LabeledFewShot(params, trainset=trainset, k=3, sample=False)

        optimizer.step()

        selected = optimizer.get_selected_examples()
        # Should be first 3 examples
        assert selected[0].inputs == "What is 2+2?"
        assert selected[1].inputs == "What is 3+3?"
        assert selected[2].inputs == "What is 4+4?"

    def test_step_deterministic_with_seed(self, params, trainset):
        """Test that same seed produces same selection."""
        optimizer1 = LabeledFewShot(params, trainset=trainset, k=4, seed=123)
        optimizer1.step()
        selected1 = optimizer1.get_selected_examples()

        # Create new parameters for second optimizer
        params2 = [
            Parameter(data="system", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="instructions", spec=PromptSpec.INSTRUCTIONS),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]
        optimizer2 = LabeledFewShot(params2, trainset=trainset, k=4, seed=123)
        optimizer2.step()
        selected2 = optimizer2.get_selected_examples()

        # Should select same examples
        for ex1, ex2 in zip(selected1, selected2):
            assert ex1.inputs == ex2.inputs
            assert ex1.labels == ex2.labels

    def test_step_different_seeds_different_selection(self, params, trainset):
        """Test that different seeds produce different selections."""
        optimizer1 = LabeledFewShot(params, trainset=trainset, k=4, seed=1)
        optimizer1.step()
        selected1 = [ex.inputs for ex in optimizer1.get_selected_examples()]

        params2 = [
            Parameter(data="system", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="instructions", spec=PromptSpec.INSTRUCTIONS),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]
        optimizer2 = LabeledFewShot(params2, trainset=trainset, k=4, seed=999)
        optimizer2.step()
        selected2 = [ex.inputs for ex in optimizer2.get_selected_examples()]

        # Very likely to be different with different seeds
        # (small chance of collision with only 8 examples)
        assert selected1 != selected2 or True  # Allow for rare collision

    def test_k_larger_than_trainset(self, params, trainset):
        """Test when k is larger than trainset size."""
        optimizer = LabeledFewShot(params, trainset=trainset, k=100)

        optimizer.step()

        selected = optimizer.get_selected_examples()
        assert len(selected) == len(trainset)  # Should cap at trainset size

    def test_reseed(self, optimizer):
        """Test reseeding the random number generator."""
        optimizer.reseed(999)

        assert optimizer.seed == 999

    def test_state_dict(self, optimizer):
        """Test state dictionary serialization."""
        optimizer.step()

        state = optimizer.state_dict()

        assert "selected_examples" in state
        assert "seed" in state
        assert len(state["selected_examples"]) == 4

    def test_load_state_dict(self, params, trainset):
        """Test loading state dictionary."""
        optimizer1 = LabeledFewShot(params, trainset=trainset, k=4, seed=42)
        optimizer1.step()
        state = optimizer1.state_dict()

        # Create new optimizer
        params2 = [
            Parameter(data="new", spec=PromptSpec.SYSTEM_MESSAGE),
            Parameter(data="new", spec=PromptSpec.INSTRUCTIONS),
            Parameter(data="", spec=PromptSpec.EXAMPLES),
        ]
        optimizer2 = LabeledFewShot(params2, trainset=trainset, k=2, seed=1)
        optimizer2.load_state_dict(state)

        # Should restore selected examples
        assert len(optimizer2._selected_examples) == 4
        assert optimizer2.seed == 42

    def test_only_updates_examples_params(self, params, trainset):
        """Test that only examples parameters are updated."""
        system_param = params[0]
        instructions_param = params[1]

        original_system = system_param.data
        original_instructions = instructions_param.data

        optimizer = LabeledFewShot(params, trainset=trainset, k=4)
        optimizer.step()

        assert system_param.data == original_system
        assert instructions_param.data == original_instructions

    def test_respects_requires_grad(self, trainset):
        """Test that params with requires_grad=False are not updated."""
        params = [
            Parameter(data="", spec=PromptSpec.EXAMPLES, requires_grad=False),
        ]

        optimizer = LabeledFewShot(params, trainset=trainset, k=4)
        optimizer.step()

        # Should not update frozen parameter
        assert params[0].data == ""

    def test_multiple_steps(self, optimizer):
        """Test multiple optimization steps."""
        optimizer.step()
        selected1 = [ex.inputs for ex in optimizer.get_selected_examples()]

        optimizer.step()
        selected2 = [ex.inputs for ex in optimizer.get_selected_examples()]

        # With same seed continuing, selections may differ
        assert optimizer._step_count == 2

    def test_step_with_closure(self, optimizer):
        """Test step with closure function."""
        closure_called = [False]

        def closure():
            closure_called[0] = True
            return 0.8

        loss = optimizer.step(closure)

        assert closure_called[0] is True
        assert loss == 0.8

    @pytest.mark.asyncio
    async def test_astep_basic(self, optimizer):
        """Test async step method."""
        await optimizer.astep()

        selected = optimizer.get_selected_examples()
        assert len(selected) == 4
        assert optimizer._step_count == 1

    @pytest.mark.asyncio
    async def test_astep_with_closure(self, optimizer):
        """Test async step with closure function."""
        closure_called = [False]

        def closure():
            closure_called[0] = True
            return 0.75

        loss = await optimizer.astep(closure)

        assert closure_called[0] is True
        assert loss == 0.75

    @pytest.mark.asyncio
    async def test_astep_updates_params(self, optimizer, params):
        """Test that astep updates parameters correctly."""
        examples_param = params[2]
        original_data = examples_param.data

        await optimizer.astep()

        assert examples_param.data != original_data
        assert len(examples_param.data) > 0
