"""Tests for msgflux.optim.optimizer module."""

import pytest

from msgflux.nn.parameter import Parameter
from msgflux.optim.optimizer import Optimizer


class ConcreteOptimizer(Optimizer):
    """Concrete implementation for testing."""

    def __init__(self, params, lr=0.01, **kwargs):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad and p.grad:
                    # Simple update: append grad to data
                    p.data = p.data + " [updated]"

        self._step_count += 1
        return loss


class TestOptimizer:
    """Test suite for Optimizer base class."""

    @pytest.fixture
    def params(self):
        """Create sample parameters."""
        return [
            Parameter(data="system prompt", spec="system_message"),
            Parameter(data="instructions", spec="instructions"),
            Parameter(data="examples", spec="examples"),
        ]

    @pytest.fixture
    def optimizer(self, params):
        """Create optimizer instance."""
        return ConcreteOptimizer(params, lr=0.01)

    def test_initialization(self, optimizer, params):
        """Test optimizer initialization."""
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]["params"]) == 3
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer._step_count == 0

    def test_initialization_with_empty_params(self):
        """Test that empty params raises ValueError."""
        with pytest.raises(ValueError, match="empty parameter list"):
            ConcreteOptimizer([])

    def test_initialization_with_single_param(self):
        """Test initialization with single Parameter raises TypeError."""
        param = Parameter(data="test", spec="spec")
        with pytest.raises(TypeError, match="should be an iterable"):
            ConcreteOptimizer(param)

    def test_zero_grad(self, optimizer, params):
        """Test zeroing gradients."""
        # Set some gradients
        for p in params:
            p.grad = "some gradient"

        optimizer.zero_grad()

        for p in params:
            assert p.grad is None

    def test_zero_grad_set_to_empty(self, optimizer, params):
        """Test zeroing gradients with set_to_none=False."""
        for p in params:
            p.grad = "some gradient"

        optimizer.zero_grad(set_to_none=False)

        for p in params:
            assert p.grad == ""

    def test_step(self, optimizer, params):
        """Test optimization step."""
        # Set gradients
        for p in params:
            p.grad = "grad"

        optimizer.step()

        assert optimizer._step_count == 1
        for p in params:
            assert "[updated]" in p.data

    def test_step_with_closure(self, optimizer, params):
        """Test step with closure."""
        closure_called = [False]

        def closure():
            closure_called[0] = True
            return 0.5

        loss = optimizer.step(closure)

        assert closure_called[0] is True
        assert loss == 0.5

    def test_step_count(self, optimizer, params):
        """Test step count increments."""
        assert optimizer.step_count == 0

        optimizer.step()
        assert optimizer.step_count == 1

        optimizer.step()
        assert optimizer.step_count == 2

    def test_state_dict(self, optimizer):
        """Test state dictionary serialization."""
        optimizer.step()

        state = optimizer.state_dict()

        assert "state" in state
        assert "param_groups" in state
        assert "step_count" in state
        assert state["step_count"] == 1

    def test_load_state_dict(self, params):
        """Test loading state dictionary."""
        optimizer1 = ConcreteOptimizer(params, lr=0.01)
        optimizer1.step()
        optimizer1.step()

        state = optimizer1.state_dict()

        # Create new optimizer and load state
        params2 = [
            Parameter(data="new", spec="system_message"),
            Parameter(data="new", spec="instructions"),
            Parameter(data="new", spec="examples"),
        ]
        optimizer2 = ConcreteOptimizer(params2, lr=0.02)

        optimizer2.load_state_dict(state)

        assert optimizer2._step_count == 2

    def test_add_param_group(self, optimizer):
        """Test adding parameter group."""
        new_params = [Parameter(data="extra", spec="extra")]

        optimizer.add_param_group({"params": new_params, "lr": 0.001})

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[1]["lr"] == 0.001

    def test_add_param_group_no_params_key(self, optimizer):
        """Test adding param group without params key raises error."""
        with pytest.raises(ValueError, match="must contain 'params'"):
            optimizer.add_param_group({"lr": 0.001})

    def test_add_param_group_duplicate_param(self, optimizer, params):
        """Test adding duplicate parameter raises error."""
        with pytest.raises(ValueError, match="more than one parameter group"):
            optimizer.add_param_group({"params": [params[0]]})

    def test_repr(self, optimizer):
        """Test string representation."""
        repr_str = repr(optimizer)

        assert "ConcreteOptimizer" in repr_str
        assert "lr" in repr_str

    def test_getstate_setstate(self, optimizer):
        """Test pickling support."""
        optimizer.step()

        state = optimizer.__getstate__()

        new_optimizer = ConcreteOptimizer.__new__(ConcreteOptimizer)
        new_optimizer.__setstate__(state)

        assert new_optimizer._step_count == 1
        assert new_optimizer.defaults == optimizer.defaults

    def test_param_groups_with_dict_input(self):
        """Test initialization with param groups as dicts."""
        params1 = [Parameter(data="p1", spec="s1")]
        params2 = [Parameter(data="p2", spec="s2")]

        optimizer = ConcreteOptimizer(
            [
                {"params": params1, "lr": 0.01},
                {"params": params2, "lr": 0.001},
            ]
        )

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[1]["lr"] == 0.001
