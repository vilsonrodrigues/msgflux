"""Tests for msgflux.nn.modules.module module."""

import pytest
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter


class SimpleModule(Module):
    """Simple module for testing."""

    def __init__(self):
        super().__init__()
        self.register_buffer("name", "simple_module")
        self.register_parameter("prompt", Parameter("test prompt", "instruction"))

    def forward(self, x):
        return f"processed: {x}"


class ChildModule(Module):
    """Child module for testing nested modules."""

    def __init__(self):
        super().__init__()
        self.register_buffer("child_name", "child")

    def forward(self, x):
        return f"child: {x}"


class ParentModule(Module):
    """Parent module with children for testing."""

    def __init__(self):
        super().__init__()
        self.register_buffer("parent_name", "parent")
        self.child1 = ChildModule()
        self.child2 = ChildModule()

    def forward(self, x):
        c1 = self.child1(x)
        c2 = self.child2(x)
        return f"parent({c1}, {c2})"


class TestModule:
    """Test suite for Module class."""

    def test_module_initialization(self):
        """Test module initialization."""
        module = Module()
        assert module.training is True
        assert isinstance(module._parameters, dict)
        assert isinstance(module._buffers, dict)
        assert isinstance(module._modules, dict)

    def test_register_buffer(self):
        """Test registering a buffer."""
        module = Module()
        module.register_buffer("test_buffer", "test_value")
        assert module.test_buffer == "test_value"
        assert "test_buffer" in module._buffers

    def test_register_buffer_invalid_name(self):
        """Test registering buffer with invalid name."""
        module = Module()

        with pytest.raises(KeyError, match="buffer name can't be empty string"):
            module.register_buffer("", "value")

        with pytest.raises(KeyError, match="buffer name can't contain '.'"):
            module.register_buffer("buffer.name", "value")

        with pytest.raises(TypeError):
            module.register_buffer(123, "value")

    def test_register_parameter(self):
        """Test registering a parameter."""
        module = Module()
        param = Parameter("test data", "spec")
        module.register_parameter("test_param", param)

        assert module.test_param == param
        assert "test_param" in module._parameters

    def test_register_parameter_invalid_name(self):
        """Test registering parameter with invalid name."""
        module = Module()
        param = Parameter("test", "spec")

        with pytest.raises(KeyError, match="parameter name can't be empty string"):
            module.register_parameter("", param)

        with pytest.raises(KeyError, match="parameter name can't contain '.'"):
            module.register_parameter("param.name", param)

        with pytest.raises(TypeError):
            module.register_parameter(123, param)

    def test_add_module(self):
        """Test adding a child module."""
        parent = Module()
        child = Module()

        parent.add_module("child", child)

        assert parent.child == child
        assert "child" in parent._modules

    def test_add_module_invalid_name(self):
        """Test adding module with invalid name."""
        parent = Module()
        child = Module()

        with pytest.raises(KeyError, match="module name can't be empty string"):
            parent.add_module("", child)

        with pytest.raises(KeyError, match="module name can't contain '.'"):
            parent.add_module("child.module", child)

    def test_get_submodule(self):
        """Test getting a submodule."""
        parent = ParentModule()

        # Get direct child
        child1 = parent.get_submodule("child1")
        assert child1 == parent.child1

        # Get self
        self_module = parent.get_submodule("")
        assert self_module == parent

    def test_get_submodule_invalid(self):
        """Test getting non-existent submodule."""
        module = Module()

        with pytest.raises(AttributeError):
            module.get_submodule("nonexistent")

    def test_state_dict(self):
        """Test state_dict generation."""
        module = SimpleModule()
        state = module.state_dict()

        assert "name" in state
        assert state["name"] == "simple_module"
        assert "prompt" in state
        assert state["prompt"] == "test prompt"

    def test_state_dict_nested(self):
        """Test state_dict with nested modules."""
        parent = ParentModule()
        state = parent.state_dict()

        assert "parent_name" in state
        assert "child1.child_name" in state
        assert "child2.child_name" in state

    def test_load_state_dict(self):
        """Test loading state from dict."""
        module = SimpleModule()
        state = {
            "name": "loaded_name",
            "prompt": "loaded prompt",
        }

        module.load_state_dict(state)

        assert module.name == "loaded_name"
        assert module.prompt.data == "loaded prompt"

    def test_load_state_dict_invalid_type(self):
        """Test load_state_dict with invalid input."""
        module = Module()

        with pytest.raises(TypeError):
            module.load_state_dict("not a dict")

    def test_parameters(self):
        """Test parameters iterator."""
        module = SimpleModule()
        params = list(module.parameters())

        assert len(params) == 1
        assert params[0].data == "test prompt"

    def test_named_parameters(self):
        """Test named_parameters iterator."""
        module = SimpleModule()
        named_params = dict(module.named_parameters())

        assert "prompt" in named_params
        assert named_params["prompt"].data == "test prompt"

    def test_buffers(self):
        """Test buffers iterator."""
        module = SimpleModule()
        buffers = list(module.buffers())

        assert "simple_module" in buffers

    def test_named_buffers(self):
        """Test named_buffers iterator."""
        module = SimpleModule()
        named_buffs = dict(module.named_buffers())

        assert "name" in named_buffs
        assert named_buffs["name"] == "simple_module"

    def test_children(self):
        """Test children iterator."""
        parent = ParentModule()
        children = list(parent.children())

        assert len(children) == 2

    def test_named_children(self):
        """Test named_children iterator."""
        parent = ParentModule()
        named_children = dict(parent.named_children())

        assert "child1" in named_children
        assert "child2" in named_children

    def test_modules(self):
        """Test modules iterator."""
        parent = ParentModule()
        modules = list(parent.modules())

        # Should include parent + 2 children
        assert len(modules) == 3

    def test_named_modules(self):
        """Test named_modules iterator."""
        parent = ParentModule()
        named_mods = list(parent.named_modules())

        # Should have parent (empty prefix) and two children
        assert len(named_mods) == 3
        assert named_mods[0][0] == ""
        assert "child1" in [name for name, _ in named_mods]
        assert "child2" in [name for name, _ in named_mods]

    def test_train_mode(self):
        """Test setting module to training mode."""
        module = SimpleModule()
        module.eval()
        assert module.training is False

        module.train()
        assert module.training is True

    def test_eval_mode(self):
        """Test setting module to evaluation mode."""
        module = SimpleModule()
        assert module.training is True

        module.eval()
        assert module.training is False

    def test_train_propagates_to_children(self):
        """Test that train mode propagates to children."""
        parent = ParentModule()
        parent.eval()

        assert parent.training is False
        assert parent.child1.training is False
        assert parent.child2.training is False

        parent.train()

        assert parent.training is True
        assert parent.child1.training is True
        assert parent.child2.training is True

    def test_call_method(self):
        """Test calling a module."""
        module = SimpleModule()
        result = module("input")
        assert result == "processed: input"

    def test_forward_not_implemented(self):
        """Test that forward raises NotImplementedError if not defined."""
        module = Module()

        with pytest.raises(NotImplementedError):
            module("input")

    def test_get_name(self):
        """Test _get_name returns class name."""
        module = SimpleModule()
        assert module._get_name() == "SimpleModule"

    def test_repr(self):
        """Test module representation."""
        module = SimpleModule()
        repr_str = repr(module)
        assert "SimpleModule" in repr_str

    def test_setattr_with_parameter(self):
        """Test setting parameter via attribute assignment."""
        module = Module()
        param = Parameter("test", "spec")

        module.my_param = param

        assert module.my_param == param
        assert "my_param" in module._parameters

    def test_setattr_with_module(self):
        """Test setting module via attribute assignment."""
        parent = Module()
        child = Module()

        parent.child_module = child

        assert parent.child_module == child
        assert "child_module" in parent._modules

    def test_delattr_parameter(self):
        """Test deleting a parameter."""
        module = Module()
        param = Parameter("test", "spec")
        module.register_parameter("param", param)

        del module.param

        assert "param" not in module._parameters

    def test_delattr_buffer(self):
        """Test deleting a buffer."""
        module = Module()
        module.register_buffer("buff", "value")

        del module.buff

        assert "buff" not in module._buffers

    def test_delattr_module(self):
        """Test deleting a child module."""
        parent = Module()
        child = Module()
        parent.add_module("child", child)

        del parent.child

        assert "child" not in parent._modules

    @pytest.mark.asyncio
    async def test_acall(self):
        """Test async call."""
        module = SimpleModule()
        result = await module.acall("async input")
        assert result == "processed: async input"

    def test_extra_repr(self):
        """Test extra_repr returns empty string by default."""
        module = Module()
        assert module.extra_repr() == ""

    def test_requires_grad_(self):
        """Test requires_grad_ method."""
        module = SimpleModule()

        module.requires_grad_(requires_pgrad=False)

        for param in module.parameters():
            assert param.requires_grad is False

        module.requires_grad_(requires_pgrad=True)

        for param in module.parameters():
            assert param.requires_grad is True
