"""Tests for msgflux.nn.parameter module."""

import pytest
from msgflux.nn.parameter import Parameter


class TestParameter:
    """Test suite for Parameter class."""

    def test_parameter_initialization(self):
        """Test parameter initialization."""
        param = Parameter(data="test prompt", spec="instruction")
        assert param.data == "test prompt"
        assert param.spec == "instruction"
        assert param.requires_grad is True

    def test_parameter_without_requires_grad(self):
        """Test parameter with requires_grad=False."""
        param = Parameter(data="test", spec="spec", requires_grad=False)
        assert param.requires_grad is False

    def test_requires_grad_setter(self):
        """Test setting requires_grad."""
        param = Parameter(data="test", spec="spec")
        assert param.requires_grad is True

        param.requires_grad_(requires_grad=False)
        assert param.requires_grad is False

        param.requires_grad_(requires_grad=True)
        assert param.requires_grad is True

    def test_hash(self):
        """Test parameter hashing."""
        param1 = Parameter(data="test", spec="spec1")
        param2 = Parameter(data="test", spec="spec1")
        param3 = Parameter(data="test", spec="spec2")

        assert hash(param1) == hash(param2)
        assert hash(param1) != hash(param3)

    def test_equality(self):
        """Test parameter equality."""
        param1 = Parameter(data="test", spec="spec1")
        param2 = Parameter(data="test", spec="spec1")
        param3 = Parameter(data="test", spec="spec2")
        param4 = Parameter(data="different", spec="spec1")

        assert param1 == param2
        assert param1 != param3
        assert param1 != param4
        assert param1 != "test"

    def test_str_representation(self):
        """Test string representation."""
        param = Parameter(data="Hello World", spec="greeting")
        assert str(param) == "Hello World"

    def test_repr(self):
        """Test repr."""
        param = Parameter(data="Hello World", spec="greeting")
        assert repr(param) == "Hello World"

    def test_add_with_string(self):
        """Test adding parameter with string."""
        param = Parameter(data="Hello", spec="greeting")
        result = param + " World"
        assert result == "Hello World"

    def test_add_with_parameter(self):
        """Test adding parameter with another parameter."""
        param1 = Parameter(data="Hello", spec="greeting")
        param2 = Parameter(data=" World", spec="continuation")
        result = param1 + param2
        assert result == "Hello World"

    def test_add_with_invalid_type(self):
        """Test adding parameter with invalid type raises TypeError."""
        param = Parameter(data="test", spec="spec")
        with pytest.raises(TypeError):
            result = param + 123

    def test_radd_with_string(self):
        """Test right addition with string."""
        param = Parameter(data="World", spec="word")
        result = "Hello " + param
        assert result == "Hello World"

    def test_radd_with_invalid_type(self):
        """Test right addition with invalid type raises TypeError."""
        param = Parameter(data="test", spec="spec")
        with pytest.raises(TypeError):
            result = 123 + param

    def test_copy_to_data(self):
        """Test copying data to parameter."""
        param = Parameter(data="original", spec="spec")
        param.copy_to_data("new data")
        assert param.data == "new data"

    def test_copy_to_data_deep_copy(self):
        """Test that copy_to_data creates a deep copy."""
        original_data = {"key": "value"}
        param = Parameter(data="test", spec="spec")
        param.copy_to_data(original_data)

        # Modify original
        original_data["key"] = "modified"

        # Parameter data should not be affected
        assert param.data == {"key": "value"}

    def test_clone(self):
        """Test cloning a parameter."""
        param = Parameter(data="test data", spec="spec")
        param.grad = "gradient"

        cloned = param.clone()

        assert cloned.data == param.data
        assert cloned.spec == param.spec
        assert cloned.grad == param.grad
        assert cloned is not param

    def test_copy_(self):
        """Test copying from another parameter."""
        param1 = Parameter(data="original", spec="spec1")
        param2 = Parameter(data="new data", spec="spec2")

        param1.copy_(param2)

        assert param1.data == "new data"
        # spec should not change
        assert param1.spec == "spec1"

    def test_copy_with_none(self):
        """Test copy_ with None source."""
        param = Parameter(data="original", spec="spec")
        original_data = param.data

        param.copy_(None)

        # Data should remain unchanged when src is None
        assert param.data == original_data

    def test_grad_attribute(self):
        """Test grad attribute."""
        param = Parameter(data="test", spec="spec")
        assert param.grad is None

        param.grad = "computed gradient"
        assert param.grad == "computed gradient"
