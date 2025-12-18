"""Tests for msgflux.nn.modules.module base class."""

import pytest
from unittest.mock import Mock

from msgflux.nn.modules.module import Module


class SimpleModule(Module):
    """Simple test module."""
    
    def __init__(self, value=10):
        super().__init__()
        self.value = value
    
    def forward(self, x):
        return x + self.value


class TestModule:
    """Test suite for Module base class."""

    def test_module_initialization(self):
        """Test Module basic initialization."""
        module = SimpleModule()
        assert module.value == 10

    def test_module_call_invokes_forward(self):
        """Test that calling module invokes forward method."""
        module = SimpleModule(value=5)
        result = module(10)
        assert result == 15

    def test_module_set_name(self):
        """Test Module set_name method."""
        module = SimpleModule()
        module.set_name("test_module")
        assert module.name == "test_module"

    def test_module_set_description(self):
        """Test Module set_description method."""
        module = SimpleModule()
        module.set_description("A test module")
        assert module.description == "A test module"

    def test_module_has_parameters_dict(self):
        """Test that Module has parameters dict."""
        module = SimpleModule()
        assert hasattr(module, "_parameters")
        assert isinstance(module._parameters, dict)

    def test_module_has_buffers_dict(self):
        """Test that Module has buffers dict."""
        module = SimpleModule()
        assert hasattr(module, "_buffers")
        assert isinstance(module._buffers, dict)

    def test_module_has_modules_dict(self):
        """Test that Module has modules dict."""
        module = SimpleModule()
        assert hasattr(module, "_modules")
        assert isinstance(module._modules, dict)
