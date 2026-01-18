"""Tests for msgflux.auto.params module."""

import pytest

from msgflux.auto import AutoParams


def test_autoparams_basic_usage():
    """Test basic AutoParams usage with class attributes as defaults."""

    class Model(metaclass=AutoParams):
        def __init__(self, learning_rate, batch_size):
            self.learning_rate = learning_rate
            self.batch_size = batch_size

    class MyModel(Model):
        learning_rate = 0.001
        batch_size = 32

    model = MyModel()
    assert model.learning_rate == 0.001
    assert model.batch_size == 32


def test_autoparams_partial_override():
    """Test AutoParams with partial parameter override."""

    class Model(metaclass=AutoParams):
        def __init__(self, learning_rate, batch_size, epochs):
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs

    class MyModel(Model):
        learning_rate = 0.001
        batch_size = 32
        epochs = 100

    model = MyModel(learning_rate=0.01, batch_size=64)
    assert model.learning_rate == 0.01
    assert model.batch_size == 64
    assert model.epochs == 100


def test_autoparams_stores_auto_params():
    """Test that _auto_params attribute is properly set."""

    class MyClass(metaclass=AutoParams):
        param1 = "value1"
        param2 = 42

        def __init__(self, param1, param2):
            self.param1 = param1
            self.param2 = param2

    assert hasattr(MyClass, "_auto_params")
    assert MyClass._auto_params["param1"] == "value1"
    assert MyClass._auto_params["param2"] == 42
