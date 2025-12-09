"""Unit tests for ToolFlowControl class."""

import pytest
from msgflux.tools import ToolFlowControl


class TestToolFlowControl:
    """Test suite for ToolFlowControl base class."""

    def test_toolflowcontrol_instantiation(self):
        """Test that ToolFlowControl can be instantiated."""
        control = ToolFlowControl()
        assert isinstance(control, ToolFlowControl)

    def test_toolflowcontrol_has_docstring(self):
        """Test that ToolFlowControl has appropriate documentation."""
        assert ToolFlowControl.__doc__ is not None
        assert "Base class for creating custom tool flow controls" in ToolFlowControl.__doc__

    def test_toolflowcontrol_inheritance(self):
        """Test that custom classes can inherit from ToolFlowControl."""
        class CustomFlowControl(ToolFlowControl):
            def __init__(self):
                super().__init__()
                self.custom_attr = "custom_value"

        custom = CustomFlowControl()
        assert isinstance(custom, ToolFlowControl)
        assert isinstance(custom, CustomFlowControl)
        assert custom.custom_attr == "custom_value"

    def test_toolflowcontrol_multiple_inheritance(self):
        """Test that ToolFlowControl works with multiple inheritance."""
        class ReActControl(ToolFlowControl):
            """ReAct generation schema control."""
            schema_name = "ReAct"

        class ChainOfThoughtControl(ToolFlowControl):
            """Chain of Thought generation schema control."""
            schema_name = "CoT"

        react = ReActControl()
        cot = ChainOfThoughtControl()

        assert react.schema_name == "ReAct"
        assert cot.schema_name == "CoT"
        assert isinstance(react, ToolFlowControl)
        assert isinstance(cot, ToolFlowControl)

    def test_toolflowcontrol_method_addition(self):
        """Test that custom methods can be added to ToolFlowControl subclasses."""
        class CustomControl(ToolFlowControl):
            def process(self, data):
                return f"Processed: {data}"

        control = CustomControl()
        assert control.process("test") == "Processed: test"

    def test_toolflowcontrol_state_management(self):
        """Test that ToolFlowControl subclasses can manage state."""
        class StatefulControl(ToolFlowControl):
            def __init__(self):
                super().__init__()
                self.state = {}

            def add_state(self, key, value):
                self.state[key] = value

            def get_state(self, key):
                return self.state.get(key)

        control = StatefulControl()
        control.add_state("key1", "value1")
        control.add_state("key2", "value2")

        assert control.get_state("key1") == "value1"
        assert control.get_state("key2") == "value2"
        assert control.get_state("nonexistent") is None

    def test_toolflowcontrol_callable_subclass(self):
        """Test that ToolFlowControl subclasses can be callable."""
        class CallableControl(ToolFlowControl):
            def __call__(self, *args, **kwargs):
                return f"Called with args: {args}, kwargs: {kwargs}"

        control = CallableControl()
        result = control(1, 2, 3, a="x", b="y")
        assert "Called with args: (1, 2, 3)" in result
        assert "kwargs: {'a': 'x', 'b': 'y'}" in result

    def test_toolflowcontrol_with_properties(self):
        """Test that ToolFlowControl subclasses can have properties."""
        class PropertyControl(ToolFlowControl):
            def __init__(self):
                super().__init__()
                self._value = 0

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, new_value):
                self._value = new_value

        control = PropertyControl()
        assert control.value == 0
        control.value = 42
        assert control.value == 42

    def test_toolflowcontrol_with_class_variables(self):
        """Test that ToolFlowControl subclasses can have class variables."""
        class ConfiguredControl(ToolFlowControl):
            default_timeout = 30
            max_retries = 3

        assert ConfiguredControl.default_timeout == 30
        assert ConfiguredControl.max_retries == 3

        control1 = ConfiguredControl()
        control2 = ConfiguredControl()
        assert control1.default_timeout == 30
        assert control2.max_retries == 3

    def test_toolflowcontrol_isolation(self):
        """Test that different ToolFlowControl instances are isolated."""
        class CounterControl(ToolFlowControl):
            def __init__(self):
                super().__init__()
                self.count = 0

            def increment(self):
                self.count += 1

        control1 = CounterControl()
        control2 = CounterControl()

        control1.increment()
        control1.increment()
        control2.increment()

        assert control1.count == 2
        assert control2.count == 1
