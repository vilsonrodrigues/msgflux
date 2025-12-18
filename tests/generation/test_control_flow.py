"""Tests for control flow functionality."""

import pytest
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl


class TestToolFlowControl:
    """Tests for ToolFlowControl base class."""

    def test_tool_flow_control_is_class(self):
        """Test that ToolFlowControl exists and is a class."""
        assert isinstance(ToolFlowControl, type)

    def test_tool_flow_control_can_be_inherited(self):
        """Test that ToolFlowControl can be inherited."""
        class CustomControl(ToolFlowControl):
            pass

        assert issubclass(CustomControl, ToolFlowControl)
        instance = CustomControl()
        assert isinstance(instance, ToolFlowControl)
