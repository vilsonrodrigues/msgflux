"""Tests for reasoning strategies (CoT, ReAct, Self-Consistency)."""

import pytest
from msgspec import Struct

from msgflux.generation.reasoning.cot import ChainOfThought
from msgflux.generation.reasoning.react import ReAct, ReActStep, ToolCall, REACT_SYSTEM_MESSAGE, REACT_TOOLS_TEMPLATE
from msgflux.generation.reasoning.self_consistency import SelfConsistency, ReasoningPath
from msgflux.generation.control_flow import ToolFlowControl


class TestChainOfThought:
    """Tests for Chain of Thought reasoning strategy."""

    def test_chain_of_thought_is_struct(self):
        """Test that ChainOfThought is a Struct."""
        assert issubclass(ChainOfThought, Struct)

    def test_chain_of_thought_has_required_fields(self):
        """Test that ChainOfThought has required fields."""
        cot = ChainOfThought(
            reasoning="First, we need to analyze the problem. Then we solve it.",
            final_answer="42"
        )
        assert cot.reasoning == "First, we need to analyze the problem. Then we solve it."
        assert cot.final_answer == "42"
