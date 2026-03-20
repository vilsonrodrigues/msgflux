"""Tests for template functionality."""

import pytest

from msgflux.generation.templates import (
    PromptSpec,
    SYSTEM_PROMPT_TEMPLATE,
    EXPECTED_OUTPUTS_TEMPLATE,
)


class TestPromptSpec:
    """Tests for PromptSpec class."""

    def test_prompt_spec_has_system_message(self):
        """Test that PromptSpec has SYSTEM_MESSAGE attribute."""
        assert hasattr(PromptSpec, "SYSTEM_MESSAGE")
        assert PromptSpec.SYSTEM_MESSAGE == "Who are you"

    def test_prompt_spec_has_instructions(self):
        """Test that PromptSpec has INSTRUCTIONS attribute."""
        assert hasattr(PromptSpec, "INSTRUCTIONS")
        assert PromptSpec.INSTRUCTIONS == "How you should do"
