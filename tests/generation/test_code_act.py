"""Tests for CodeAct reasoning strategy."""

from unittest.mock import MagicMock

import pytest
from msgspec import Struct

from msgflux.generation.control_flow import FlowControl
from msgflux.generation.reasoning.code_act import (
    CODEACT_SYSTEM_MESSAGE,
    CODEACT_TOOLS_TEMPLATE,
    CodeAct,
    CodeActStep,
    CodeCall,
)


class TestCodeActStructure:
    """Tests for CodeAct data structures."""

    def test_code_call_is_struct(self):
        """Test that CodeCall is a Struct."""
        assert issubclass(CodeCall, Struct)

    def test_code_call_has_required_fields(self):
        """Test that CodeCall has required fields."""
        code_call = CodeCall(code="print('hello')")
        assert code_call.code == "print('hello')"

    def test_code_act_step_is_struct(self):
        """Test that CodeActStep is a Struct."""
        assert issubclass(CodeActStep, Struct)

    def test_code_act_step_has_required_fields(self):
        """Test that CodeActStep has required fields."""
        step = CodeActStep(
            thought="I need to search for information",
            actions=CodeCall(code="result = search('query')"),
        )
        assert step.thought == "I need to search for information"
        assert step.actions.code == "result = search('query')"


class TestCodeActFlowControl:
    """Tests for CodeAct FlowControl interface implementation."""

    def test_code_act_inherits_flow_control(self):
        """Test that CodeAct inherits from FlowControl."""
        assert issubclass(CodeAct, FlowControl)

    def test_code_act_has_class_attributes(self):
        """Test that CodeAct has system_message and tools_template."""
        assert CodeAct.system_message == CODEACT_SYSTEM_MESSAGE
        assert CodeAct.tools_template == CODEACT_TOOLS_TEMPLATE

    def test_extract_flow_result_with_final_answer(self):
        """Test extract_flow_result when final_answer is present."""
        raw_response = {"current_step": None, "final_answer": "The answer is 42"}
        result = CodeAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.environment_call is None
        assert result.tool_calls is None
        assert result.reasoning is None
        assert result.final_response is raw_response

    def test_extract_flow_result_with_code(self):
        """Test extract_flow_result when current_step has code."""
        raw_response = {
            "current_step": {
                "thought": "I need to search for information",
                "actions": {"code": "result = search('Python')\nprint(result)"},
            },
            "final_answer": None,
        }
        result = CodeAct.extract_flow_result(raw_response)

        assert result.is_complete is False
        assert result.environment_call is not None
        assert result.environment_call.action == "result = search('Python')\nprint(result)"
        assert result.environment_call.inject_vars is True
        assert result.environment_call.inject_tools is True
        assert result.reasoning == "I need to search for information"

    def test_extract_flow_result_empty_state(self):
        """Test extract_flow_result with no step and no final_answer."""
        raw_response = {"current_step": None, "final_answer": None}
        result = CodeAct.extract_flow_result(raw_response)

        assert result.is_complete is True
        assert result.final_response is raw_response

    def test_inject_environment_result_success(self):
        """Test injecting successful execution result."""
        raw_response = {
            "current_step": {
                "thought": "Search for info",
                "actions": {"code": "result = search('query')\nprint(result)"},
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "Found: Python documentation\n", "error": None}
        updated = CodeAct.inject_environment_result(raw_response, result)

        assert updated["current_step"]["actions"]["result"] == "Found: Python documentation"

    def test_inject_environment_result_error(self):
        """Test injecting error result."""
        raw_response = {
            "current_step": {
                "thought": "Search for info",
                "actions": {"code": "result = undefined_function()"},
            },
            "final_answer": None,
        }

        result = {
            "success": False,
            "output": "",
            "error": "NameError: name 'undefined_function' is not defined",
        }
        updated = CodeAct.inject_environment_result(raw_response, result)

        assert "Error:" in updated["current_step"]["actions"]["result"]
        assert "NameError" in updated["current_step"]["actions"]["result"]

    def test_inject_environment_result_no_output(self):
        """Test injecting result with no output."""
        raw_response = {
            "current_step": {
                "thought": "Assign variable",
                "actions": {"code": "x = 42"},
            },
            "final_answer": None,
        }

        result = {"success": True, "output": "", "error": None}
        updated = CodeAct.inject_environment_result(raw_response, result)

        assert updated["current_step"]["actions"]["result"] == "(no output)"

    def test_inject_tool_results_noop(self):
        """Test inject_tool_results does nothing (CodeAct uses environment)."""
        raw_response = {"current_step": {"thought": "Test", "actions": {}}, "final_answer": None}
        mock_results = MagicMock()

        result = CodeAct.inject_tool_results(raw_response, mock_results)

        assert result is raw_response

    def test_build_history_new_message(self):
        """Test build_history adds new assistant message."""
        raw_response = {
            "current_step": {
                "thought": "Testing",
                "actions": {"code": "print(1)", "result": "1"},
            },
            "final_answer": None,
        }

        messages = []
        result = CodeAct.build_history(raw_response, messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"

    def test_build_history_append_to_existing(self):
        """Test build_history appends to existing assistant message."""
        import msgspec

        first_response = {
            "current_step": {
                "thought": "First step",
                "actions": {"code": "x = search('a')", "result": "result a"},
            },
            "final_answer": None,
        }
        second_response = {
            "current_step": {
                "thought": "Second step",
                "actions": {"code": "y = search('b')", "result": "result b"},
            },
            "final_answer": None,
        }

        messages = []
        CodeAct.build_history(first_response, messages)
        CodeAct.build_history(second_response, messages)

        # Should have one message with both steps
        assert len(messages) == 1
        content = msgspec.json.decode(messages[0]["content"])
        assert len(content) == 2

    @pytest.mark.asyncio
    async def test_async_methods_work(self):
        """Test that async methods work (default to sync)."""
        raw_response = {"current_step": None, "final_answer": "Done"}

        result = await CodeAct.aextract_flow_result(raw_response)
        assert result.is_complete is True

        raw_response2 = await CodeAct.ainject_tool_results(raw_response, MagicMock())
        assert raw_response2 is raw_response

        history = await CodeAct.abuild_history(raw_response, [])
        assert len(history) == 1


class TestCodeActSystemPrompt:
    """Tests for CodeAct system message and tools template."""

    def test_system_message_mentions_tools(self):
        """Test system message mentions using tools."""
        assert "tools" in CODEACT_SYSTEM_MESSAGE.lower()
        assert "code" in CODEACT_SYSTEM_MESSAGE.lower()

    def test_tools_template_has_function_details(self):
        """Test tools template includes function details."""
        assert "{{ tool['function']['name'] }}" in CODEACT_TOOLS_TEMPLATE
        assert "{{ tool['function']['description'] }}" in CODEACT_TOOLS_TEMPLATE
        assert "parameters" in CODEACT_TOOLS_TEMPLATE.lower()

    def test_tools_template_has_example_usage(self):
        """Test tools template includes example usage."""
        assert "example" in CODEACT_TOOLS_TEMPLATE.lower()
