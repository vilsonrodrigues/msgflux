"""Recursive Language Model (RLM) control flow for code-based reasoning.

RLMs treat long contexts as part of an external environment rather than feeding
them directly to the model. The LLM writes Python code to programmatically
examine, decompose, and process data iteratively.

References:
    - Recursive Language Models (Zhang, Kraska, Khattab, 2025)
    - DSPy RLM: https://github.com/stanfordnlp/dspy
      (Prompts inspired by DSPy's implementation)
"""

from typing import TYPE_CHECKING, Any, List, Mapping, Optional
from uuid import uuid4

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult
from msgflux.utils.chat import ChatBlock

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses

RLM_SYSTEM_MESSAGE = """
You are tasked with producing outputs given inputs through code execution.

You have access to a Python REPL environment. Write Python code and it will be
executed. You will see the output, then write more code based on what you
learned. This is an iterative process.

## Response Format

For each step, provide:
- `reasoning`: Think step-by-step. What do you know? What remains? Plan next.
- `code`: Python code to execute (use print() to see results)

When you have computed and verified the final answer, provide:
- `final_answer`: The complete answer to the task

## Guidelines

1. EXPLORE FIRST - Look at your data before processing. Print samples, check
   types/lengths, understand the structure.

2. ITERATE - Write small code snippets, observe outputs, then decide next
   steps. State persists between iterations.

3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected),
   reconsider your approach.

4. USE print() - Always print to see results. Variables persist between
   iterations but you can only see what you print.

5. MINIMIZE RETYPING - When values are long or precise, re-access them via
   variables instead of retyping. Use code to parse/compute.

Do NOT provide `final_answer` until you have computed and verified the result.
"""

RLM_TOOLS_TEMPLATE = """
## Code Execution Environment

Your code will be executed in a sandboxed Python environment.
{% if sandbox_name %}Sandbox: `{{ sandbox_name }}`.{% endif %}

Available in the environment:
- Variables from `vars` are pre-loaded (access them directly by name)
- Standard libraries: re, json, collections, math, itertools, etc.
{% if tool_schemas %}

## Available Tools

The following functions are pre-loaded:

{% for tool in tool_schemas %}
- `{{ tool['function']['name'] }}`: {{ tool['function']['description'] }}
{%- endfor %}

Call these functions directly in your code.
{% endif %}
"""


class RLMStep(Struct):
    """A single step in RLM reasoning."""

    reasoning: str
    code: str


# Default sandbox name for code execution
DEFAULT_SANDBOX_NAME = "execute_code"


class RLM(Struct, ToolFlowControl):
    """Recursive Language Model control flow for code-based reasoning.

    RLM is designed for tasks involving long contexts where the LLM writes
    Python code to programmatically examine and process data. The user provides
    tools (including `llm_query` if sub-LLM calls are needed) and context via
    `vars` which are injected into the sandbox environment.

    Example:
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.environments.sandboxes import DenoPyodideSandbox
        >>> from msgflux.generation.reasoning import RLM
        >>>
        >>> # User provides llm_query tool if needed
        >>> def llm_query(prompt: str) -> str:
        ...     return model(prompt)
        >>>
        >>> agent = Agent(
        ...     "analyzer",
        ...     model,
        ...     sandbox=Environment(sandbox=DenoPyodideSandbox()),
        ...     tools=[llm_query, search],
        ...     generation_schema=RLM,
        ... )
        >>> # Context injected via vars
        >>> result = agent("Analyze this data", vars={"context": long_text})

    Attributes:
        current_step: The current reasoning step with reasoning and code.
        final_answer: The final answer when reasoning is complete.
    """

    current_step: Optional[RLMStep]
    final_answer: Optional[str]

    @classmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> ToolFlowResult:
        """Extract flow information from RLM response.

        If `final_answer` is present, the flow is complete.
        If `current_step` with `code` is present, create a tool call to execute.
        """
        final_answer = raw_response.get("final_answer")
        if final_answer is not None:
            return ToolFlowResult(
                is_complete=True,
                tool_calls=None,
                reasoning=None,
                final_response=raw_response,
            )

        current_step = raw_response.get("current_step")
        if current_step is not None:
            code = current_step.get("code", "")
            reasoning = current_step.get("reasoning", "")

            if code:
                tool_id = str(uuid4())
                tool_calls = [(tool_id, DEFAULT_SANDBOX_NAME, {"code": code})]

                return ToolFlowResult(
                    is_complete=False,
                    tool_calls=tool_calls,
                    reasoning=reasoning,
                    final_response=None,
                )

        # No code and no final answer - treat as complete
        return ToolFlowResult(
            is_complete=True,
            tool_calls=None,
            reasoning=None,
            final_response=raw_response,
        )

    @classmethod
    def inject_results(
        cls, raw_response: Mapping[str, Any], tool_results: "ToolResponses"
    ) -> Mapping[str, Any]:
        """Inject execution results back into RLM structure."""
        current_step = raw_response.get("current_step")
        if current_step is not None and tool_results.tool_calls:
            call = tool_results.tool_calls[0]
            sandbox_result = call.parameters.get("_sandbox_result", {})

            if sandbox_result:
                if sandbox_result.get("success"):
                    output = sandbox_result.get("output", "").strip()
                    current_step["output"] = output if output else "(no output)"
                else:
                    error = sandbox_result.get("error", "Unknown error")
                    current_step["output"] = f"Error: {error}"
            else:
                current_step["output"] = call.result or call.error or "(no output)"

        return raw_response

    @classmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration."""
        if messages and messages[-1].get("role") == "assistant":
            last_msg = messages[-1].get("content")
            trajectory = msgspec.json.decode(last_msg)
            trajectory.append(raw_response)
            messages[-1] = ChatBlock.assist(trajectory)
        else:
            trajectory = [raw_response]
            messages.append(ChatBlock.assist(trajectory))

        return messages


# Set class attributes for system message and tools template
RLM.system_message = RLM_SYSTEM_MESSAGE
RLM.tools_template = RLM_TOOLS_TEMPLATE
