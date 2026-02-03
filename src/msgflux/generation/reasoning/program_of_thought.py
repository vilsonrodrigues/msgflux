"""Program of Thought control flow for code-based reasoning.

This module implements the Program of Thought pattern, where the LLM generates
Python code to solve problems instead of making discrete tool calls. The code
is executed in an environment with all registered tools available as Python functions.

References:
    - Program of Thoughts: https://arxiv.org/abs/2211.12588
    - DSPy ProgramOfThought: https://github.com/stanfordnlp/dspy
      (Prompts inspired by DSPy's implementation)
"""

from typing import TYPE_CHECKING, Any, List, Mapping, Optional

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import EnvironmentCall, FlowControl, FlowResult
from msgflux.utils.chat import ChatBlock

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses

POT_SYSTEM_MESSAGE = """
You are an intelligent agent that solves problems by writing and executing Python code.

For each iteration, you will receive the task as input and can see your past
trajectory (code executed and observations).

Your goal is to generate executable Python code that programmatically computes
the correct answer. For each iteration, generate code that either solves the
task or progresses towards the solution.

## Response Format

For each step, provide:
- `thought`: Your reasoning about the current situation and plan for next steps
- `code`: Python code to execute

When you have computed and verified the final answer, provide:
- `final_answer`: The complete answer to the task

## Guidelines

- EXPLORE FIRST: Look at your data before processing. Print samples, check types.
- ITERATE: Write small code snippets, observe outputs, then decide next steps.
- USE print(): Always print to see results. State persists between iterations.
- VERIFY: If results seem wrong, reconsider your approach before finalizing.
- Do NOT provide `final_answer` until you have computed and verified the result.
"""

POT_TOOLS_TEMPLATE = """
## Code Execution Environment

Your code will be executed in a sandboxed Python environment.
{% if environment_name %}Environment: `{{ environment_name }}`.{% endif %}
{% if tool_schemas %}

## Available Tools

The following functions are pre-loaded in the execution environment:

{% for tool in tool_schemas %}
- `{{ tool['function']['name'] }}`: {{ tool['function']['description'] }}
{%- endfor %}

Call these functions directly in your code as regular Python functions.
{% else %}

Standard Python libraries are available: re, json, collections, math, etc.
{% endif %}
"""


class ProgramOfThoughtStep(Struct):
    """A single step in Program of Thought reasoning."""

    thought: str
    code: str


class ProgramOfThought(Struct, FlowControl):
    """Program of Thought control flow for code-based reasoning.

    Instead of making discrete tool calls, the LLM writes Python code that
    is executed in an environment. All registered tools are available as
    callable Python functions within the code.

    Example:
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.environments import Environments
        >>> from msgflux.generation.reasoning import ProgramOfThought
        >>>
        >>> agent = Agent(
        ...     "solver",
        ...     model,
        ...     environment=Environment(environment=Environments.code("python")),
        ...     tools=[search, calculate],
        ...     generation_schema=ProgramOfThought,
        ... )
        >>> result = agent("What is 2 + 2?")

    Attributes:
        current_step: The current reasoning step with thought and code.
        final_answer: The final answer when reasoning is complete.
    """

    current_step: Optional[ProgramOfThoughtStep]
    final_answer: Optional[str]

    @classmethod
    def extract_flow_result(
        cls,
        raw_response: Mapping[str, Any],
        vars: Mapping[str, Any],  # noqa: ARG003
    ) -> FlowResult:
        """Extract flow information from ProgramOfThought response.

        If `final_answer` is present, the flow is complete.
        If `current_step` with `code` is present, create an environment call.
        """
        final_answer = raw_response.get("final_answer")
        if final_answer is not None:
            return FlowResult(
                is_complete=True,
                final_response=raw_response,
            )

        current_step = raw_response.get("current_step")
        if current_step is not None:
            code = current_step.get("code", "")
            thought = current_step.get("thought", "")

            if code:
                return FlowResult(
                    is_complete=False,
                    environment_call=EnvironmentCall(
                        action=code,
                        inject_vars=True,
                        inject_tools=True,
                    ),
                    reasoning=thought,
                )

        # No code and no final answer - treat as complete
        return FlowResult(
            is_complete=True,
            final_response=raw_response,
        )

    @classmethod
    def inject_environment_result(
        cls,
        raw_response: Mapping[str, Any],
        result: Mapping[str, Any],
        vars: Mapping[str, Any],  # noqa: ARG003
    ) -> Mapping[str, Any]:
        """Inject environment execution result back into ProgramOfThought structure.

        Args:
            raw_response: The original model response.
            result: The environment execution result dict with:
                - success: bool
                - output: str
                - error: Optional[str]
            vars: Current variables dict (not used in ProgramOfThought).

        Returns:
            Updated raw_response with result injected.
        """
        current_step = raw_response.get("current_step")
        if current_step is not None:
            if result.get("success"):
                output = result.get("output", "").strip()
                current_step["result"] = output if output else "(no output)"
            else:
                error = result.get("error", "Unknown error")
                current_step["result"] = f"Error: {error}"

        return raw_response

    @classmethod
    def inject_tool_results(
        cls,
        raw_response: Mapping[str, Any],
        tool_results: "ToolResponses",  # noqa: ARG003
        vars: Mapping[str, Any],  # noqa: ARG003
    ) -> Mapping[str, Any]:
        """Inject tool results back into the structure.

        Note: ProgramOfThought primarily uses environment_call, not tool_calls.
        This method is provided for interface compliance.
        """
        return raw_response

    @classmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration.

        Accumulates the trajectory of (thought, code, result) steps.
        """
        if messages and messages[-1].get("role") == "assistant":
            # Append to existing trajectory
            last_msg = messages[-1].get("content")
            trajectory = msgspec.json.decode(last_msg)
            trajectory.append(raw_response)
            messages[-1] = ChatBlock.assist(trajectory)
        else:
            # Start new trajectory
            trajectory = [raw_response]
            messages.append(ChatBlock.assist(trajectory))

        return messages


# Set class attributes for system message and tools template
ProgramOfThought.system_message = POT_SYSTEM_MESSAGE
ProgramOfThought.tools_template = POT_TOOLS_TEMPLATE
