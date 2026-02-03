"""CodeAct control flow for tool-augmented code-based reasoning.

This module implements the CodeAct pattern, where the LLM generates Python code
that can call tools directly as functions. Unlike ProgramOfThought which focuses
on pure computation, CodeAct emphasizes tool usage through code.

The model writes code that calls tools (e.g., `result = search("query")`)
and the environment executes the code with all tools available as functions.

References:
    - CodeAct: https://arxiv.org/abs/2402.01030
    - DSPy CodeAct: https://github.com/stanfordnlp/dspy
"""

from typing import TYPE_CHECKING, Any, ClassVar, List, Mapping, Optional

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import EnvironmentCall, FlowControl, FlowResult
from msgflux.utils.chat import ChatBlock

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses

CODEACT_SYSTEM_MESSAGE = """
You are an intelligent agent that solves problems by writing Python code.

For each iteration, you will receive the task as input and can see your past
trajectory (code executed and observations).

Your goal is to generate executable Python code that collects necessary information
by calling tools and computes the correct answer.

## Response Format

For each step, provide:
- `thought`: Your reasoning about the current situation and plan for next steps
- `code`: Python code to execute

When you have computed and verified the final answer, provide:
- `final_answer`: The complete answer to the task

## Guidelines

- USE TOOLS: Call the available tools directly in your code as regular functions.
- PRINT RESULTS: Always use print() to see tool outputs and intermediate results.
- ITERATE: Write small code snippets, observe outputs, then decide next steps.
- STATE PERSISTS: Variables from previous iterations are available in later ones.
- VERIFY: Check results before providing final_answer.
- Do NOT provide `final_answer` until you have collected and verified all information.
"""

CODEACT_TOOLS_TEMPLATE = """
## Code Execution Environment

Your code will be executed in a sandboxed Python environment.
{% if environment_name %}Environment: `{{ environment_name }}`.{% endif %}
{% if tool_schemas %}

## Available Tools

The following functions are pre-loaded in the execution environment.
Call them directly in your code as regular Python functions:

{% for tool in tool_schemas %}
### `{{ tool['function']['name'] }}`

{{ tool['function']['description'] }}

{% set params = tool['function']['parameters'] %}
{% if params and params.get('properties') %}
Parameters:
{%- for param_name, param_spec in params['properties'].items() %}
- `{{ param_name }}` ({{ param_spec.get('type', 'any') }})
{%- endfor %}
{% endif %}

{% endfor %}

Example usage:
```python
# Call tools directly
result = {{ tool_schemas[0]['function']['name'] }}(...)
print(result)
```
{% else %}

Standard Python libraries are available: re, json, collections, math, etc.
{% endif %}
"""


class CodeActStep(Struct):
    """A single step in CodeAct reasoning.

    Attributes:
        thought: The agent's reasoning about current situation and plan.
        code: Python code to execute in the environment.
    """

    thought: str
    code: str


class CodeAct(Struct, FlowControl):
    """CodeAct control flow for tool-augmented code-based reasoning.

    CodeAct combines the flexibility of code generation with tool calling.
    The model writes Python code that calls tools directly as functions,
    and the code is executed in an environment with all tools available.

    This is useful for tasks that require:
    - Complex control flow (loops, conditionals)
    - Data transformation between tool calls
    - Combining results from multiple tools

    Example:
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.environments import Environments
        >>> from msgflux.generation.reasoning import CodeAct
        >>>
        >>> def search(query: str) -> str:
        ...     '''Search for information.'''
        ...     return f"Results for: {query}"
        >>>
        >>> agent = Agent(
        ...     "researcher",
        ...     model,
        ...     environment=Environment(environment=Environments.code("python")),
        ...     tools=[search],
        ...     generation_schema=CodeAct,
        ... )
        >>> result = agent("Find information about Python")
        >>> # The model will generate code like:
        >>> # result = search("Python")
        >>> # print(result)

    Attributes:
        current_step: The current reasoning step with thought and code.
        final_answer: The final answer when reasoning is complete.
    """

    system_message: ClassVar[str] = CODEACT_SYSTEM_MESSAGE
    tools_template: ClassVar[str] = CODEACT_TOOLS_TEMPLATE

    current_step: Optional[CodeActStep] = None
    final_answer: Optional[str] = None

    @classmethod
    def extract_flow_result(
        cls,
        raw_response: Mapping[str, Any],
        vars: Mapping[str, Any],  # noqa: ARG003
    ) -> FlowResult:
        """Extract flow information from CodeAct response.

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
        """Inject environment execution result back into CodeAct structure.

        Args:
            raw_response: The original model response.
            result: The environment execution result dict with:
                - success: bool
                - output: str
                - error: Optional[str]
            vars: Current variables dict (not used in CodeAct).

        Returns:
            Updated raw_response with result injected into current_step.result.
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

        Note: CodeAct primarily uses environment_call, not tool_calls.
        Tools are called within the code executed in the environment.
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
