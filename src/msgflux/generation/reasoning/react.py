from typing import TYPE_CHECKING, Any, ClassVar, List, Mapping, Optional
from uuid import uuid4

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import FlowControl, FlowResult
from msgflux.utils.chat import ChatBlock

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses


def _generate_short_id() -> str:
    """Generate a short unique ID for tool calls."""
    return str(uuid4())[:8]

REACT_SYSTEM_MESSAGE = """
You are an Agent. In each episode, you will be given the task as input.
And you can see your past trajectory so far.

Your goal is to use one or more of the supplied tools to collect any necessary
information for producing the `final_response`.

To do this, you will generate a `thought` containing your reasoning and plan.
Identify and define necessary `actions` by creating a list of `toolcall` objects.
You MUST use the available tools when needed to achieve the objective.
Include the function `name` and `arguments` for each call.
Await the observations for the tool calls.
Analyze the results and repeat the thought-action cycle if necessary.
Once the objective is met using the tools, provide the `final_answer`.

Do NOT provide the `final_answer` before completing the required tool calls.
Optional fields may be omitted.
"""

REACT_TOOLS_TEMPLATE = """
{% set tool_choice = tool_choice or "auto" %}
You are a function calling AI model. You may call one or more functions
to assist with the user query. Don't make assumptions about what values
to plug into functions. Here are the available tools:

{%- macro render_properties(properties, indent=0) -%}
{%- for arg, spec in properties.items() %}
{{ "  " * indent }}- {{ arg }} ({{ spec.get('type', 'unknown') }})
{%- if spec.get('description') %}
{{ "  " * (indent + 1) }}{{ spec['description'] }}
{% endif %}
{%- if spec.get('enum') %}
{{ "  " * (indent + 1) }}Options: {{ spec['enum'] | join(', ') }}
{%- endif %}
{%- if spec.get('type') == "object" and spec.get('properties') %}
{{ render_properties(spec['properties'], indent + 1) }}
{%- elif spec.get('type') == "array" and spec.get('items') and
spec['items'].get('type') == "object" %}
{{ "  " * (indent + 1) }}Array items:
{{ render_properties(spec['items']['properties'], indent + 2) }}
{%- endif %}
{%- endfor %}
{%- endmacro %}

{%- for tool in tool_schemas %}
<tool>{{ tool['function']['name'] }}
{{ tool['function']['description'] }}
{{ render_properties(tool['function']['parameters']['properties']) }}
</tool>
{%- endfor %}

Tool choice: {{ tool_choice }}

For each function call return a encoded json object with function name and arguments.
"""


class ToolCall(Struct):
    """A tool call representation.

    The `id` field is used to correlate tool calls with their results.
    IDs are generated automatically by `extract_flow_result` using short UUIDs.
    """

    name: str
    arguments: Optional[Any] = None
    id: Optional[str] = None
    result: Optional[Any] = None


class ReActStep(Struct):
    thought: str
    actions: List[ToolCall]


class ReAct(Struct, FlowControl):
    """ReAct (Reasoning + Acting) flow control schema.

    This schema implements the ReAct pattern where the LLM alternates between
    reasoning (thought) and acting (tool calls) until it reaches a final answer.
    """

    system_message: ClassVar[str] = REACT_SYSTEM_MESSAGE
    tools_template: ClassVar[str] = REACT_TOOLS_TEMPLATE

    current_step: Optional[ReActStep] = None
    final_answer: Optional[str] = None

    @classmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> FlowResult:
        """Extract flow information from ReAct response."""
        final_answer = raw_response.get("final_answer")
        if final_answer is not None:
            return FlowResult(
                is_complete=True,
                final_response=raw_response,
            )

        current_step = raw_response.get("current_step")
        if current_step is not None:
            actions = current_step.get("actions", [])
            tool_calls = []
            for act in actions:
                tool_id = _generate_short_id()
                act["id"] = tool_id
                tool_calls.append((tool_id, act.get("name"), act.get("arguments")))

            return FlowResult(
                is_complete=False,
                tool_calls=tool_calls,
                reasoning=current_step.get("thought"),
            )

        return FlowResult(
            is_complete=True,
            final_response=raw_response,
        )

    @classmethod
    def inject_tool_results(
        cls, raw_response: Mapping[str, Any], tool_results: "ToolResponses"
    ) -> Mapping[str, Any]:
        """Inject tool results back into ReAct structure."""
        current_step = raw_response.get("current_step")
        if current_step is not None:
            actions = current_step.get("actions", [])
            for act in actions:
                call = tool_results.get_by_id(act.get("id"))
                if call is not None:
                    act["result"] = call.result or call.error
        return raw_response

    @classmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration."""
        if messages and messages[-1].get("role") == "assistant":
            last_react_msg = messages[-1].get("content")
            react_state = msgspec.json.decode(last_react_msg)
            react_state.append(raw_response)
            messages[-1] = ChatBlock.assist(react_state)
        else:
            react_state = [raw_response]
            messages.append(ChatBlock.assist(react_state))
        return messages
