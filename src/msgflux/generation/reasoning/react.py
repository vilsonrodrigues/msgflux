from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union
from uuid import uuid4

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult
from msgflux.utils.chat import ChatBlock

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses

REACT_SYSTEM_MESSAGE = """
You are an Agent. In each episode, you will be given the task as input.
And you can see your past trajectory so far.

Your goal is to use one or more of the supplied tools to collect any necessary
information for producing the `final_answer`.

To do this, you will generate a `thought` containing your reasoning and plan.
Identify and define necessary `actions` by creating a list of action objects.
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


class Argument(Struct):
    """A single named argument for a tool call."""

    name: str
    value: Union[str, int, float, bool, List[str]]


class Action(Struct):
    """A tool call action with name and typed arguments."""

    name: str
    arguments: Optional[List[Argument]] = None


class ReAct(Struct, ToolFlowControl):
    thought: Optional[str] = None
    actions: Optional[List[Action]] = None
    final_answer: Optional[str] = None

    @classmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> ToolFlowResult:
        """Extract flow information from ReAct response."""
        final_answer = raw_response.get("final_answer")
        if final_answer is not None:
            return ToolFlowResult(
                is_complete=True,
                tool_calls=None,
                reasoning=None,
                final_response=raw_response,
            )

        actions = raw_response.get("actions")
        if actions:
            tool_calls = []
            for act in actions:
                tool_id = str(uuid4())
                act["_id"] = tool_id
                # Convert List[Argument] to dict for tool executor
                args = act.get("arguments")
                if isinstance(args, list):
                    args_dict = {
                        a["name"] if isinstance(a, dict) else a.name: (
                            a["value"] if isinstance(a, dict) else a.value
                        )
                        for a in args
                    }
                else:
                    args_dict = args
                tool_calls.append((tool_id, act.get("name"), args_dict))

            return ToolFlowResult(
                is_complete=False,
                tool_calls=tool_calls,
                reasoning=raw_response.get("thought"),
                final_response=None,
            )

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
        """Inject tool results back into ReAct structure as observations."""
        actions = raw_response.get("actions") or []
        observations = []
        for act in actions:
            call = tool_results.get_by_id(act.get("_id"))
            if call is not None:
                observations.append(
                    {
                        "tool": act.get("name"),
                        "result": call.result or call.error,
                    }
                )
        raw_response["observations"] = observations
        return raw_response

    @classmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration."""
        step = {
            "thought": raw_response.get("thought"),
            "actions": [
                {"name": a.get("name"), "arguments": a.get("arguments")}
                for a in (raw_response.get("actions") or [])
            ],
            "observations": raw_response.get("observations", []),
        }
        if messages and messages[-1].get("role") == "assistant":
            last_react_msg = messages[-1].get("content")
            react_state = msgspec.json.decode(last_react_msg)
            react_state.append(step)
            messages[-1] = ChatBlock.assist(react_state)
        else:
            react_state = [step]
            messages.append(ChatBlock.assist(react_state))
        return messages


ReAct.system_message = REACT_SYSTEM_MESSAGE
ReAct.tools_template = REACT_TOOLS_TEMPLATE
