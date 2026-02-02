"""Recursive Language Model (RLM) control flow for code-based reasoning.

RLMs treat long contexts as part of an external environment rather than feeding
them directly to the model. The LLM writes Python code to programmatically
examine, decompose, and process data iteratively.

References:
    - Recursive Language Models (Zhang, Kraska, Khattab, 2025)
    - DSPy RLM: https://github.com/stanfordnlp/dspy
      (Prompts inspired by DSPy's implementation)
"""

from typing import TYPE_CHECKING, Any, Callable, ClassVar, List, Mapping, Optional, Union

import msgspec
from msgspec import Struct

from msgflux.generation.control_flow import EnvironmentCall, FlowControl, FlowResult
from msgflux.models.gateway import ModelGateway
from msgflux.models.types import ChatCompletionModel
from msgflux.nn import functional as F
from msgflux.nn.modules.lm import LM
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
- `thought`: Think step-by-step. What do you know? What remains? Plan next.
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
{% if environment_name %}Environment: `{{ environment_name }}`.{% endif %}

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


class RLM(Struct, FlowControl):
    """Recursive Language Model control flow for code-based reasoning.

    RLM is designed for tasks involving long contexts where the LLM writes
    Python code to programmatically examine and process data. The user provides
    tools (including `llm_query` if sub-LLM calls are needed) and context via
    `vars` which are injected into the environment.

    Example:
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.environments import Environments
        >>> from msgflux.generation.reasoning import RLM
        >>>
        >>> # User provides llm_query tool if needed
        >>> def llm_query(prompt: str) -> str:
        ...     return model(prompt)
        >>>
        >>> agent = Agent(
        ...     "analyzer",
        ...     model,
        ...     environment=Environment(environment=Environments.code("python")),
        ...     tools=[llm_query, search],
        ...     generation_schema=RLM,
        ... )
        >>> # Context injected via vars
        >>> result = agent("Analyze this data", vars={"context": long_text})

    Attributes:
        current_step: The current reasoning step with reasoning and code.
        final_answer: The final answer when reasoning is complete.
    """

    system_message: ClassVar[str] = RLM_SYSTEM_MESSAGE
    tools_template: ClassVar[str] = RLM_TOOLS_TEMPLATE

    current_step: Optional[RLMStep] = None
    final_answer: Optional[str] = None

    @classmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> FlowResult:
        """Extract flow information from RLM response.

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
            reasoning = current_step.get("reasoning", "")

            if code:
                return FlowResult(
                    is_complete=False,
                    environment_call=EnvironmentCall(
                        action=code,
                        inject_vars=True,
                        inject_tools=True,
                    ),
                    reasoning=reasoning,
                )

        # No code and no final answer - treat as complete
        return FlowResult(
            is_complete=True,
            final_response=raw_response,
        )

    @classmethod
    def inject_environment_result(
        cls, raw_response: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Inject environment execution result back into RLM structure.

        Args:
            raw_response: The original model response.
            result: The environment execution result dict with:
                - success: bool
                - output: str
                - error: Optional[str]

        Returns:
            Updated raw_response with result injected.
        """
        current_step = raw_response.get("current_step")
        if current_step is not None:
            if result.get("success"):
                output = result.get("output", "").strip()
                current_step["output"] = output if output else "(no output)"
            else:
                error = result.get("error", "Unknown error")
                current_step["output"] = f"Error: {error}"

        return raw_response

    @classmethod
    def inject_tool_results(
        cls,
        raw_response: Mapping[str, Any],
        tool_results: "ToolResponses",  # noqa: ARG003
    ) -> Mapping[str, Any]:
        """Inject tool results back into the structure.

        Note: RLM primarily uses environment_call, not tool_calls.
        This method is provided for interface compliance.
        """
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


DEFAULT_QUERY_LLM = {
    "name": "query_llm",
    "signature": "query, context: Optional[str] -> response",
    "description": (
        "Query a sub-LLM for semantic analysis. "
        "Use for understanding meaning, summarization, or extraction. "
        "IMPORTANT: Always use keyword arguments. "
        "Example: query_llm(query='What is X?', context=my_text)"
    ),
    "instructions": (
        "You are a helpful assistant that answers queries based on the provided context. "
        "Analyze the context carefully and provide a clear, concise response to the query."
    ),
    "templates": {"response": "{{ response }}"},
}


def make_rlm_tools(
    sub_lm: Union[ChatCompletionModel, ModelGateway, LM],
    *,
    query_config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Callable]:
    """Create RLM tools (query_llm and batched_query_llm) using nn.Agent.

    This factory creates tools that use nn.Agent internally with a signature
    of "query, context -> response", providing structured sub-LLM queries
    within RLM code execution environments.

    Args:
        sub_lm: The language model (nn.LM) to use for sub-queries.
        query_config: Optional configuration to override DEFAULT_QUERY_LLM.
            Keys: name, signature, description, instructions, templates.

    Returns:
        Dictionary with 'query_llm' and 'batched_query_llm' callable tools.

    Example:    
        >>> from msgflux import Model
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.generation.reasoning.rlm import make_rlm_tools, RLM
        >>>
        >>> lm = Model.chat_completion("openai/gpt-4.1-mini")
        >>> rlm_tools = make_rlm_tools(lm)
        >>>
        >>> agent = Agent(
        ...     "researcher",
        ...     model=lm,
        ...     environment=env,
        ...     tools=[search, *rlm_tools.values()],
        ...     generation_schema=RLM,
        ... )
    """
    from msgflux.nn.modules.agent import Agent

    # Merge config with defaults
    config = {**DEFAULT_QUERY_LLM, **(query_config or {})}

    query_llm_agent = Agent(model=sub_lm, **config)

    class QueryLLM:
        """Query a sub-LLM for semantic analysis."""

        name = "query_llm"
        description = (
            "Query a sub-LLM for semantic analysis. "
            "Use for understanding meaning, summarization, or extraction."
        )

        def __init__(self, agent: Optional[Callable] = None):
            self._agent = agent if agent else query_llm_agent

        def __call__(self, query: str, context: str = "") -> str:
            """Query the sub-LLM.

            Args:
                query: The question or task for the LLM.
                context: Optional context to analyze.

            Returns:
                The LLM's response as a string.
            """
            return self._agent(query=query, context=context)

        async def acall(self, query: str, context: str = "") -> str:
            """Async version of __call__."""
            return await self._agent.acall(query=query, context=context)

    class BatchedQueryLLM:
        """Query a sub-LLM with multiple prompts concurrently."""

        name = "batched_query_llm"
        description = (
            "Query a sub-LLM with multiple prompts concurrently. "
            "Much faster than calling query_llm multiple times sequentially."
        )

        def __init__(self, query_llm_instance: Optional[Callable] = None):
            self._query_llm = query_llm_instance if query_llm_instance else QueryLLM()

        def __call__(
            self,
            queries: list[str],
            contexts: list[str] | None = None,
        ) -> list[str]:
            """Query the sub-LLM with multiple prompts concurrently.

            Args:
                queries: List of query strings.
                contexts: Optional list of context strings (same length as queries).
                    If not provided, empty context is used for all queries.

            Returns:
                List of responses in the same order as queries.
            """
            if not queries:
                return []

            # Default contexts to empty strings
            if contexts is None:
                contexts = [""] * len(queries)
            elif len(contexts) != len(queries):
                raise ValueError(
                    f"Length mismatch: {len(queries)} queries vs {len(contexts)} contexts"
                )

            # Build kwargs list for map_gather
            kwargs_list = [
                {"query": q, "context": c}
                for q, c in zip(queries, contexts)
            ]

            # Execute in parallel
            args_list = [() for _ in queries]  # Empty positional args
            results = F.map_gather(
                self._query_llm, args_list=args_list, kwargs_list=kwargs_list
            )

            # Convert to list of strings, handling None for errors
            return [str(r) if r is not None else "[ERROR] Query failed" for r in results]

        async def acall(
            self,
            queries: list[str],
            contexts: list[str] | None = None,
        ) -> list[str]:
            """Async version of __call__."""
            if not queries:
                return []

            # Default contexts to empty strings
            if contexts is None:
                contexts = [""] * len(queries)
            elif len(contexts) != len(queries):
                raise ValueError(
                    f"Length mismatch: {len(queries)} queries vs {len(contexts)} contexts"
                )

            # Build kwargs list for amap_gather
            kwargs_list = [
                {"query": q, "context": c}
                for q, c in zip(queries, contexts)
            ]

            # Execute in parallel
            args_list = [() for _ in queries]  # Empty positional args
            results = await F.amap_gather(
                self._query_llm.acall, args_list=args_list, kwargs_list=kwargs_list
            )

            # Convert to list of strings, handling None for errors
            return [str(r) if r is not None else "[ERROR] Query failed" for r in results]

    return {
        "query_llm": QueryLLM(),
        "batched_query_llm": BatchedQueryLLM(),
    }
