"""LLM Query tools for code execution environments.

This module provides tools for querying sub-LLMs within code execution
environments like RLM (Recursive Language Model) and other flow controls.

The `make_llm_query_tools` factory creates `query_llm` and `batched_query_llm`
tools that can be injected into sandboxed Python environments.
"""

from typing import Any, Callable, Mapping, Optional, Union

from msgflux.models.gateway import ModelGateway
from msgflux.models.types import ChatCompletionModel
from msgflux.nn import functional as F
from msgflux.nn.modules.lm import LM

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


def make_llm_query_tools(
    sub_lm: Union[ChatCompletionModel, ModelGateway, LM],
    *,
    query_config: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Callable]:
    """Create LLM query tools (query_llm and batched_query_llm) using nn.Agent.

    This factory creates tools that use nn.Agent internally with a signature
    of "query, context -> response", providing structured sub-LLM queries
    within code execution environments.

    Args:
        sub_lm: The language model to use for sub-queries.
        query_config: Optional configuration to override DEFAULT_QUERY_LLM.
            Keys: name, signature, description, instructions, templates.

    Returns:
        Dictionary with 'query_llm' and 'batched_query_llm' callable tools.

    Example:
        >>> from msgflux import Model
        >>> from msgflux.nn import Agent, Environment
        >>> from msgflux.tools import make_llm_query_tools
        >>> from msgflux.generation.reasoning import RLM
        >>>
        >>> lm = Model.chat_completion("openai/gpt-4.1-mini")
        >>> llm_tools = make_llm_query_tools(lm)
        >>>
        >>> agent = Agent(
        ...     "researcher",
        ...     model=lm,
        ...     environment=env,
        ...     tools=[search, *llm_tools.values()],
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
                {"query": q, "context": c} for q, c in zip(queries, contexts)
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
                {"query": q, "context": c} for q, c in zip(queries, contexts)
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
