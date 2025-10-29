"""
OpenTelemetry attribute helpers for Gen AI semantic conventions.

This module provides standardized attribute names and helper functions
following OpenTelemetry Gen AI specification and msgtrace conventions.

References:
- OpenTelemetry Gen AI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- msgtrace conventions: Custom extensions for cost tracking, workflows, etc.
"""

import json
from typing import Any, Dict, Optional, List

from opentelemetry.trace import Span


# ============================================================================
# OpenTelemetry Gen AI Semantic Conventions
# ============================================================================


class GenAIAttributes:
    """
    Standard OpenTelemetry Gen AI semantic convention attributes.

    These follow the official OpenTelemetry specification for generative AI
    operations and should be used consistently across all AI operations.
    """

    # Operation identification
    OPERATION_NAME = "gen_ai.operation.name"  # "chat", "embeddings", "rerank", "tool"
    SYSTEM = "gen_ai.system"  # System/provider name (legacy, prefer provider.name)
    PROVIDER_NAME = "gen_ai.provider.name"  # "openai", "anthropic", "google", "mistral"

    # Model identification
    REQUEST_MODEL = "gen_ai.request.model"  # Model requested (e.g., "gpt-4")
    RESPONSE_MODEL = "gen_ai.response.model"  # Actual model used (may differ)

    # Token usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"

    # Request parameters
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"

    # Response metadata
    RESPONSE_FINISH_REASON = "gen_ai.response.finish_reason"  # "stop", "length", "tool_calls"
    RESPONSE_ID = "gen_ai.response.id"  # Response ID from provider

    # Prompt and completion content (optional, may be large)
    PROMPT = "gen_ai.prompt"  # Prompt text (use sparingly)
    COMPLETION = "gen_ai.completion"  # Completion text (use sparingly)


# ============================================================================
# msgtrace Custom Attributes
# ============================================================================


class MsgTraceAttributes:
    """
    Custom msgtrace attribute extensions.

    These extend OpenTelemetry with msgflux/msgtrace-specific features
    like cost tracking, workflow context, and specialized operations.
    """

    # Cost tracking
    COST_INPUT = "msgtrace.cost.input"  # Cost for input tokens (float)
    COST_OUTPUT = "msgtrace.cost.output"  # Cost for output tokens (float)
    COST_TOTAL = "msgtrace.cost.total"  # Total cost (float)
    COST_CURRENCY = "msgtrace.cost.currency"  # Currency code (default: "USD")

    # Workflow/pipeline context
    WORKFLOW_NAME = "msgtrace.workflow.name"  # Workflow or pipeline name
    WORKFLOW_VERSION = "msgtrace.workflow.version"  # Version string
    WORKFLOW_STEP = "msgtrace.workflow.step"  # Step name within workflow
    WORKFLOW_STEP_INDEX = "msgtrace.workflow.step_index"  # Step index (int)

    # Tool execution
    TOOL_CALLINGS = "msgtrace.tool.callings"  # All tool calls in batch (JSON array)
    TOOL_RESPONSES = "msgtrace.tool.responses"  # Tool execution responses (JSON)

    # Retrieval/RAG operations
    RETRIEVAL_QUERY = "msgtrace.retrieval.query"  # Search query
    RETRIEVAL_DATABASE = "msgtrace.retrieval.database"  # Vector DB name
    RETRIEVAL_TOP_K = "msgtrace.retrieval.top_k"  # Number of results requested
    RETRIEVAL_RESULTS = "msgtrace.retrieval.results"  # Retrieved documents (JSON)
    RETRIEVAL_SCORE_THRESHOLD = "msgtrace.retrieval.score_threshold"  # Min score

    # Agent operations
    AGENT_NAME = "msgtrace.agent.name"  # Agent name
    AGENT_ID = "msgtrace.agent.id"  # Agent instance ID
    AGENT_CONVERSATION_ID = "msgtrace.agent.conversation_id"  # Conversation ID
    AGENT_ITERATION = "msgtrace.agent.iteration"  # Current iteration (int)
    AGENT_MAX_ITERATIONS = "msgtrace.agent.max_iterations"  # Max iterations allowed
    AGENT_SYSTEM_PROMPT = "msgtrace.agent.system_prompt"  # System prompt (optional)
    AGENT_TOOLS = "msgtrace.agent.tools"  # Available tools (JSON)

    # Service/framework metadata (generic for any framework)
    SERVICE_VERSION = "msgtrace.service.version"  # Service/library version (optional)
    MODULE_NAME = "msgtrace.module.name"  # Module/component name

# TODO: tirar tudo abaixo

# ============================================================================
# Helper Functions
# ============================================================================


def set_chat_attributes(
    span: Span,
    model: str,
    provider: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop_sequences: Optional[List[str]] = None,
) -> None:
    """
    Set standard attributes for chat completion spans.

    Args:
        span: OpenTelemetry span to add attributes to
        model: Model name (e.g., "gpt-4", "claude-3-sonnet")
        provider: Provider name (e.g., "openai", "anthropic")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        stop_sequences: Stop sequences

    Example:
        >>> with tracer.start_as_current_span("chat") as span:
        ...     set_chat_attributes(span, model="gpt-4", provider="openai", temperature=0.7)
    """
    span.set_attribute(GenAIAttributes.OPERATION_NAME, "chat")
    span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
    span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
    span.set_attribute(GenAIAttributes.SYSTEM, provider)  # Legacy

    if temperature is not None:
        span.set_attribute(GenAIAttributes.REQUEST_TEMPERATURE, temperature)
    if max_tokens is not None:
        span.set_attribute(GenAIAttributes.REQUEST_MAX_TOKENS, max_tokens)
    if top_p is not None:
        span.set_attribute(GenAIAttributes.REQUEST_TOP_P, top_p)
    if top_k is not None:
        span.set_attribute(GenAIAttributes.REQUEST_TOP_K, top_k)
    if frequency_penalty is not None:
        span.set_attribute(GenAIAttributes.REQUEST_FREQUENCY_PENALTY, frequency_penalty)
    if presence_penalty is not None:
        span.set_attribute(GenAIAttributes.REQUEST_PRESENCE_PENALTY, presence_penalty)
    if stop_sequences is not None:
        span.set_attribute(GenAIAttributes.REQUEST_STOP_SEQUENCES, json.dumps(stop_sequences))


def set_usage_attributes(
    span: Span,
    input_tokens: int,
    output_tokens: int,
    total_tokens: Optional[int] = None,
) -> None:
    """
    Set token usage attributes.

    Args:
        span: OpenTelemetry span
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (if different from input + output)

    Example:
        >>> set_usage_attributes(span, input_tokens=100, output_tokens=50)
    """
    span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)

    if total_tokens is not None:
        span.set_attribute(GenAIAttributes.USAGE_TOTAL_TOKENS, total_tokens)
    else:
        span.set_attribute(GenAIAttributes.USAGE_TOTAL_TOKENS, input_tokens + output_tokens)


def set_cost_attributes(
    span: Span,
    input_cost: float,
    output_cost: float,
    currency: str = "USD",
) -> None:
    """
    Set cost tracking attributes.

    Args:
        span: OpenTelemetry span
        input_cost: Cost for input tokens
        output_cost: Cost for output tokens
        currency: Currency code (default: "USD")

    Example:
        >>> set_cost_attributes(span, input_cost=0.003, output_cost=0.006)
    """
    total_cost = input_cost + output_cost

    span.set_attribute(MsgTraceAttributes.COST_INPUT, input_cost)
    span.set_attribute(MsgTraceAttributes.COST_OUTPUT, output_cost)
    span.set_attribute(MsgTraceAttributes.COST_TOTAL, total_cost)
    span.set_attribute(MsgTraceAttributes.COST_CURRENCY, currency)


def set_response_attributes(
    span: Span,
    response_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
    response_model: Optional[str] = None,
) -> None:
    """
    Set response metadata attributes.

    Args:
        span: OpenTelemetry span
        response_id: Response ID from provider
        finish_reason: Finish reason ("stop", "length", "tool_calls", etc.)
        response_model: Actual model used (may differ from request)

    Example:
        >>> set_response_attributes(span, response_id="chatcmpl-123", finish_reason="stop")
    """
    if response_id:
        span.set_attribute(GenAIAttributes.RESPONSE_ID, response_id)
    if finish_reason:
        span.set_attribute(GenAIAttributes.RESPONSE_FINISH_REASON, finish_reason)
    if response_model:
        span.set_attribute(GenAIAttributes.RESPONSE_MODEL, response_model)


def set_agent_attributes(
    span: Span,
    agent_name: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    agent_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    iteration: Optional[int] = None,
    max_iterations: Optional[int] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[str]] = None,
) -> None:
    """
    Set agent execution attributes.

    Args:
        span: OpenTelemetry span
        agent_name: Name of the agent
        model: Model being used by agent
        provider: Provider name
        agent_id: Unique agent instance ID
        conversation_id: Conversation/session ID
        iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        system_prompt: System prompt (optional, may be large)
        tools: List of available tool names

    Example:
        >>> set_agent_attributes(
        ...     span,
        ...     agent_name="support_agent",
        ...     model="gpt-4",
        ...     provider="openai",
        ...     iteration=1,
        ...     max_iterations=10
        ... )
    """
    span.set_attribute(GenAIAttributes.OPERATION_NAME, "agent")
    span.set_attribute(MsgTraceAttributes.AGENT_NAME, agent_name)

    if model:
        span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
    if provider:
        span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
    if agent_id:
        span.set_attribute(MsgTraceAttributes.AGENT_ID, agent_id)
    if conversation_id:
        span.set_attribute(MsgTraceAttributes.AGENT_CONVERSATION_ID, conversation_id)
    if iteration is not None:
        span.set_attribute(MsgTraceAttributes.AGENT_ITERATION, iteration)
    if max_iterations is not None:
        span.set_attribute(MsgTraceAttributes.AGENT_MAX_ITERATIONS, max_iterations)
    if system_prompt:
        span.set_attribute(MsgTraceAttributes.AGENT_SYSTEM_PROMPT, system_prompt)
    if tools:
        span.set_attribute(MsgTraceAttributes.AGENT_TOOLS, json.dumps(tools))


def set_retrieval_attributes(
    span: Span,
    query: str,
    database: str,
    top_k: int = 10,
    score_threshold: Optional[float] = None,
    results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Set retrieval/RAG operation attributes.

    Args:
        span: OpenTelemetry span
        query: Search query
        database: Vector database name
        top_k: Number of results to retrieve
        score_threshold: Minimum similarity score
        results: Retrieved documents (will be JSON serialized)

    Example:
        >>> set_retrieval_attributes(
        ...     span,
        ...     query="What is Python?",
        ...     database="pinecone",
        ...     top_k=5
        ... )
    """
    span.set_attribute(GenAIAttributes.OPERATION_NAME, "retrieval")
    span.set_attribute(MsgTraceAttributes.RETRIEVAL_QUERY, query)
    span.set_attribute(MsgTraceAttributes.RETRIEVAL_DATABASE, database)
    span.set_attribute(MsgTraceAttributes.RETRIEVAL_TOP_K, top_k)

    if score_threshold is not None:
        span.set_attribute(MsgTraceAttributes.RETRIEVAL_SCORE_THRESHOLD, score_threshold)

    if results:
        try:
            span.set_attribute(MsgTraceAttributes.RETRIEVAL_RESULTS, json.dumps(results))
        except (TypeError, ValueError):
            # If results can't be JSON serialized, store count only
            span.set_attribute(MsgTraceAttributes.RETRIEVAL_RESULTS, f"[{len(results)} results]")


def set_workflow_attributes(
    span: Span,
    workflow_name: str,
    version: Optional[str] = None,
    step: Optional[str] = None,
    step_index: Optional[int] = None,
) -> None:
    """
    Set workflow/pipeline attributes.

    Args:
        span: OpenTelemetry span
        workflow_name: Name of the workflow or pipeline
        version: Workflow version
        step: Name of the current step
        step_index: Index of the current step

    Example:
        >>> set_workflow_attributes(
        ...     span,
        ...     workflow_name="rag_pipeline",
        ...     version="1.0",
        ...     step="retrieval",
        ...     step_index=0
        ... )
    """
    span.set_attribute(MsgTraceAttributes.WORKFLOW_NAME, workflow_name)

    if version:
        span.set_attribute(MsgTraceAttributes.WORKFLOW_VERSION, version)
    if step:
        span.set_attribute(MsgTraceAttributes.WORKFLOW_STEP, step)
    if step_index is not None:
        span.set_attribute(MsgTraceAttributes.WORKFLOW_STEP_INDEX, step_index)


def set_embedding_attributes(
    span: Span,
    model: str,
    provider: str,
    input_texts: List[str],
    dimensions: Optional[int] = None,
) -> None:
    """
    Set embedding operation attributes.

    Args:
        span: OpenTelemetry span
        model: Embedding model name
        provider: Provider name
        input_texts: List of texts to embed
        dimensions: Embedding vector dimensions

    Example:
        >>> set_embedding_attributes(
        ...     span,
        ...     model="text-embedding-ada-002",
        ...     provider="openai",
        ...     input_texts=["Hello", "World"]
        ... )
    """
    span.set_attribute(GenAIAttributes.OPERATION_NAME, "embeddings")
    span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
    span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
    span.set_attribute("embedding.input_count", len(input_texts))

    if dimensions:
        span.set_attribute("embedding.dimensions", dimensions)
