"""
OpenTelemetry attribute helpers for Gen AI semantic conventions.

This module provides standardized attribute names and helper functions
following OpenTelemetry Gen AI specification and msgtrace conventions.

References:
- OpenTelemetry Gen AI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

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
    SYSTEM_INSTRUCTIONS  = "gen_ai.system_instructions"

    # Agent
    AGENT_NAME = "gen_ai.agent.name" # Math Tutor
    AGENT_ID = "gen_ai.agent.id" # asst_5j66UpCpwteGg4YSxUnt7lPY
    AGENT_DESCRIPTION = "gen_ai.agent.description" # conv_5j66UpCpwteGg4YSxUnt7lPY
    CONVERSATION_ID = "gen_ai.conversation.id"

    # Tool
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_TYPE = "gen_ai.tool.type" # function; extension; datastore
    TOOL_CALL_ARGS = "gen_ai.tool.call.arguments"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"


class MsgTraceAttributes:
    """
    Custom msgtrace attribute extensions.

    These extend OpenTelemetry with msgflux/msgtrace-specific features
    like cost tracking, workflow context, and specialized operations.
    """

    # Service/framework metadata (generic for any framework)
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"

    # Platform
    PLATFORM_NUM_CPUS = "cpu.logical_number"

    # Module
    MODULE_NAME = "module.name"
    MODULE_TYPE = "module.type" # Agent, Transcriber, Tool, etc

    # Tool execution
    TOOL_EXECUTION_TYPE = "gen_ai.tool.execution.type" # local, remote
    TOOL_PROTOCOL = "gen_ai.tool.protocol" # mcp, a2a

    # Agent
    AGENT_RESPONSE = "gen_ai.agent.response"
