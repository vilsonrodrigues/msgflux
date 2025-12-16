"""Model State - Unified context management for LLM conversations.

This module provides:
- ModelState: Main interface for managing conversation context
- Git-like versioning (commits, branches, checkout, merge)
- Provider adapters for format conversion
- Compaction policies for context window management
- Full serialization for persistence

Example:
    >>> from msgflux.models.state import ModelState, LifecycleType, Policy
    >>>
    >>> # Create state with policy
    >>> state = ModelState(
    ...     adapter="openai-chat",
    ...     policy=Policy(type="sliding_window", max_messages=50),
    ... )
    >>>
    >>> # Add messages
    >>> state.add_user("Hello!")
    >>> state.add_assistant("Hi there!")
    >>>
    >>> # Add temporary message (expires after 3 turns)
    >>> state.add_user(
    ...     "Temporary context",
    ...     lifecycle=LifecycleType.EPHEMERAL_TURNS,
    ...     ttl_turns=3,
    ... )
    >>>
    >>> # Use scope for tool execution
    >>> with state.scope("tool_loop"):
    ...     state.add_assistant(tool_calls=[...])
    ...     state.add_tool_result("call_123", "Result")
    >>>
    >>> # Get provider format (system prompt passed separately)
    >>> messages = state.to_provider(system_prompt="You are helpful")
    >>>
    >>> # Version control
    >>> state.commit("Checkpoint")
    >>> state.branch("experiment")
    >>> state.checkout("experiment")
    >>>
    >>> # Compaction when needed
    >>> if state.needs_compaction():
    ...     state.compact()
    >>>
    >>> # Serialize for persistence
    >>> data = state.serialize()
    >>> restored = ModelState.deserialize(data)
"""

# Main interface
# Adapters
from msgflux.models.state.adapters import (
    MessageAdapter,
    OpenAIChatAdapter,
    VLLMAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)
from msgflux.models.state.model_state import InternalState, ModelState

# Policies
from msgflux.models.state.policies import (
    CompactionPolicy,
    ImportancePolicy,
    LifecyclePolicy,
    Policy,
    PolicyResult,
    PositionBasedPolicy,
    SlidingWindowPolicy,
    create_policy,
)

# Types
from msgflux.models.state.types import (
    AudioContent,
    ChatMessage,
    ContentBlock,
    FileContent,
    ImageContent,
    LifecycleType,
    MessageMetadata,
    Reasoning,
    Role,
    TextContent,
    ToolCall,
    ToolResult,
    VideoContent,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)

# Utils
from msgflux.models.state.utils import (
    chatml_to_model_state,
    ensure_model_state,
)

# Versioning
from msgflux.models.state.versioning import (
    Branch,
    Commit,
    HistoryState,
    MessageHistory,
)

__all__ = [
    # Main interface
    "ModelState",
    "InternalState",
    # Types
    "ChatMessage",
    "Role",
    "LifecycleType",
    "ContentBlock",
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "FileContent",
    "ToolCall",
    "ToolResult",
    "Reasoning",
    "MessageMetadata",
    # Factory functions
    "user_message",
    "assistant_message",
    "system_message",
    "tool_message",
    # Versioning
    "MessageHistory",
    "Commit",
    "Branch",
    "HistoryState",
    # Policies
    "Policy",
    "PolicyResult",
    "CompactionPolicy",
    "SlidingWindowPolicy",
    "PositionBasedPolicy",
    "LifecyclePolicy",
    "ImportancePolicy",
    "create_policy",
    # Adapters
    "MessageAdapter",
    "OpenAIChatAdapter",
    "VLLMAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    # Utils
    "chatml_to_model_state",
    "ensure_model_state",
]
