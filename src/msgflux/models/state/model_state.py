"""ModelState - Unified context management for LLM conversations.

The main entry point for context window management, combining:
- Git-like version control (commits, branches, checkout)
- Provider-specific format conversion
- Compaction policies for context window management
- Full serialization for persistence
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import msgspec

from msgflux.models.state.adapters import get_adapter
from msgflux.models.state.policies import (
    LifecyclePolicy,
    Policy,
    PolicyResult,
    create_policy,
)
from msgflux.models.state.types import (
    ChatMessage,
    ContentBlock,
    LifecycleType,
    Reasoning,
    Role,
    ToolCall,
    assistant_message,
    tool_message,
    user_message,
)
from msgflux.models.state.versioning import (
    Branch,
    Commit,
    HistoryState,
    MessageHistory,
)


class InternalState(msgspec.Struct, kw_only=True):
    """Serializable state of ModelState."""

    history_state: HistoryState
    system_prompt: Optional[str] = None
    adapter_name: str = "openai-chat"
    policy_config: Optional[Dict[str, Any]] = None
    current_scope: Optional[str] = None
    current_turn: int = 0
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


class ModelState:
    """Unified model state for LLM conversation management.

    Combines:
    - MessageHistory for Git-like version control
    - Provider adapters for format conversion
    - Compaction policies for context management
    - Full serialization for persistence

    Example:
        >>> ck = ModelState()
        >>> ck.add_user("Hello!")
        >>> ck.add_assistant("Hi there!")
        >>>
        >>> # Get provider-specific format
        >>> openai_format = ck.to_provider()
        >>>
        >>> # Create checkpoint
        >>> ck.commit("After greeting")
        >>>
        >>> # Use scope for tool loops
        >>> with ck.scope("tool_execution"):
        ...     ck.add_tool_result("call_123", "Result here")
        >>>
        >>> # Apply compaction
        >>> ck.compact()
    """

    def __init__(
        self,
        adapter: str = "openai-chat",
        policy: Optional[Union[Policy, Dict[str, Any], str]] = None,
        summarizer: Optional[Callable[[List[ChatMessage]], str]] = None,
    ):
        """Initialize ModelState.

        Args:
            adapter: Adapter name (openai-chat, vllm).
            policy: Compaction policy configuration.
            summarizer: Function to summarize messages for compaction.
        """
        self._history = MessageHistory()
        self._adapter_name = adapter.lower()
        self._adapter = get_adapter(self._adapter_name)
        self._summarizer = summarizer

        # Policy setup
        if policy is not None:
            self._policy = create_policy(policy, summarizer=summarizer)
            self._policy_config = (
                policy if isinstance(policy, dict) else msgspec.to_builtins(policy)
            )
        else:
            self._policy = None
            self._policy_config = None

        # Lifecycle tracking
        self._current_scope: Optional[str] = None
        self._scope_stack: List[str] = []
        self._current_turn: int = 0

        # Working state (uncommitted changes)
        self._working_messages: List[ChatMessage] = []
        self._dirty = False

    # Properties

    @property
    def messages(self) -> List[ChatMessage]:
        """Get current messages (including uncommitted)."""
        if self._dirty:
            return self._working_messages
        return self._history.get_messages()

    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)

    @property
    def adapter_name(self) -> str:
        """Get current adapter name."""
        return self._adapter_name

    @adapter_name.setter
    def adapter_name(self, value: str) -> None:
        """Change adapter."""
        self._adapter_name = value.lower()
        self._adapter = get_adapter(self._adapter_name)

    @property
    def current_branch(self) -> Optional[str]:
        """Get current branch name."""
        return self._history.current_branch

    @property
    def branches(self) -> List[str]:
        """Get all branch names."""
        return self._history.branches

    @property
    def head(self) -> Commit:
        """Get current commit (HEAD)."""
        return self._history.head

    @property
    def turn(self) -> int:
        """Get current turn number."""
        return self._current_turn

    @property
    def current_scope(self) -> Optional[str]:
        """Get current scope ID."""
        return self._current_scope

    # Message Operations

    def add(self, message: ChatMessage) -> ChatMessage:
        """Add a message to the kernel."""
        if not self._dirty:
            self._working_messages = list(self._history.get_messages())
            self._dirty = True

        # Update index
        message = message.with_index(len(self._working_messages))

        # Set scope if in scope context
        if self._current_scope and message.metadata.scope_id is None:
            new_metadata = msgspec.structs.replace(
                message.metadata, scope_id=self._current_scope
            )
            message = msgspec.structs.replace(message, metadata=new_metadata)

        self._working_messages.append(message)
        return message

    def add_user(
        self,
        content: Union[str, List[ContentBlock]],
        lifecycle: LifecycleType = LifecycleType.PERMANENT,
        ttl_turns: Optional[int] = None,
        importance: float = 1.0,
        **metadata_kwargs,
    ) -> ChatMessage:
        """Add a user message.

        Args:
            content: Text or content blocks.
            lifecycle: Message lifecycle type.
            ttl_turns: Turns until expiration (for EPHEMERAL_TURNS).
            importance: Message importance (0.0-1.0).
            **metadata_kwargs: Additional metadata.

        Returns:
            The new message.
        """
        msg = user_message(
            content=content,
            lifecycle=lifecycle,
            ttl_turns=ttl_turns,
            importance=importance,
            **metadata_kwargs,
        )
        return self.add(msg)

    def add_assistant(
        self,
        content: Optional[Union[str, List[ContentBlock]]] = None,
        reasoning: Optional[Union[str, Reasoning]] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        lifecycle: LifecycleType = LifecycleType.PERMANENT,
        ttl_turns: Optional[int] = None,
        importance: float = 1.0,
        **metadata_kwargs,
    ) -> ChatMessage:
        """Add an assistant message."""
        msg = assistant_message(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            lifecycle=lifecycle,
            ttl_turns=ttl_turns,
            importance=importance,
            **metadata_kwargs,
        )
        return self.add(msg)

    def add_tool_result(
        self,
        call_id: str,
        content: str,
        name: Optional[str] = None,
        is_error: bool = False,
        lifecycle: LifecycleType = LifecycleType.PERMANENT,
        ttl_turns: Optional[int] = None,
        importance: float = 0.7,
        **metadata_kwargs,
    ) -> ChatMessage:
        """Add a tool result message."""
        msg = tool_message(
            call_id=call_id,
            content=content,
            name=name,
            is_error=is_error,
            lifecycle=lifecycle,
            ttl_turns=ttl_turns,
            importance=importance,
            **metadata_kwargs,
        )
        return self.add(msg)

    def add_from_response(self, response: Any, **kwargs) -> ChatMessage:
        """Add message from provider response."""
        msg = self._adapter.from_provider_response(response, **kwargs)
        return self.add(msg)

    def clear(self) -> None:
        """Clear all messages (creates new commit)."""
        self._working_messages = []
        self._dirty = True
        self.commit("Clear messages")

    def get_last(self, n: int = 1) -> List[ChatMessage]:
        """Get last N messages."""
        return self.messages[-n:]

    def get_by_role(self, role: Role) -> List[ChatMessage]:
        """Get messages by role."""
        return [m for m in self.messages if m.role == role]

    def get_by_lifecycle(self, lifecycle: LifecycleType) -> List[ChatMessage]:
        """Get messages by lifecycle type."""
        return [m for m in self.messages if m.metadata.lifecycle == lifecycle]

    # Version Control

    def commit(self, message: str = "", **metadata) -> Commit:
        """Create a commit (checkpoint) of current state."""
        msgs = self._working_messages if self._dirty else self._history.get_messages()
        commit = self._history.commit(
            messages=msgs,
            commit_message=message,
            turn=self._current_turn,
            scope=self._current_scope,
            **metadata,
        )
        self._working_messages = list(msgs)
        self._dirty = False
        return commit

    def checkout(
        self,
        ref: str,
        create: bool = False,
        orphan: bool = False,
    ) -> Commit:
        """Checkout a branch or commit."""
        if self._dirty:
            self.commit("Auto-commit before checkout")

        commit = self._history.checkout(ref, create=create, orphan=orphan)
        self._working_messages = []
        self._dirty = False
        return commit

    def branch(self, name: str, start_point: Optional[str] = None) -> Branch:
        """Create a new branch."""
        if self._dirty:
            self.commit("Auto-commit before branch")
        return self._history.branch(name, start_point)

    def merge(self, source: str, strategy: str = "concat") -> Commit:
        """Merge another branch."""
        if self._dirty:
            self.commit("Auto-commit before merge")
        commit = self._history.merge(source, strategy)
        self._working_messages = []
        self._dirty = False
        return commit

    def revert(self, steps: int = 1) -> Commit:
        """Revert to a previous commit."""
        if self._dirty:
            self.commit("Auto-commit before revert")
        commit = self._history.revert(steps)
        self._working_messages = []
        self._dirty = False
        return commit

    def reset(self, ref: str) -> Commit:
        """Reset current branch to a commit."""
        commit = self._history.reset(ref)
        self._working_messages = []
        self._dirty = False
        return commit

    def log(self, limit: int = 10) -> List[Commit]:
        """Get commit history."""
        return self._history.log(limit)

    def diff(self, ref1: str = "HEAD", ref2: str = "HEAD~1") -> Dict[str, Any]:
        """Compare two commits."""
        if ref2.startswith("HEAD~"):
            try:
                steps = int(ref2[5:])
                history = self.log(steps + 1)
                if len(history) > steps:
                    ref2 = history[steps].hash
            except (ValueError, IndexError):
                pass
        return self._history.diff(ref1, ref2)

    # Scope Management

    @contextmanager
    def scope(self, scope_id: str) -> Iterator[str]:
        """Context manager for message scopes.

        Messages added within the scope will be marked with the scope_id.
        When the scope exits, ephemeral_scope messages can be removed.

        Example:
            with ck.scope("tool_execution"):
                ck.add_assistant(
                    tool_calls=[...],
                    lifecycle=LifecycleType.EPHEMERAL_SCOPE
                )
                ck.add_tool_result(
                    call_id="...",
                    content="...",
                    lifecycle=LifecycleType.EPHEMERAL_SCOPE
                )
            # Scope ends, ephemeral_scope messages can be cleaned up

        Args:
            scope_id: Unique scope identifier.

        Yields:
            The scope_id.
        """
        self._begin_scope(scope_id)
        try:
            yield scope_id
        finally:
            self._end_scope()

    def _begin_scope(self, scope_id: str) -> None:
        """Begin a new scope (internal)."""
        self._scope_stack.append(scope_id)
        self._current_scope = scope_id

    def _end_scope(self) -> None:
        """End current scope (internal)."""
        if self._scope_stack:
            self._scope_stack.pop()
        self._current_scope = self._scope_stack[-1] if self._scope_stack else None

    # Legacy methods for backward compatibility
    def begin_scope(self, scope_id: str) -> str:
        """Begin a new scope (legacy - prefer using context manager)."""
        self._begin_scope(scope_id)
        return scope_id

    def end_scope(self) -> None:
        """End current scope (legacy - prefer using context manager)."""
        self._end_scope()

    # Turn Management

    def advance_turn(self) -> int:
        """Advance turn counter (decrements TTLs).

        Returns:
            New turn number.
        """
        self._current_turn += 1

        # Decrement TTLs and process lifecycle
        if self._policy and isinstance(self._policy, LifecyclePolicy):
            self._policy.advance_turn()

        self._apply_lifecycle_updates()

        return self._current_turn

    def _apply_lifecycle_updates(self) -> None:
        """Apply lifecycle updates (TTL decrement, scope cleanup)."""
        if not self._dirty:
            self._working_messages = list(self._history.get_messages())
            self._dirty = True

        updated = []
        for msg in self._working_messages:
            if msg.metadata.lifecycle == LifecycleType.EPHEMERAL_TURNS:
                if msg.metadata.ttl_turns is not None and msg.metadata.ttl_turns > 0:
                    new_metadata = msgspec.structs.replace(
                        msg.metadata, ttl_turns=msg.metadata.ttl_turns - 1
                    )
                    msg = msgspec.structs.replace(msg, metadata=new_metadata)
            updated.append(msg)

        self._working_messages = updated

    # Compaction

    def set_policy(
        self,
        policy: Union[Policy, Dict[str, Any], str],
        **kwargs,
    ) -> None:
        """Set compaction policy.

        Args:
            policy: Policy config, dict, or type string.
            **kwargs: Additional config options.
        """
        self._policy = create_policy(policy, summarizer=self._summarizer, **kwargs)

    def needs_compaction(self, token_count: int = 0) -> bool:
        """Check if compaction is needed."""
        if not self._policy:
            return False
        return self._policy.needs_compaction(self.messages, token_count)

    def compact(
        self,
        policy: Optional[Union[Policy, Dict[str, Any], str]] = None,
        auto_commit: bool = True,
    ) -> PolicyResult:
        """Apply compaction policy to reduce context size.

        Args:
            policy: Policy to use (default: kernel's policy).
            auto_commit: Whether to commit after compaction.

        Returns:
            PolicyResult with statistics.
        """
        if policy is not None:
            p = create_policy(policy, summarizer=self._summarizer)
        else:
            p = self._policy

        if not p:
            return PolicyResult(messages=self.messages, stats={"compacted": False})

        # Update lifecycle policy state if applicable
        if isinstance(p, LifecyclePolicy):
            p.current_turn = self._current_turn
            p.current_scope = self._current_scope

        # Apply policy
        result = p.apply(self.messages)

        # Update working state
        self._working_messages = result.messages
        self._dirty = True

        if auto_commit:
            self.commit(
                f"Compact: removed {len(result.removed)}, summarized {len(result.summarized)}"
            )

        return result

    def retroactive_edit(
        self,
        summary: str,
        from_index: int = 0,
        to_index: Optional[int] = None,
        keep_last_n: int = 3,
    ) -> int:
        """Replace a range of messages with a summary ("message to the past")."""
        if not self._dirty:
            self._working_messages = list(self._history.get_messages())
            self._dirty = True

        msgs = self._working_messages
        total = len(msgs)

        if to_index is None:
            to_index = max(from_index, total - keep_last_n)

        if to_index <= from_index:
            return 0

        before = msgs[:from_index]
        to_replace = msgs[from_index:to_index]
        after = msgs[to_index:]

        if not to_replace:
            return 0

        summary_msg = assistant_message(
            content=f"[Context Summary]\n{summary}",
            lifecycle=LifecycleType.PERMANENT,
            importance=0.9,
        )

        new_msgs = before + [summary_msg] + after
        self._working_messages = [m.with_index(i) for i, m in enumerate(new_msgs)]

        return len(to_replace)

    # Provider Format Conversion

    def to_provider(
        self,
        adapter: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert messages to provider-specific format.

        Note: System prompt is passed separately, not part of messages.

        Args:
            adapter: Adapter name (default: kernel's adapter).
            system_prompt: System prompt to include (passed separately to model).

        Returns:
            Dict ready to unpack into provider API call.
        """
        if adapter and adapter.lower() != self._adapter_name:
            a = get_adapter(adapter)
        else:
            a = self._adapter

        return a.to_provider_format(self.messages, system_prompt)

    # Serialization

    def get_state(self) -> InternalState:
        """Get serializable state."""
        if self._dirty:
            self.commit("Auto-commit for serialization")

        return InternalState(
            history_state=self._history.get_state(),
            adapter_name=self._adapter_name,
            policy_config=self._policy_config,
            current_scope=self._current_scope,
            current_turn=self._current_turn,
        )

    @classmethod
    def from_state(cls, state: InternalState) -> "ModelState":
        """Restore from serialized state."""
        kernel = cls.__new__(cls)
        kernel._history = MessageHistory.from_state(state.history_state)
        kernel._adapter_name = state.adapter_name
        kernel._adapter = get_adapter(state.adapter_name)
        kernel._summarizer = None
        kernel._current_scope = state.current_scope
        kernel._scope_stack = [state.current_scope] if state.current_scope else []
        kernel._current_turn = state.current_turn
        kernel._working_messages = []
        kernel._dirty = False

        if state.policy_config:
            kernel._policy = create_policy(state.policy_config)
            kernel._policy_config = state.policy_config
        else:
            kernel._policy = None
            kernel._policy_config = None

        return kernel

    def serialize(self) -> bytes:
        """Serialize to bytes (MessagePack)."""
        return msgspec.msgpack.encode(self.get_state())

    @classmethod
    def deserialize(cls, data: bytes) -> "ModelState":
        """Deserialize from bytes."""
        state = msgspec.msgpack.decode(data, type=InternalState)
        return cls.from_state(state)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return msgspec.json.encode(self.get_state()).decode()

    @classmethod
    def from_json(cls, json_str: str) -> "ModelState":
        """Deserialize from JSON string."""
        state = msgspec.json.decode(json_str.encode(), type=InternalState)
        return cls.from_state(state)

    # Utilities

    def gc(self) -> int:
        """Garbage collect unreachable commits and messages."""
        return self._history.gc()

    def stats(self) -> Dict[str, Any]:
        """Get kernel statistics."""
        msgs = self.messages
        return {
            "message_count": len(msgs),
            "commit_count": self._history.commit_count,
            "branch_count": len(self._history.branches),
            "current_branch": self._history.current_branch,
            "adapter": self._adapter_name,
            "turn": self._current_turn,
            "scope": self._current_scope,
            "dirty": self._dirty,
            "messages_by_role": {
                "user": len([m for m in msgs if m.role == Role.USER]),
                "assistant": len([m for m in msgs if m.role == Role.ASSISTANT]),
                "tool": len([m for m in msgs if m.role == Role.TOOL]),
            },
            "messages_by_lifecycle": {
                "permanent": len(
                    [m for m in msgs if m.metadata.lifecycle == LifecycleType.PERMANENT]
                ),
                "ephemeral_turns": len(
                    [
                        m
                        for m in msgs
                        if m.metadata.lifecycle == LifecycleType.EPHEMERAL_TURNS
                    ]
                ),
                "ephemeral_scope": len(
                    [
                        m
                        for m in msgs
                        if m.metadata.lifecycle == LifecycleType.EPHEMERAL_SCOPE
                    ]
                ),
                "summarizable": len(
                    [
                        m
                        for m in msgs
                        if m.metadata.lifecycle == LifecycleType.SUMMARIZABLE
                    ]
                ),
            },
        }

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def __repr__(self) -> str:
        branch = self._history.current_branch or "detached"
        return (
            f"ModelState(adapter={self._adapter_name}, "
            f"messages={len(self.messages)}, branch={branch})"
        )
