"""Git-like version control for message history.

Provides branching, commits, checkout, and merge capabilities for
managing conversation state over time.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from msgflux.models.state.types import ChatMessage


class Commit(msgspec.Struct, kw_only=True):
    """A snapshot of the message state at a point in time."""

    hash: str
    parent: Optional[str] = None
    message_hashes: Tuple[str, ...] = ()
    system_prompt: Optional[str] = None
    timestamp: float = msgspec.field(default_factory=time.time)
    commit_message: str = ""
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)

    @classmethod
    def create(
        cls,
        messages: List[ChatMessage],
        parent: Optional["Commit"] = None,
        system_prompt: Optional[str] = None,
        commit_message: str = "",
        **metadata,
    ) -> "Commit":
        """Create a new commit from messages."""
        message_hashes = tuple(m.hash for m in messages)

        content_data = {
            "parent": parent.hash if parent else None,
            "messages": message_hashes,
            "system": system_prompt,
            "ts": time.time(),
        }
        hash_input = msgspec.json.encode(content_data)
        commit_hash = hashlib.sha256(hash_input).hexdigest()[:12]

        return cls(
            hash=commit_hash,
            parent=parent.hash if parent else None,
            message_hashes=message_hashes,
            system_prompt=system_prompt,
            commit_message=commit_message,
            metadata=metadata,
        )


class Branch(msgspec.Struct, kw_only=True):
    """A named reference to a commit."""

    name: str
    commit: str
    created_at: float = msgspec.field(default_factory=time.time)
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)


class HistoryState(msgspec.Struct, kw_only=True):
    """Serializable state of MessageHistory for persistence."""

    commits: Dict[str, Commit]
    branches: Dict[str, Branch]
    messages: Dict[str, ChatMessage]
    head: str
    detached: bool


class MessageHistory:
    """Git-like version control for message history.

    Example:
        >>> history = MessageHistory()
        >>> history.commit([msg1, msg2], commit_message="Initial")
        >>> history.branch("experiment")
        >>> history.checkout("experiment")
        >>> history.commit([msg1, msg2, msg3], commit_message="Added msg3")
        >>> history.checkout("main")
        >>> history.merge("experiment")
    """

    def __init__(self):
        self._commits: Dict[str, Commit] = {}
        self._branches: Dict[str, Branch] = {}
        self._messages: Dict[str, ChatMessage] = {}
        self._head: str = "main"
        self._detached: bool = False

        # Create initial empty commit and main branch
        initial = Commit.create([], system_prompt=None, commit_message="Initial commit")
        self._commits[initial.hash] = initial
        self._branches["main"] = Branch(name="main", commit=initial.hash)

    # Properties

    @property
    def head(self) -> Commit:
        """Get the current commit (HEAD)."""
        if self._detached:
            return self._commits[self._head]
        return self._commits[self._branches[self._head].commit]

    @property
    def current_branch(self) -> Optional[str]:
        """Get current branch name (None if detached)."""
        return None if self._detached else self._head

    @property
    def branches(self) -> List[str]:
        """Get list of branch names."""
        return list(self._branches.keys())

    @property
    def commit_count(self) -> int:
        """Get total number of commits."""
        return len(self._commits)

    # Message Resolution

    def get_messages(self, commit: Optional[Commit] = None) -> List[ChatMessage]:
        """Get messages for a commit (default: HEAD)."""
        if commit is None:
            commit = self.head

        messages = []
        for i, msg_hash in enumerate(commit.message_hashes):
            msg = self._messages.get(msg_hash)
            if msg:
                if msg.index != i:
                    msg = msg.with_index(i)
                messages.append(msg)
        return messages

    def get_system_prompt(self, commit: Optional[Commit] = None) -> Optional[str]:
        """Get system prompt for a commit (default: HEAD)."""
        if commit is None:
            commit = self.head
        return commit.system_prompt

    # Commit Operations

    def commit(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        commit_message: str = "",
        **metadata,
    ) -> Commit:
        """Create a new commit with the given messages."""
        parent = self.head

        if system_prompt is None:
            system_prompt = parent.system_prompt

        indexed_messages = []
        for i, msg in enumerate(messages):
            if msg.index != i:
                msg = msg.with_index(i)
            indexed_messages.append(msg)
            if msg.hash not in self._messages:
                self._messages[msg.hash] = msg

        new_commit = Commit.create(
            messages=indexed_messages,
            parent=parent,
            system_prompt=system_prompt,
            commit_message=commit_message,
            **metadata,
        )
        self._commits[new_commit.hash] = new_commit

        if not self._detached:
            self._branches[self._head] = msgspec.structs.replace(
                self._branches[self._head], commit=new_commit.hash
            )
        else:
            self._head = new_commit.hash

        return new_commit

    # Branch Operations

    def branch(self, name: str, start_point: Optional[str] = None) -> Branch:
        """Create a new branch."""
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists")

        if start_point is None:
            commit_hash = self.head.hash
        elif start_point in self._branches:
            commit_hash = self._branches[start_point].commit
        elif start_point in self._commits:
            commit_hash = start_point
        else:
            raise ValueError(f"Unknown start point: {start_point}")

        new_branch = Branch(name=name, commit=commit_hash)
        self._branches[name] = new_branch
        return new_branch

    def delete_branch(self, name: str) -> None:
        """Delete a branch."""
        if name == self._head and not self._detached:
            raise ValueError("Cannot delete current branch")
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' does not exist")
        if name == "main":
            raise ValueError("Cannot delete 'main' branch")

        del self._branches[name]

    # Checkout Operations

    def checkout(
        self,
        ref: str,
        create: bool = False,
        orphan: bool = False,
    ) -> Commit:
        """Move HEAD to a different commit or branch."""
        if create:
            if ref in self._branches:
                raise ValueError(f"Branch '{ref}' already exists")

            if orphan:
                initial = Commit.create([], commit_message="Orphan branch start")
                self._commits[initial.hash] = initial
                self._branches[ref] = Branch(name=ref, commit=initial.hash)
            else:
                self._branches[ref] = Branch(name=ref, commit=self.head.hash)

            self._head = ref
            self._detached = False

        elif ref in self._branches:
            self._head = ref
            self._detached = False

        elif ref in self._commits:
            self._head = ref
            self._detached = True

        else:
            raise ValueError(f"Unknown ref: {ref}")

        return self.head

    # Merge Operations

    def merge(
        self,
        source: str,
        strategy: str = "concat",
    ) -> Commit:
        """Merge another branch into current branch."""
        if self._detached:
            raise ValueError("Cannot merge in detached HEAD state")

        if source in self._branches:
            source_commit = self._commits[self._branches[source].commit]
        elif source in self._commits:
            source_commit = self._commits[source]
        else:
            raise ValueError(f"Unknown source: {source}")

        current_msgs = self.get_messages()
        source_msgs = self.get_messages(source_commit)

        if strategy == "concat":
            merged = current_msgs + source_msgs
        elif strategy == "interleave":
            merged = self._interleave_messages(current_msgs, source_msgs)
        elif strategy == "replace":
            merged = source_msgs
        elif strategy == "dedupe":
            seen_hashes = set()
            merged = []
            for msg in current_msgs + source_msgs:
                if msg.hash not in seen_hashes:
                    seen_hashes.add(msg.hash)
                    merged.append(msg)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return self.commit(
            merged,
            commit_message=f"Merge {source} into {self._head}",
            merge_source=source,
            merge_strategy=strategy,
        )

    def _interleave_messages(
        self,
        msgs1: List[ChatMessage],
        msgs2: List[ChatMessage],
    ) -> List[ChatMessage]:
        """Interleave messages by timestamp."""

        def get_time(m: ChatMessage) -> float:
            return m.metadata.timestamp or 0

        combined = msgs1 + msgs2
        return sorted(combined, key=get_time)

    # History Navigation

    def revert(self, steps: int = 1) -> Commit:
        """Revert to a previous commit (detached HEAD)."""
        current = self.head
        for _ in range(steps):
            if current.parent is None:
                raise ValueError("Cannot revert past initial commit")
            current = self._commits[current.parent]

        self._head = current.hash
        self._detached = True
        return current

    def reset(self, ref: str, soft: bool = True) -> Commit:
        """Reset current branch to a different commit."""
        if self._detached:
            raise ValueError("Cannot reset in detached HEAD state")

        if ref in self._branches:
            commit_hash = self._branches[ref].commit
        elif ref in self._commits:
            commit_hash = ref
        else:
            raise ValueError(f"Unknown ref: {ref}")

        self._branches[self._head] = msgspec.structs.replace(
            self._branches[self._head], commit=commit_hash
        )

        return self._commits[commit_hash]

    def log(self, limit: int = 10) -> List[Commit]:
        """Get commit history from HEAD."""
        history = []
        current = self.head

        while current and len(history) < limit:
            history.append(current)
            if current.parent:
                current = self._commits.get(current.parent)
            else:
                break

        return history

    def diff(self, ref1: str, ref2: str) -> Dict[str, Any]:
        """Compare two commits."""
        commit1 = self._resolve_ref(ref1)
        commit2 = self._resolve_ref(ref2)

        msgs1 = set(commit1.message_hashes)
        msgs2 = set(commit2.message_hashes)

        return {
            "added": list(msgs2 - msgs1),
            "removed": list(msgs1 - msgs2),
            "common": list(msgs1 & msgs2),
            "ref1_count": len(commit1.message_hashes),
            "ref2_count": len(commit2.message_hashes),
        }

    def _resolve_ref(self, ref: str) -> Commit:
        """Resolve a ref to a commit."""
        if ref == "HEAD":
            return self.head
        if ref in self._branches:
            return self._commits[self._branches[ref].commit]
        if ref in self._commits:
            return self._commits[ref]
        raise ValueError(f"Unknown ref: {ref}")

    # Serialization

    def get_state(self) -> HistoryState:
        """Get serializable state for persistence."""
        return HistoryState(
            commits=self._commits,
            branches=self._branches,
            messages=self._messages,
            head=self._head,
            detached=self._detached,
        )

    @classmethod
    def from_state(cls, state: HistoryState) -> "MessageHistory":
        """Restore from serialized state."""
        history = cls.__new__(cls)
        history._commits = dict(state.commits)
        history._branches = dict(state.branches)
        history._messages = dict(state.messages)
        history._head = state.head
        history._detached = state.detached
        return history

    def serialize(self) -> bytes:
        """Serialize entire history to bytes."""
        return msgspec.msgpack.encode(self.get_state())

    @classmethod
    def deserialize(cls, data: bytes) -> "MessageHistory":
        """Deserialize history from bytes."""
        state = msgspec.msgpack.decode(data, type=HistoryState)
        return cls.from_state(state)

    # Garbage Collection

    def gc(self, keep_unreachable: bool = False) -> int:
        """Garbage collect unreachable commits and messages."""
        if keep_unreachable:
            return 0

        reachable_commits = set()
        for branch in self._branches.values():
            commit = self._commits.get(branch.commit)
            while commit:
                reachable_commits.add(commit.hash)
                if commit.parent:
                    commit = self._commits.get(commit.parent)
                else:
                    break

        reachable_messages = set()
        for commit_hash in reachable_commits:
            commit = self._commits[commit_hash]
            reachable_messages.update(commit.message_hashes)

        removed = 0
        for commit_hash in list(self._commits.keys()):
            if commit_hash not in reachable_commits:
                del self._commits[commit_hash]
                removed += 1

        for msg_hash in list(self._messages.keys()):
            if msg_hash not in reachable_messages:
                del self._messages[msg_hash]
                removed += 1

        return removed

    def __repr__(self) -> str:
        branch = self.current_branch or f"detached@{self._head[:8]}"
        return f"MessageHistory(commits={len(self._commits)}, branch={branch})"
