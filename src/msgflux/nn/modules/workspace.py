import threading
from typing import Any, Callable, Dict, List, Optional

from msgflux.tools.config import tool_config
from msgflux.utils.msgspec import msgspec_dumps

__all__ = [
    "Workspace",
]


class Workspace:
    """Thread-safe shared state for multi-agent collaboration.

    A Workspace provides a key-value store with an audit trail that allows
    multiple agents to read and write artifacts. It is designed to be used
    with `Team` for iterative collaboration via workspace tools.

    The workspace is thread-safe, using a lock to protect concurrent access
    from agents running in parallel (e.g., via `F.scatter_gather`).

    Args:
        permissions: Optional mapping of agent names to lists of allowed
            artifact keys they can write to. When set, agents can only
            write to their permitted keys. Agents not listed in the
            permissions dict can write to any key.

    Examples:
        >>> ws = Workspace()
        >>> ws.put("draft", "Initial draft", author="writer")
        >>> ws.get("draft")
        'Initial draft'
        >>> ws.list_keys()
        ['draft']
        >>> ws.snapshot()
        '{"draft": "Initial draft"}'

        >>> permissions = {"writer": ["draft"], "editor": ["final_answer"]}
        >>> ws = Workspace(permissions=permissions)
        >>> ws.put("draft", "text", author="writer")  # OK
        >>> ws.put("final_answer", "text", author="writer")  # Blocked
        "Agent 'writer' cannot write to 'final_answer'. Allowed keys: draft"
    """

    def __init__(self, permissions: Optional[Dict[str, List[str]]] = None) -> None:
        self._artifacts: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._permissions: Dict[str, List[str]] = permissions or {}

    def put(self, key: str, value: Any, author: Optional[str] = None) -> Optional[str]:
        """Write an artifact to the workspace.

        Args:
            key: The artifact key.
            value: The artifact value.
            author: Optional author identifier for the audit trail.

        Returns:
            None on success, or an error message string if the author
            is not permitted to write to the given key.
        """
        if self._permissions and author:
            allowed = self._permissions.get(author)
            if allowed is not None and key not in allowed:
                return (
                    f"Agent '{author}' cannot write to '{key}'. "
                    f"Allowed keys: {', '.join(allowed)}"
                )
        with self._lock:
            self._artifacts[key] = value
            self._history.append({"action": "put", "key": key, "author": author})
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Read an artifact from the workspace.

        Args:
            key: The artifact key.
            default: Default value if key is not found.

        Returns:
            The artifact value, or default if not found.
        """
        with self._lock:
            return self._artifacts.get(key, default)

    def list_keys(self) -> List[str]:
        """List all artifact keys in the workspace.

        Returns:
            A list of artifact keys.
        """
        with self._lock:
            return list(self._artifacts.keys())

    def snapshot(self) -> str:
        """Serialize all artifacts to a JSON string for LLM context.

        Returns:
            A JSON string representation of all artifacts.
        """
        with self._lock:
            return msgspec_dumps(self._artifacts)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return a copy of the audit trail.

        Returns:
            A list of history entries.
        """
        with self._lock:
            return list(self._history)

    def get_tools(self) -> List[Callable]:
        """Return workspace tools ready to be passed to an Agent.

        The tools use `inject_vars=True` so they can access the workspace
        instance via `kwargs["vars"]["workspace"]`.

        Returns:
            A list of tool functions: [write_artifact, read_artifact,
            list_artifacts].
        """
        return [_write_artifact, _read_artifact, _list_artifacts]


@tool_config(inject_vars=True)
def _write_artifact(key: str, content: str, **kwargs) -> str:
    """Write an artifact to the shared workspace.

    Args:
        key: The artifact key.
        content: The content to write.

    Returns:
        Confirmation message, or an error message if the write is not permitted.
    """
    workspace: Workspace = kwargs["vars"]["workspace"]
    author = kwargs["vars"].get("_agent_name")
    error = workspace.put(key, content, author=author)
    if error:
        return error
    return f"Artifact '{key}' written successfully."


@tool_config(inject_vars=True)
def _read_artifact(key: str, **kwargs) -> str:
    """Read an artifact from the shared workspace.

    Args:
        key: The artifact key to read.

    Returns:
        The artifact content, or an error message if not found.
    """
    workspace: Workspace = kwargs["vars"]["workspace"]
    value = workspace.get(key)
    if value is None:
        return f"Artifact '{key}' not found."
    return str(value)


@tool_config(inject_vars=True)
def _list_artifacts(**kwargs) -> str:
    """List all artifact keys in the shared workspace.

    Returns:
        A comma-separated list of keys, or a message if empty.
    """
    workspace: Workspace = kwargs["vars"]["workspace"]
    keys = workspace.list_keys()
    if not keys:
        return "No artifacts in workspace."
    return ", ".join(keys)
