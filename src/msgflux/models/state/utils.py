"""Utility functions for ModelState operations."""

from typing import Any, Dict, List, Optional, Tuple, Union

import msgspec

from msgflux.models.state.model_state import ModelState
from msgflux.models.state.policies import Policy
from msgflux.models.state.types import ToolCall as KernelToolCall


def chatml_to_model_state(
    messages: List[Dict[str, Any]],
    policy: Optional[Policy] = None,
) -> ModelState:
    """Convert ChatML format (List[Dict]) to ModelState.

    Args:
        messages: List of messages in ChatML format (role, content, etc.)
        policy: Optional policy for the ModelState.

    Returns:
        ModelState populated with the messages.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> ck = chatml_to_model_state(messages)
        >>> ck.message_count
        2
    """
    kernel = ModelState(policy=policy)

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            kernel.add_user(content)

        elif role == "assistant":
            # Check for tool calls
            tool_calls_data = msg.get("tool_calls")
            if tool_calls_data:
                # Convert tool calls from ChatML to ModelState format
                tool_calls = []
                for tc in tool_calls_data:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = msgspec.json.decode(args.encode())
                        except Exception:
                            args = {}
                    tool_calls.append(
                        KernelToolCall(
                            id=tc.get("id", ""),
                            name=func.get("name", ""),
                            arguments=args,
                        )
                    )
                kernel.add_assistant(content, tool_calls=tool_calls)
            else:
                kernel.add_assistant(content)

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            kernel.add_tool_result(tool_call_id, content)

        # Skip system messages - they're handled separately via system_prompt

    return kernel


def ensure_model_state(
    obj: Union[ModelState, List[Dict[str, Any]]],
    policy: Optional[Policy] = None,
) -> ModelState:
    """Ensure object is a ModelState, converting if necessary.

    Args:
        obj: Either a ModelState or List[Dict] in ChatML format.
        policy: Optional policy to use if creating new ModelState.

    Returns:
        ModelState instance.

    Example:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> ck = ensure_model_state(messages)
        >>> isinstance(ck, ModelState)
        True
    """
    if isinstance(obj, ModelState):
        return obj
    elif isinstance(obj, list):
        return chatml_to_model_state(obj, policy=policy)
    else:
        raise TypeError(
            f"Expected ModelState or List[Dict], got {type(obj)}"
        )


def get_tool_lifecycle_configs(
    tool_library: Any,
    tool_callings: List[Tuple[str, str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Get lifecycle configuration for each tool from tool_config.

    Args:
        tool_library: ToolLibrary instance to get tools from.
        tool_callings: List of (tool_id, tool_name, arguments) tuples.

    Returns:
        Dictionary mapping tool_id to lifecycle configuration.

    Example:
        >>> configs = get_tool_lifecycle_configs(tool_library, [
        ...     ("call_1", "search", {"query": "test"}),
        ...     ("call_2", "fetch", {"url": "example.com"})
        ... ])
        >>> "call_1" in configs
        True
    """
    configs = {}
    for tool_id, tool_name, _ in tool_callings:
        tool = tool_library.get_tool(tool_name)
        if tool and hasattr(tool, "tool_config"):
            tc = tool.tool_config
            configs[tool_id] = {
                "ephemeral": getattr(tc, "ephemeral", False),
                "ephemeral_ttl": getattr(tc, "ephemeral_ttl", None),
                "result_importance": getattr(tc, "result_importance", None),
                "summarize_result": getattr(tc, "summarize_result", False),
            }
        else:
            configs[tool_id] = {
                "ephemeral": False,
                "ephemeral_ttl": None,
                "result_importance": None,
                "summarize_result": False,
            }
    return configs
