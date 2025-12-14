"""ToolSignal - Dynamic tool behavior signaling."""

from typing import Any

from msgflux.dotdict import dotdict


class ToolSignal(dotdict):
    """Dynamic signaling for tool behavior.

    Allows a tool to override static configurations (like `return_direct`)
    at runtime based on execution results. Inherits from dotdict for
    attribute access and serialization.

    Args:
        result: The actual result of the tool (any serializable type)
        **kwargs: Override any tool_config property (e.g., return_direct=True)

    Example:
        Basic usage with return_direct override:

            >>> from msgflux.tools import ToolSignal
            >>>
            >>> @tool_config(return_direct=False)
            ... def smart_search(query: str):
            ...     '''Search that decides dynamically.'''
            ...     result = search(query)
            ...
            ...     if result.is_final:
            ...         return ToolSignal(result=result.answer, return_direct=True)
            ...
            ...     return result  # Default behavior

        Accessing signal properties:

            >>> signal = ToolSignal(result="data", return_direct=True, priority=10)
            >>> signal.result
            'data'
            >>> signal.return_direct
            True
            >>> signal.get("priority")
            10

        Serialization (for telemetry):

            >>> signal.to_dict()
            {'result': 'data', 'return_direct': True, 'priority': 10}
            >>> signal.to_json()
            b'{"result":"data","return_direct":true,"priority":10}'

    Supported overrides (same keys as tool_config):
        - return_direct: bool - Return directly to user instead of agent loop

    Note:
        - Backward compatible: tools can return normal values
        - Unknown overrides are ignored by ToolLibrary
        - Only use ToolSignal when you need to override behavior dynamically
    """

    def __init__(self, result: Any, **kwargs: Any):
        """Initialize ToolSignal.

        Args:
            result: The actual result of the tool
            **kwargs: Override any tool_config property
        """
        super().__init__({"result": result, **kwargs})
