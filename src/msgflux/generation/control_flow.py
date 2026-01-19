from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses


@dataclass
class ToolFlowResult:
    """Result of a tool flow iteration.

    Attributes:
        is_complete: True if final answer reached
        tool_calls: List of (id, name, args) to execute
        reasoning: Reasoning text (for verbose mode)
        final_response: Final response if complete
    """

    is_complete: bool
    tool_calls: Optional[List[Tuple[str, str, Any]]]
    reasoning: Optional[str]
    final_response: Optional[Any]


class ToolFlowControl:
    """Base class for creating custom tool flow controls.

    Each generation schema, such as ReAct, can be treated as a custom
    tool flow control by inheriting from this class and implementing
    the required methods.

    Note: Model responses are converted to dotdict, so methods receive
    raw_response as a parameter rather than being called on the instance.

    Subclasses must implement:
        - extract_flow_result(raw_response): Extract flow information
        - inject_results(raw_response, tool_results): Inject tool results
        - build_history(raw_response, model_state): Build history message

    And async versions:
        - aextract_flow_result(raw_response): Async version
        - ainject_results(raw_response, tool_results): Async version
        - abuild_history(raw_response, model_state): Async version

    Class attributes:
        system_message: Optional system message template
        tools_template: Optional Jinja template for tool schemas
    """

    system_message: Optional[str] = None
    tools_template: Optional[str] = None

    @classmethod
    @abstractmethod
    def extract_flow_result(cls, raw_response: Mapping[str, Any]) -> ToolFlowResult:
        """Extract flow information from the response.

        Args:
            raw_response: The model response (as dotdict)

        Returns:
            ToolFlowResult with:
            - is_complete: True if final_answer reached
            - tool_calls: List of (id, name, args) to execute
            - reasoning: Reasoning text (for verbose mode)
            - final_response: Final response if complete
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inject_results(
        cls, raw_response: Mapping[str, Any], tool_results: "ToolResponses"
    ) -> Mapping[str, Any]:
        """Inject tool results back into the structure.

        Args:
            raw_response: The model response (as dotdict)
            tool_results: Results from tool execution

        Returns:
            Updated raw_response with results injected
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        model_state: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration.

        Args:
            raw_response: The model response (as dotdict)
            model_state: Current state (chat history)

        Returns:
            Updated model_state with step history
        """
        raise NotImplementedError

    @classmethod
    async def aextract_flow_result(
        cls, raw_response: Mapping[str, Any]
    ) -> ToolFlowResult:
        """Async version of extract_flow_result.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.extract_flow_result(raw_response)

    @classmethod
    async def ainject_results(
        cls, raw_response: Mapping[str, Any], tool_results: "ToolResponses"
    ) -> Mapping[str, Any]:
        """Async version of inject_results.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.inject_results(raw_response, tool_results)

    @classmethod
    async def abuild_history(
        cls,
        raw_response: Mapping[str, Any],
        model_state: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Async version of build_history.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.build_history(raw_response, model_state)
