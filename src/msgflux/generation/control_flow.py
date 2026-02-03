from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolResponses


@dataclass
class EnvironmentCall:
    """Request to execute code in an environment.

    Attributes:
        action: The code/action to execute in the environment.
        inject_vars: Whether to inject context variables into the environment.
        inject_tools: Whether to inject tools as callable functions.
    """

    action: str
    inject_vars: bool = True
    inject_tools: bool = True


@dataclass
class FlowResult:
    """Result of a flow control iteration.

    Attributes:
        is_complete: True if final answer reached.
        tool_calls: List of (id, name, args) to execute as tool calls.
        environment_call: Request to execute code in an environment.
        reasoning: Reasoning text (for verbose mode).
        final_response: Final response if complete.
    """

    is_complete: bool
    tool_calls: Optional[List[Tuple[str, str, Any]]] = None
    environment_call: Optional[EnvironmentCall] = None
    reasoning: Optional[str] = None
    final_response: Optional[Any] = None


class FlowControl:
    """Base class for creating custom flow controls.

    Each generation schema, such as ReAct or ProgramOfThought, can be treated
    as a custom flow control by inheriting from this class and implementing
    the required methods.

    Note: Model responses are converted to dotdict, so methods receive
    raw_response as a parameter rather than being called on the instance.

    All methods receive `vars` as an optional parameter, which is the current
    variables dict. FlowControl implementations can modify vars in-place to
    persist state across iterations (e.g., merge environment variables).

    Subclasses must implement:
        - extract_flow_result(raw_response, vars): Extract flow information
        - inject_tool_results(raw_response, tool_results, vars): Inject tool results
        - build_history(raw_response, messages): Build history message

    Optional methods:
        - inject_environment_result(raw_response, result, vars): Inject env results

    Class attributes:
        system_message: Optional system message template
        tools_template: Optional Jinja template for tool schemas
        inject_vars_info: Whether to inject variable type info as system_note
    """

    system_message: Optional[str] = None
    tools_template: Optional[str] = None
    inject_vars_info: bool = False

    @classmethod
    @abstractmethod
    def extract_flow_result(
        cls,
        raw_response: Mapping[str, Any],
        vars: Mapping[str, Any],
    ) -> FlowResult:
        """Extract flow information from the response.

        Args:
            raw_response: The model response (as dotdict)
            vars: Current variables dict (can be modified in-place if needed).

        Returns:
            FlowResult with:
            - is_complete: True if final_answer reached
            - tool_calls: List of (id, name, args) to execute
            - environment_call: Request for environment execution
            - reasoning: Reasoning text (for verbose mode)
            - final_response: Final response if complete
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def inject_tool_results(
        cls,
        raw_response: Mapping[str, Any],
        tool_results: "ToolResponses",
        vars: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Inject tool results back into the structure.

        Args:
            raw_response: The model response (as dotdict)
            tool_results: Results from tool execution
            vars: Current variables dict (can be modified in-place if needed).

        Returns:
            Updated raw_response with results injected
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Build history message for next iteration.

        Args:
            raw_response: The model response (as dotdict)
            messages: Current messages (chat history)

        Returns:
            Updated messages with step history
        """
        raise NotImplementedError

    @classmethod
    async def aextract_flow_result(
        cls,
        raw_response: Mapping[str, Any],
        vars: Mapping[str, Any],
    ) -> FlowResult:
        """Async version of extract_flow_result.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.extract_flow_result(raw_response, vars)

    @classmethod
    async def ainject_tool_results(
        cls,
        raw_response: Mapping[str, Any],
        tool_results: "ToolResponses",
        vars: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Async version of inject_tool_results.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.inject_tool_results(raw_response, tool_results, vars)

    @classmethod
    async def abuild_history(
        cls,
        raw_response: Mapping[str, Any],
        messages: List[Mapping[str, Any]],
    ) -> List[Mapping[str, Any]]:
        """Async version of build_history.

        Default implementation calls sync version.
        Override for async-specific behavior.
        """
        return cls.build_history(raw_response, messages)
