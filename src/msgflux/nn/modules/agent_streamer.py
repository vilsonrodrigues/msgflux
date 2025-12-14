"""AgentStreamer - Agent with real-time streaming and inline tool execution."""

from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)
from uuid import uuid4

from msgflux.message import Message
from msgflux.models.streaming import ParsedToolCall, StreamingXMLParser
from msgflux.nn.modules.agent import Agent
from msgflux.utils.chat import ChatBlock


@dataclass
class StreamEvent:
    """Event emitted during agent streaming.

    Types:
    - text: Text content to display to the user
    - reasoning: Model reasoning/thinking content
    - tool_start: Tool execution is starting
    - tool_result: Tool execution completed
    - done: Streaming completed
    - error: An error occurred
    """

    type: Literal["text", "reasoning", "tool_start", "tool_result", "done", "error"]
    content: Optional[str] = None
    tool_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    error: Optional[str] = None


class AgentStreamer(Agent):
    """Agent with real-time streaming and inline tool execution.

    Unlike the base Agent which uses SDK-based tool calling (separate blocks),
    AgentStreamer uses text-based streaming where the model generates text
    with embedded XML tool calls. This allows:

    - Text and tool calls in the same stream
    - Immediate text display while tools are detected
    - Inline tool execution as they are parsed
    - Callbacks for real-time UI updates

    Example:
        >>> from msgflux.nn import AgentStreamer
        >>> from msgflux import Model
        >>>
        >>> def on_text(text):
        ...     print(text, end="", flush=True)
        >>>
        >>> def on_tool_start(tool_id, name, params):
        ...     print(f"\\n[Calling {name}...]")
        >>>
        >>> agent = AgentStreamer(
        ...     name="assistant",
        ...     model=Model("openai/gpt-4o"),
        ...     tools=[search, calculator],
        ...     on_text=on_text,
        ...     on_tool_start=on_tool_start,
        ... )
        >>>
        >>> # Stream events
        >>> for event in agent.stream("What is 2+2?"):
        ...     if event.type == "done":
        ...         print("\\nDone!")
    """

    def __init__(
        self,
        name: str,
        model: Any,
        *,
        on_text: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """Initialize AgentStreamer.

        Args:
            name: Agent name.
            model: Chat completion model.
            on_text: Callback for text chunks. Called with each text fragment.
            on_reasoning: Callback for reasoning content.
            on_tool_start: Callback when tool starts. Args: (tool_id, name, params)
            on_tool_result: Callback when tool completes. Args: (tool_id, result)
            on_error: Callback for errors.
            max_iterations: Maximum tool call iterations to prevent infinite loops.
            **kwargs: Additional arguments passed to Agent.
        """
        # Ensure stream is not set in config (we handle streaming ourselves)
        config = kwargs.pop("config", {}) or {}
        config["stream"] = False  # We use our own streaming
        kwargs["config"] = config

        super().__init__(name, model, **kwargs)

        self.on_text = on_text
        self.on_reasoning = on_reasoning
        self.on_tool_start = on_tool_start
        self.on_tool_result = on_tool_result
        self.on_error = on_error
        self.max_iterations = max_iterations

    def stream(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        **kwargs,
    ) -> Iterator[StreamEvent]:
        """Stream agent response with real-time tool execution.

        Yields StreamEvent objects as the model generates content and
        tools are executed.

        Args:
            message: Input message (string, dict, or Message object).
            **kwargs: Additional arguments passed to _prepare_task.

        Yields:
            StreamEvent objects with types: text, reasoning, tool_start,
            tool_result, done, or error.
        """
        try:
            inputs = self._prepare_task(message, **kwargs)
            model_state = inputs["model_state"]
            vars = inputs["vars"]
            model_preference = inputs.get("model_preference")

            parser = StreamingXMLParser()
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1

                # Prepare model execution params
                exec_params = self._prepare_model_execution(
                    model_state=model_state,
                    prefilling=self.prefilling,
                    model_preference=model_preference,
                    vars=vars,
                )

                # Get system prompt and messages
                system_prompt = exec_params.get("system_prompt")
                messages = exec_params.get("messages", [])

                # Add tool schemas to system prompt for XML-based calling
                tool_schemas = self.tool_library.get_tool_json_schemas()
                if tool_schemas:
                    tools_prompt = self._build_tools_system_prompt(tool_schemas)
                    if system_prompt:
                        system_prompt = tools_prompt + "\n\n" + system_prompt
                    else:
                        system_prompt = tools_prompt

                # Stream from model
                text_buffer = ""
                tools_to_execute: List[ParsedToolCall] = []

                for chunk in self.lm.model.stream(
                    messages=messages,
                    system_prompt=system_prompt,
                    prefilling=exec_params.get("prefilling"),
                ):
                    if chunk.type == "text":
                        # Feed to XML parser
                        text, tools = parser.feed(chunk.content or "")

                        # Yield text immediately
                        if text:
                            text_buffer += text
                            yield StreamEvent(type="text", content=text)
                            if self.on_text:
                                self.on_text(text)

                        # Collect detected tools
                        if tools:
                            tools_to_execute.extend(tools)

                    elif chunk.type == "reasoning":
                        yield StreamEvent(type="reasoning", content=chunk.content)
                        if self.on_reasoning:
                            self.on_reasoning(chunk.content or "")

                # Flush any remaining buffer
                remaining = parser.flush()
                if remaining:
                    text_buffer += remaining
                    yield StreamEvent(type="text", content=remaining)
                    if self.on_text:
                        self.on_text(remaining)

                # If no tools detected, we're done
                if not tools_to_execute:
                    yield StreamEvent(type="done")
                    return

                # Execute tools and continue loop
                for tool in tools_to_execute:
                    tool_id = str(uuid4())

                    yield StreamEvent(
                        type="tool_start",
                        tool_id=tool_id,
                        tool_name=tool.name,
                        tool_params=tool.params,
                    )
                    if self.on_tool_start:
                        self.on_tool_start(tool_id, tool.name, tool.params)

                    # Execute tool
                    tool_callings = [(tool_id, tool.name, tool.params)]
                    tool_results = self.tool_library(
                        tool_callings=tool_callings,
                        model_state=model_state,
                        vars=vars,
                    )

                    result = tool_results.tool_calls[0]
                    tool_output = result.result or result.error

                    yield StreamEvent(
                        type="tool_result",
                        tool_id=tool_id,
                        tool_name=tool.name,
                        tool_result=tool_output,
                    )
                    if self.on_tool_result:
                        self.on_tool_result(tool_id, tool_output)

                    if tool_results.return_directly:
                        yield StreamEvent(type="done")
                        return

                # Update model state with assistant message and tool results
                if text_buffer:
                    model_state.append(ChatBlock.assist(text_buffer))

                # Add tool results as user message for next iteration
                tool_results_text = self._format_tool_results(tools_to_execute, vars)
                model_state.append(ChatBlock.user(tool_results_text))

                # Reset for next iteration
                parser.reset()
                tools_to_execute = []
                text_buffer = ""

            # Max iterations reached
            yield StreamEvent(
                type="error",
                error=f"Max iterations ({self.max_iterations}) reached",
            )

        except Exception as e:
            error_msg = str(e)
            yield StreamEvent(type="error", error=error_msg)
            if self.on_error:
                self.on_error(error_msg)

    async def astream(
        self,
        message: Optional[Union[str, Mapping[str, Any], Message]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Async version of stream().

        Yields StreamEvent objects as the model generates content and
        tools are executed asynchronously.

        Args:
            message: Input message (string, dict, or Message object).
            **kwargs: Additional arguments passed to _prepare_task.

        Yields:
            StreamEvent objects with types: text, reasoning, tool_start,
            tool_result, done, or error.
        """
        try:
            inputs = await self._aprepare_task(message, **kwargs)
            model_state = inputs["model_state"]
            vars = inputs["vars"]
            model_preference = inputs.get("model_preference")

            parser = StreamingXMLParser()
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1

                # Prepare model execution params
                exec_params = self._prepare_model_execution(
                    model_state=model_state,
                    prefilling=self.prefilling,
                    model_preference=model_preference,
                    vars=vars,
                )

                # Get system prompt and messages
                system_prompt = exec_params.get("system_prompt")
                messages = exec_params.get("messages", [])

                # Add tool schemas to system prompt for XML-based calling
                tool_schemas = self.tool_library.get_tool_json_schemas()
                if tool_schemas:
                    tools_prompt = self._build_tools_system_prompt(tool_schemas)
                    if system_prompt:
                        system_prompt = tools_prompt + "\n\n" + system_prompt
                    else:
                        system_prompt = tools_prompt

                # Stream from model
                text_buffer = ""
                tools_to_execute: List[ParsedToolCall] = []

                async for chunk in self.lm.model.astream(
                    messages=messages,
                    system_prompt=system_prompt,
                    prefilling=exec_params.get("prefilling"),
                ):
                    if chunk.type == "text":
                        # Feed to XML parser
                        text, tools = parser.feed(chunk.content or "")

                        # Yield text immediately
                        if text:
                            text_buffer += text
                            yield StreamEvent(type="text", content=text)
                            if self.on_text:
                                self.on_text(text)

                        # Collect detected tools
                        if tools:
                            tools_to_execute.extend(tools)

                    elif chunk.type == "reasoning":
                        yield StreamEvent(type="reasoning", content=chunk.content)
                        if self.on_reasoning:
                            self.on_reasoning(chunk.content or "")

                # Flush any remaining buffer
                remaining = parser.flush()
                if remaining:
                    text_buffer += remaining
                    yield StreamEvent(type="text", content=remaining)
                    if self.on_text:
                        self.on_text(remaining)

                # If no tools detected, we're done
                if not tools_to_execute:
                    yield StreamEvent(type="done")
                    return

                # Execute tools and continue loop
                for tool in tools_to_execute:
                    tool_id = str(uuid4())

                    yield StreamEvent(
                        type="tool_start",
                        tool_id=tool_id,
                        tool_name=tool.name,
                        tool_params=tool.params,
                    )
                    if self.on_tool_start:
                        self.on_tool_start(tool_id, tool.name, tool.params)

                    # Execute tool
                    tool_callings = [(tool_id, tool.name, tool.params)]
                    tool_results = await self.tool_library.acall(
                        tool_callings=tool_callings,
                        model_state=model_state,
                        vars=vars,
                    )

                    result = tool_results.tool_calls[0]
                    tool_output = result.result or result.error

                    yield StreamEvent(
                        type="tool_result",
                        tool_id=tool_id,
                        tool_name=tool.name,
                        tool_result=tool_output,
                    )
                    if self.on_tool_result:
                        self.on_tool_result(tool_id, tool_output)

                    if tool_results.return_directly:
                        yield StreamEvent(type="done")
                        return

                # Update model state with assistant message and tool results
                if text_buffer:
                    model_state.append(ChatBlock.assist(text_buffer))

                # Add tool results as user message for next iteration
                tool_results_text = self._format_tool_results(tools_to_execute, vars)
                model_state.append(ChatBlock.user(tool_results_text))

                # Reset for next iteration
                parser.reset()
                tools_to_execute = []
                text_buffer = ""

            # Max iterations reached
            yield StreamEvent(
                type="error",
                error=f"Max iterations ({self.max_iterations}) reached",
            )

        except Exception as e:
            error_msg = str(e)
            yield StreamEvent(type="error", error=error_msg)
            if self.on_error:
                self.on_error(error_msg)

    def _build_tools_system_prompt(self, tool_schemas: List[Dict[str, Any]]) -> str:
        """Build system prompt section for XML-based tool calling.

        Args:
            tool_schemas: List of tool JSON schemas.

        Returns:
            System prompt section describing available tools and XML format.
        """
        tools_desc = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})

            param_lines = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                param_lines.append(f"    - {pname} ({ptype}): {pdesc}")

            tool_desc = f"- {name}: {desc}"
            if param_lines:
                tool_desc += "\n" + "\n".join(param_lines)
            tools_desc.append(tool_desc)

        tools_text = "\n".join(tools_desc)

        # Build the prompt with XML format instructions
        # Using string concatenation to avoid XML interpretation
        open_tag = "<"
        close_tag = ">"
        prompt = f"""You have access to the following tools:

{tools_text}

To use a tool, output XML in the following format:

{open_tag}function_calls{close_tag}
{open_tag}invoke name="tool_name"{close_tag}
{open_tag}parameter name="param1"{close_tag}value1{open_tag}/parameter{close_tag}
{open_tag}parameter name="param2"{close_tag}value2{open_tag}/parameter{close_tag}
{open_tag}/invoke{close_tag}
{open_tag}/function_calls{close_tag}

You can call multiple tools by including multiple invoke tags.
After using tools, you will receive the results and can continue your response."""

        return prompt

    def _format_tool_results(
        self,
        tools: List[ParsedToolCall],
        vars: Mapping[str, Any],
    ) -> str:
        """Format tool results for the next model iteration.

        Args:
            tools: List of executed tools with results.
            vars: Variables from the context.

        Returns:
            Formatted string with tool results.
        """
        results = []
        for tool in tools:
            # Get the result from the tool (stored during execution)
            result_text = getattr(tool, "_result", "No result")
            results.append(f"Tool '{tool.name}' returned: {result_text}")

        return "\n".join(results)
