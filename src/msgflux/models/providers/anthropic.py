from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union

import msgspec

try:
    import httpx
    from anthropic import Anthropic, AsyncAnthropic
    from opentelemetry import trace
    from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor

    if not hasattr(Anthropic, "_otel_instrumented"):
        AnthropicInstrumentor().instrument()
        Anthropic._otel_instrumented = True
except ImportError:
    httpx = None
    Anthropic = None
    AsyncAnthropic = None
    trace = None

from msgflux.dotdict import dotdict
from msgflux.dsl.typed_parsers import typed_parser_registry
from msgflux.exceptions import TypedParserNotFoundError
from msgflux.logger import logger
from msgflux.models.base import BaseModel
from msgflux.models.cache import ResponseCache, generate_cache_key
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.types import ChatCompletionModel
from msgflux.nn import functional as F
from msgflux.utils.chat import ChatBlock
from msgflux.utils.console import cprint
from msgflux.utils.msgspec import struct_to_dict
from msgflux.utils.tenacity import model_retry


class _BaseAnthropic(BaseModel):
    provider: str = "anthropic"

    def _initialize(self):
        """Initialize the Anthropic client with API key."""
        if Anthropic is None or AsyncAnthropic is None:
            raise ImportError(
                "`anthropic` client is not available. "
                "Install with `pip install msgflux[anthropic]`."
            )
        self.current_key_index = 0
        max_retries = int(getenv("ANTHROPIC_MAX_RETRIES", "2"))
        timeout = getenv("ANTHROPIC_TIMEOUT", None)

        self.client = Anthropic(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=float(timeout) if timeout else httpx.Timeout(60.0),
            max_retries=max_retries,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
        self.aclient = AsyncAnthropic(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=float(timeout) if timeout else httpx.Timeout(60.0),
            max_retries=int(max_retries),
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )

    def _get_base_url(self):
        return None

    def _get_api_key(self):
        """Load API key from environment variable."""
        key = getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "The Anthropic API key is not available. Please set `ANTHROPIC_API_KEY`"
            )
        return key


@register_model
class AnthropicChatCompletion(_BaseAnthropic, ChatCompletionModel):
    """Anthropic Chat Completion."""

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: Optional[int] = 4096,
        thinking: Optional[Dict[str, Any]] = None,
        return_thinking: Optional[bool] = False,
        thinking_in_tool_call: Optional[bool] = True,
        validate_typed_parser_output: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        base_url: Optional[str] = None,
        context_length: Optional[int] = None,
        reasoning_max_tokens: Optional[int] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
    ):
        """Args:
        model_id:
            Model ID in provider.
        max_tokens:
            The maximum number of tokens to generate before stopping.
        thinking:
            Extended thinking configuration.
            Example: {"type": "enabled", "budget_tokens": 10000}.
        return_thinking:
            If the model returns the `thinking` field it will be added
            along with the response.
        thinking_in_tool_call:
            If True, maintains the thinking for using the tool call.
        validate_typed_parser_output:
            If True, use the generation_schema to validate typed parser output.
        temperature:
            Amount of randomness injected into the response. Ranges from 0.0 to 1.0.
        top_p:
            Use nucleus sampling. Recommended for advanced use cases only.
        top_k:
            Only sample from the top K options for each subsequent token.
        stop_sequences:
            Custom text sequences that will cause the model to stop generating.
        verbose:
            If True, prints the model output to the console before it is transformed
            into typed structured output.
        base_url:
            URL to model provider.
        context_length:
            The maximum context length supported by the model.
        reasoning_max_tokens:
            Maximum number of tokens for reasoning/thinking.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        """
        super().__init__()
        self.model_id = model_id
        self.context_length = context_length
        self.reasoning_max_tokens = reasoning_max_tokens
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {"max_tokens": max_tokens}
        if temperature is not None:
            sampling_run_params["temperature"] = temperature
        if top_p is not None:
            sampling_run_params["top_p"] = top_p
        if top_k is not None:
            sampling_run_params["top_k"] = top_k
        if stop_sequences:
            sampling_run_params["stop_sequences"] = stop_sequences
        if thinking:
            sampling_run_params["thinking"] = thinking
        self.sampling_run_params = sampling_run_params
        self.thinking_in_tool_call = thinking_in_tool_call
        self.validate_typed_parser_output = validate_typed_parser_output
        self.return_thinking = return_thinking
        self.verbose = verbose
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._response_cache = ResponseCache(maxsize=cache_size) if enable_cache else None
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.messages.create(**kwargs, **self.sampling_run_params)

        # Add provider name to the current span for accurate tracking
        if trace:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("gen_ai.provider.name", self.provider)

        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.messages.create(
            **kwargs, **self.sampling_run_params
        )

        # Add provider name to the current span for accurate tracking
        if trace:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("gen_ai.provider.name", self.provider)

        return model_output

    def _process_model_output(  # noqa: C901
        self, model_output, typed_parser=None, generation_schema=None
    ):
        """Shared logic to process model output for both sync and async."""
        response = ModelResponse()
        metadata = dotdict()

        # Convert usage to dict format
        usage_dict = {
            "input_tokens": model_output.usage.input_tokens,
            "output_tokens": model_output.usage.output_tokens,
        }
        if hasattr(model_output.usage, "cache_creation_input_tokens"):
            usage_dict["cache_creation_input_tokens"] = (
                model_output.usage.cache_creation_input_tokens
            )
        if hasattr(model_output.usage, "cache_read_input_tokens"):
            usage_dict["cache_read_input_tokens"] = (
                model_output.usage.cache_read_input_tokens
            )

        metadata.update({"usage": usage_dict})

        # Extract thinking content if present
        thinking_content = None
        thinking_tool_call = None

        for block in model_output.content:
            if block.type == "thinking":
                thinking_content = block.thinking
                if self.thinking_in_tool_call:
                    thinking_tool_call = thinking_content
                break

        prefix_response_type = ""
        if self.return_thinking and thinking_content:
            prefix_response_type = "thinking_"

        # Process content blocks
        text_content = []
        tool_calls = []

        for block in model_output.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

        if tool_calls:
            aggregator = ToolCallAggregator(thinking_tool_call)
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(tool_calls):
                tool_id = tool_call.id
                name = tool_call.name
                arguments = msgspec.json.encode(tool_call.input).decode()
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif text_content:
            combined_text = "".join(text_content)
            if (typed_parser or generation_schema) and self.verbose:
                repr_text = f"[{self.model_id}][raw_response] {combined_text}"
                cprint(repr_text, lc="r", ls="b")

            if typed_parser is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.parse(combined_text))
                # Type validation
                if generation_schema and self.validate_typed_parser_output:
                    encoded_response_content = msgspec.json.encode(response_content)
                    msgspec.json.decode(
                        encoded_response_content, type=generation_schema
                    )
            elif generation_schema is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                struct = msgspec.json.decode(combined_text, type=generation_schema)
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type(f"{prefix_response_type}text_generation")
                if thinking_content is not None:
                    response_content = dotdict({"answer": combined_text})
                else:
                    response_content = combined_text

        if thinking_content is not None and self.return_thinking:
            response_content.think = thinking_content

        response.add(response_content)
        response.set_metadata(metadata)
        return response

    def _generate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                logger.debug(
                    "Cache hit for anthropic chat completion",
                    extra={"cache_key": cache_key, "model_id": self.model_id},
                )
                return cached_response

        model_output = self._execute_model(**kwargs)
        response = self._process_model_output(model_output, typed_parser, generation_schema)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            self._response_cache.set(cache_key, response)
            logger.debug(
                "Cached anthropic chat completion response",
                extra={"cache_key": cache_key, "model_id": self.model_id},
            )

        return response

    async def _agenerate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                logger.debug(
                    "Cache hit for anthropic chat completion",
                    extra={"cache_key": cache_key, "model_id": self.model_id},
                )
                return cached_response

        model_output = await self._aexecute_model(**kwargs)
        response = self._process_model_output(model_output, typed_parser, generation_schema)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            self._response_cache.set(cache_key, response)
            logger.debug(
                "Cached anthropic chat completion response",
                extra={"cache_key": cache_key, "model_id": self.model_id},
            )

        return response

    def _stream_generate(self, **kwargs: Mapping[str, Any]) -> ModelStreamResponse:  # noqa: C901
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")

        thinking_tool_call = ""

        with self.client.messages.stream(
            **kwargs, **self.sampling_run_params
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        if (
                            stream_response.response_type is None
                            and self.return_thinking
                        ):
                            stream_response.set_response_type(
                                "thinking_text_generation"
                            )
                            stream_response.first_chunk_event.set()
                    elif event.content_block.type == "text":
                        if stream_response.response_type is None:
                            stream_response.set_response_type("text_generation")
                            stream_response.first_chunk_event.set()
                    elif event.content_block.type == "tool_use":
                        if stream_response.response_type is None:
                            stream_response.set_response_type("tool_call")

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        thinking_chunk = event.delta.thinking
                        if self.thinking_in_tool_call:
                            thinking_tool_call += thinking_chunk
                        if self.return_thinking:
                            stream_response.add(thinking_chunk)
                    elif event.delta.type == "text_delta":
                        stream_response.add(event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        # For tool use, we need to aggregate the input
                        pass

                elif event.type == "content_block_stop":
                    if (
                        hasattr(event, "content_block")
                        and event.content_block.type == "tool_use"
                    ):
                        tool_id = event.content_block.id
                        name = event.content_block.name
                        arguments = msgspec.json.encode(
                            event.content_block.input
                        ).decode()
                        aggregator.process(event.index, tool_id, name, arguments)

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage_dict = {}
                        if hasattr(event.usage, "output_tokens"):
                            usage_dict["output_tokens"] = event.usage.output_tokens
                        metadata.update({"usage": usage_dict})

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        usage_dict = {
                            "input_tokens": event.message.usage.input_tokens,
                        }
                        if hasattr(event.message.usage, "cache_creation_input_tokens"):
                            usage_dict["cache_creation_input_tokens"] = (
                                event.message.usage.cache_creation_input_tokens
                            )
                        if hasattr(event.message.usage, "cache_read_input_tokens"):
                            usage_dict["cache_read_input_tokens"] = (
                                event.message.usage.cache_read_input_tokens
                            )
                        metadata.update({"usage": usage_dict})

        if aggregator.tool_calls:
            if thinking_tool_call:
                aggregator.thinking = thinking_tool_call
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    async def _astream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> ModelStreamResponse:
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")

        thinking_tool_call = ""

        async with self.aclient.messages.stream(
            **kwargs, **self.sampling_run_params
        ) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        if (
                            stream_response.response_type is None
                            and self.return_thinking
                        ):
                            stream_response.set_response_type(
                                "thinking_text_generation"
                            )
                            stream_response.first_chunk_event.set()
                    elif event.content_block.type == "text":
                        if stream_response.response_type is None:
                            stream_response.set_response_type("text_generation")
                            stream_response.first_chunk_event.set()
                    elif event.content_block.type == "tool_use":
                        if stream_response.response_type is None:
                            stream_response.set_response_type("tool_call")

                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta":
                        thinking_chunk = event.delta.thinking
                        if self.thinking_in_tool_call:
                            thinking_tool_call += thinking_chunk
                        if self.return_thinking:
                            stream_response.add(thinking_chunk)
                    elif event.delta.type == "text_delta":
                        stream_response.add(event.delta.text)
                    elif event.delta.type == "input_json_delta":
                        # For tool use, we need to aggregate the input
                        pass

                elif event.type == "content_block_stop":
                    if (
                        hasattr(event, "content_block")
                        and event.content_block.type == "tool_use"
                    ):
                        tool_id = event.content_block.id
                        name = event.content_block.name
                        arguments = msgspec.json.encode(
                            event.content_block.input
                        ).decode()
                        aggregator.process(event.index, tool_id, name, arguments)

                elif event.type == "message_delta":
                    if hasattr(event, "usage"):
                        usage_dict = {}
                        if hasattr(event.usage, "output_tokens"):
                            usage_dict["output_tokens"] = event.usage.output_tokens
                        metadata.update({"usage": usage_dict})

                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        usage_dict = {
                            "input_tokens": event.message.usage.input_tokens,
                        }
                        if hasattr(event.message.usage, "cache_creation_input_tokens"):
                            usage_dict["cache_creation_input_tokens"] = (
                                event.message.usage.cache_creation_input_tokens
                            )
                        if hasattr(event.message.usage, "cache_read_input_tokens"):
                            usage_dict["cache_read_input_tokens"] = (
                                event.message.usage.cache_read_input_tokens
                            )
                        metadata.update({"usage": usage_dict})

        if aggregator.tool_calls:
            if thinking_tool_call:
                aggregator.thinking = thinking_tool_call
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Convert messages to Anthropic format and extract system prompt."""
        system_prompt = None
        converted_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                # Convert message content to Anthropic format
                role = msg.get("role")
                content = msg.get("content")

                # Handle multi-modal content
                if isinstance(content, list):
                    converted_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                converted_content.append(
                                    {"type": "text", "text": item.get("text")}
                                )
                            elif item.get("type") == "image":
                                # Anthropic uses image format
                                converted_content.append(
                                    {"type": "image", "source": item.get("source")}
                                )
                            elif item.get("type") == "tool_result":
                                converted_content.append(item)
                    converted_messages.append(
                        {"role": role, "content": converted_content}
                    )
                else:
                    converted_messages.append({"role": role, "content": content})

        return system_prompt, converted_messages

    @model_retry
    def __call__(  # noqa: C901
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use.
                You can force specific behavior with the tool_choice parameter.
                    1. auto:
                        (Default) Call zero, one, or multiple functions.
                    2. any:
                        Call one or more functions.
                    3. Forced Tool:
                        Call exactly one specific tool.
                        Example: {"type": "tool", "name": "get_weather"}.
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_parser` and `stream=True`.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]

        # Extract system prompt from messages if present
        extracted_system, converted_messages = self._convert_messages(messages)
        if system_prompt is None and extracted_system:
            system_prompt = extracted_system

        generation_params = {
            "messages": converted_messages,
            "model": self.model_id,
        }

        if system_prompt:
            generation_params["system"] = system_prompt

        if tool_schemas:
            generation_params["tools"] = tool_schemas

        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    generation_params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "any":
                    generation_params["tool_choice"] = {"type": "any"}
                elif tool_choice not in ["auto", "any", "none"]:
                    generation_params["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice,
                    }
            else:
                generation_params["tool_choice"] = tool_choice

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse()
            F.background_task(
                self._stream_generate,
                **generation_params,
                stream_response=stream_response,
            )
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            if typed_parser and typed_parser not in typed_parser_registry:
                available = ", ".join(typed_parser_registry.keys())
                raise TypedParserNotFoundError(
                    f"Typed parser `{typed_parser}` not found. "
                    f"Available parsers: {available}"
                )
            response = self._generate(
                **generation_params,
                typed_parser=typed_parser,
                generation_schema=generation_schema,
            )
            return response

    @model_retry
    async def acall(  # noqa: C901
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        system_prompt: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use.
                You can force specific behavior with the tool_choice parameter.
                    1. auto:
                        (Default) Call zero, one, or multiple functions.
                    2. any:
                        Call one or more functions.
                    3. Forced Tool:
                        Call exactly one specific tool.
                        Example: {"type": "tool", "name": "get_weather"}.
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_parser` and `stream=True`.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]

        # Extract system prompt from messages if present
        extracted_system, converted_messages = self._convert_messages(messages)
        if system_prompt is None and extracted_system:
            system_prompt = extracted_system

        generation_params = {
            "messages": converted_messages,
            "model": self.model_id,
        }

        if system_prompt:
            generation_params["system"] = system_prompt

        if tool_schemas:
            generation_params["tools"] = tool_schemas

        if tool_choice:
            if isinstance(tool_choice, str):
                if tool_choice == "auto":
                    generation_params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "any":
                    generation_params["tool_choice"] = {"type": "any"}
                elif tool_choice not in ["auto", "any", "none"]:
                    generation_params["tool_choice"] = {
                        "type": "tool",
                        "name": tool_choice,
                    }
            else:
                generation_params["tool_choice"] = tool_choice

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse()
            await F.abackground_task(
                self._astream_generate,
                **generation_params,
                stream_response=stream_response,
            )
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            if typed_parser and typed_parser not in typed_parser_registry:
                available = ", ".join(typed_parser_registry.keys())
                raise TypedParserNotFoundError(
                    f"Typed parser `{typed_parser}` not found. "
                    f"Available parsers: {available}"
                )
            response = await self._agenerate(
                **generation_params,
                typed_parser=typed_parser,
                generation_schema=generation_schema,
            )
            return response
