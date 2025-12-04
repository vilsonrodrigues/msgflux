from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union

import msgspec

try:
    from mistralai.async_client import MistralAsyncClient
    from mistralai.client import MistralClient
    from opentelemetry import trace
except ImportError:
    MistralClient = None
    MistralAsyncClient = None
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
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.console import cprint
from msgflux.utils.msgspec import struct_to_dict
from msgflux.utils.tenacity import model_retry


class _BaseMistral(BaseModel):
    provider: str = "mistral"

    def _initialize(self):
        """Initialize the Mistral client with API key."""
        if MistralClient is None:
            raise ImportError(
                "`mistralai` client is not available. "
                "Install with `pip install msgflux[mistral]`."
            )
        self.client = MistralClient(api_key=self._get_api_key())
        self.aclient = MistralAsyncClient(api_key=self._get_api_key())

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("MISTRAL_API_KEY")
        if not key:
            raise ValueError(
                "The Mistral API key is not available. Please set `MISTRAL_API_KEY`"
            )
        return key


@register_model
class MistralChatCompletion(_BaseMistral, ChatCompletionModel):
    """Mistral Chat Completion."""

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        random_seed: Optional[int] = None,
        safe_prompt: bool = False,
        validate_typed_parser_output: Optional[bool] = False,
        verbose: Optional[bool] = False,
        context_length: Optional[int] = None,
        reasoning_max_tokens: Optional[int] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
    ):
        """Args:
        model_id:
            Model ID in provider.
        max_tokens:
            An upper bound for the number of tokens that can be
            generated for a completion.
        temperature:
            What sampling temperature to use, between 0 and 1.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and
            deterministic.
        top_p:
            An alternative to sampling with temperature, called nucleus
            sampling, where the model considers the results of the tokens
            with top_p probability mass.
        random_seed:
            The seed to use for random sampling.
        safe_prompt:
            Whether to inject a safety prompt before the user's prompt.
        validate_typed_parser_output:
            If True, use the generation_schema to validate typed parser output.
        verbose:
            If True, Prints the model output to the console before it is transformed
            into typed structured output.
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
        self.sampling_params = {}
        if max_tokens:
            self.sampling_params["max_tokens"] = max_tokens
        if temperature:
            self.sampling_params["temperature"] = temperature
        if top_p:
            self.sampling_params["top_p"] = top_p
        if random_seed:
            self.sampling_params["random_seed"] = random_seed
        self.safe_prompt = safe_prompt
        self.validate_typed_parser_output = validate_typed_parser_output
        self.verbose = verbose
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._response_cache = ResponseCache(maxsize=cache_size) if enable_cache else None
        self._initialize()

    def _execute_model(self, **kwargs):
        model_output = self.client.chat(**kwargs, **self.sampling_params)

        # Add provider name to the current span for accurate tracking
        if trace:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("gen_ai.provider.name", self.provider)

        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.chat(**kwargs, **self.sampling_params)

        # Add provider name to the current span for accurate tracking
        if trace:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("gen_ai.provider.name", self.provider)

        return model_output

    def _process_model_output(self, model_output, typed_parser=None, generation_schema=None):
        """Shared logic to process model output for both sync and async."""
        response = ModelResponse()
        metadata = dotdict()

        metadata.update({"usage": model_output.usage.to_dict()})

        choice = model_output.choices[0]

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator()
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                tool_id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif choice.message.content:
            if (typed_parser or generation_schema) and self.verbose:
                repr = f"[{self.model_id}][raw_response] {choice.message.content}"
                cprint(repr, lc="r", ls="b")
            if typed_parser is not None:
                response.set_response_type("structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.parse(choice.message.content))
                if generation_schema and self.validate_typed_parser_output:
                    encoded_response_content = msgspec.json.encode(response_content)
                    msgspec.json.decode(
                        encoded_response_content, type=generation_schema
                    )
            elif generation_schema is not None:
                response.set_response_type("structured")
                struct = msgspec.json.decode(
                    choice.message.content, type=generation_schema
                )
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type("text_generation")
                response_content = choice.message.content

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
                    "Cache hit for mistral chat completion",
                    extra={"cache_key": cache_key, "model_id": self.model_id},
                )
                return cached_response

        if generation_schema is not None and typed_parser is None:
            kwargs["response_format"] = {"type": "json_object"}

        model_output = self._execute_model(**kwargs)
        response = self._process_model_output(model_output, typed_parser, generation_schema)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            self._response_cache.set(cache_key, response)
            logger.debug(
                "Cached mistral chat completion response",
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
                    "Cache hit for mistral chat completion",
                    extra={"cache_key": cache_key, "model_id": self.model_id},
                )
                return cached_response

        if generation_schema is not None and typed_parser is None:
            kwargs["response_format"] = {"type": "json_object"}

        model_output = await self._aexecute_model(**kwargs)
        response = self._process_model_output(model_output, typed_parser, generation_schema)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(
                **kwargs, typed_parser=typed_parser, generation_schema=generation_schema
            )
            self._response_cache.set(cache_key, response)
            logger.debug(
                "Cached mistral chat completion response",
                extra={"cache_key": cache_key, "model_id": self.model_id},
            )

        return response

    def _stream_generate(self, **kwargs: Mapping[str, Any]) -> ModelStreamResponse:
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")
        model_output = self.client.chat_stream(**kwargs, **self.sampling_params)

        for chunk in model_output:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(delta.content)
                elif delta.tool_calls:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("tool_call")
                    tool_call = delta.tool_calls[0]
                    call_index = tool_call.index
                    tool_id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    aggregator.process(call_index, tool_id, name, arguments)

            if chunk.usage:
                metadata.update(chunk.usage.to_dict())

        if aggregator.tool_calls:
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    async def _astream_generate(self, **kwargs: Mapping[str, Any]) -> ModelStreamResponse:
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")
        model_output = self.aclient.chat_stream(**kwargs, **self.sampling_params)

        async for chunk in model_output:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(delta.content)
                elif delta.tool_calls:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("tool_call")
                    tool_call = delta.tool_calls[0]
                    call_index = tool_call.index
                    tool_id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    aggregator.process(call_index, tool_id, name, arguments)

            if chunk.usage:
                metadata.update(chunk.usage.to_dict())

        if aggregator.tool_calls:
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    @model_retry
    def __call__(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use.
                You can force specific behavior with the tool_choice parameter.
            typed_parser:
                Converts the model raw output into a typed-dict.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]

        generation_params = {
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": tool_choice,
            "model": self.model_id,
            "safe_prompt": self.safe_prompt,
        }

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
    async def acall(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        *,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
            messages:
                Conversation history. Can be simple string or list of messages.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use.
                You can force specific behavior with the tool_choice parameter.
            typed_parser:
                Converts the model raw output into a typed-dict.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]

        generation_params = {
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": tool_choice,
            "model": self.model_id,
            "safe_prompt": self.safe_prompt,
        }

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
