import base64
import inspect
import tempfile
from contextlib import asynccontextmanager, contextmanager
from os import getenv
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import msgspec

try:
    import httpx
    import openai
    from openai import AsyncOpenAI, OpenAI
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    if not getattr(openai, "_otel_instrumented", False):
        OpenAIInstrumentor().instrument()
        openai._otel_instrumented = True
except ImportError:
    httpx = None
    openai = None
    OpenAI = None
    AsyncOpenAI = None

import msgflux.nn.functional as F
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.dsl.typed_parsers import typed_parser_registry
from msgflux.exceptions import TypedParserNotFoundError
from msgflux.models.base import BaseModel
from msgflux.models.cache import ResponseCache, generate_cache_key
from msgflux.models.profiles import get_model_profile
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.types import (
    ChatCompletionModel,
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextToImageModel,
    TextToSpeechModel,
)
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.console import cprint
from msgflux.utils.encode import encode_data_to_bytes
from msgflux.utils.msgspec import struct_to_dict
from msgflux.utils.tenacity import apply_retry, default_model_retry


class _BaseOpenAI(BaseModel):
    provider: str = "openai"

    def _initialize(self):
        """Initialize the OpenAI client with empty API key."""
        if openai is None or OpenAI is None:
            raise ImportError(
                "`openai` client is not available. "
                "Install with `pip install msgflux[openai]`."
            )
        self.current_key_index = 0
        max_retries = getenv("OPENAI_MAX_RETRIES", openai.DEFAULT_MAX_RETRIES)
        timeout = getenv("OPENAI_TIMEOUT", None)
        self.client = OpenAI(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
        self.aclient = AsyncOpenAI(
            **self.sampling_params,
            api_key=self._get_api_key(),
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )
        # Initialize response cache
        cache_size = getattr(self, "cache_size", 128)
        enable_cache = getattr(self, "enable_cache", None)
        self._response_cache = (
            ResponseCache(maxsize=cache_size) if enable_cache else None
        )

        # Apply retry
        retry_config = getattr(self, "retry", None)
        self.__call__ = apply_retry(
            self.__call__, retry_config, default=default_model_retry
        )
        self.acall = apply_retry(self.acall, retry_config, default=default_model_retry)

    def _get_base_url(self):
        return None

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "The OpenAI key is not available. Please set `OPENAI_API_KEY`"
            )
        return key

    @property
    def profile(self):
        """Get model profile from registry.

        Returns:
            ModelProfile if found, None otherwise
        """
        return get_model_profile(self.model_id, provider_id=self.provider)


@register_model
class OpenAIChatCompletion(_BaseOpenAI, ChatCompletionModel):
    """OpenAI Chat Completion."""

    api_mode: str = "completion"

    def __init__(  # noqa: C901
        self,
        model_id: str,
        *,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        return_reasoning: Optional[bool] = True,
        reasoning_in_tool_call: Optional[bool] = True,
        validate_typed_parser_output: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        parallel_tool_calls: Optional[bool] = True,
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, str]] = None,
        verbosity: Optional[str] = None,
        web_search_options: Optional[Dict[str, Any]] = None,
        provider_tools: Optional[List[Dict[str, Any]]] = None,
        api_mode: Optional[str] = None,
        verbose: Optional[bool] = False,
        base_url: Optional[str] = None,
        context_length: Optional[int] = None,
        reasoning_max_tokens: Optional[int] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        max_tokens:
            An upper bound for the number of tokens that can be
            generated for a completion, including visible output
            tokens and reasoning tokens.
        reasoning_effort:
            Constrains effort on reasoning for reasoning models.
            Currently supported values are low, medium, and high.
            Reducing reasoning effort can result in faster responses
            and fewer tokens used on reasoning in a response.
            Can be: "minimal", "low", "medium" or "high".
        enable_thinking:
            If True, enable the model reasoning.
        return_reasoning:
            If the model returns the `reasoning` field it will be added
            along with the response.
        reasoning_in_tool_call:
            If True, maintains the reasoning for using the tool call.
        validate_typed_parser_output:
            If True, use the generation_schema to validate typed parser output.
        temperature:
            What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and
            deterministic.
        stop:
            Up to 4 sequences where the API will stop generating further
            tokens. The returned text will not contain the stop sequence.
        top_p:
            An alternative to sampling with temperature, called nucleus
            sampling, where the model considers the results of the tokens
            with top_p probability mass. So 0.1 means only the tokens
            comprising the top 10% probability mass are considered.
        parallel_tool_calls:
            If True, enable parallel tool calls.
        modalities:
            Types of output you would like the model to generate.
            Can be: ["text"], ["audio"] or ["text", "audio"].
        audio:
            Audio configurations. Define voice and output format.
        verbosity:
            Constrains the verbosity of the model's response. Lower
            values will result in more concise responses, while higher
            values will result in more verbose responses. Currently
            supported values are low, medium, and high.
        web_search_options:
            This tool searches the web for relevant results to use in a response.
            OpenAI and OpenRouter only.
        provider_tools:
            Provider-native tools configuration (e.g. `web_search`, `file_search`,
            `mcp`, `image_generation`) for `/responses`.
        api_mode:
            Select API mode. `completion` uses `/chat/completions`,
            `response` uses `/responses`.
        verbose:
            If True, Prints the model output to the console before it is transformed
            into typed structured output.
        base_url:
            URL to model provider.
        context_length:
            The maximum context length supported by the model.
        reasoning_max_tokens:
            Maximum number of tokens for reasoning/thinking.
        enable_cache:
            If True, enable response caching to avoid redundant API calls.
        cache_size:
            Maximum number of cached responses (default: 128).
        """
        super().__init__()
        self.model_id = model_id
        self.context_length = context_length
        self.reasoning_max_tokens = reasoning_max_tokens
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {"max_tokens": max_tokens}
        if temperature:
            sampling_run_params["temperature"] = temperature
        if top_p:
            sampling_run_params["top_p"] = top_p
        if stop:
            sampling_run_params["stop"] = stop
        if verbosity:
            sampling_run_params["verbosity"] = verbosity
        if modalities:
            sampling_run_params["modalities"] = modalities
        if web_search_options:
            sampling_run_params["web_search_options"] = web_search_options
        if audio:
            sampling_run_params["audio"] = audio
        if reasoning_effort:
            sampling_run_params["reasoning_effort"] = reasoning_effort
        self.sampling_run_params = sampling_run_params
        self.enable_thinking = enable_thinking
        self.parallel_tool_calls = parallel_tool_calls
        self.reasoning_in_tool_call = reasoning_in_tool_call
        self.validate_typed_parser_output = validate_typed_parser_output
        self.return_reasoning = return_reasoning
        self.verbose = verbose
        self.retry = retry
        resolved_api_mode = api_mode
        if resolved_api_mode is None:
            resolved_api_mode = getattr(self.__class__, "api_mode", "completion")
        self.api_mode = resolved_api_mode

        if self.api_mode not in {"completion", "response"}:
            raise ValueError(
                "`api_mode` must be `completion` or `response`, "
                f"given `{self.api_mode}`"
            )
        if self.api_mode == "response" and self.provider != "openai":
            raise ValueError(
                f"`api_mode=response` is not supported for provider `{self.provider}`"
            )
        self.provider_tools = provider_tools
        if self.provider_tools and self.api_mode != "response":
            raise ValueError(
                "`provider_tools` requires `api_mode='response'` for OpenAI."
            )
        self._initialize()
        self._get_api_key()

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params.pop("provider_tools", None)
        if self.provider == "openai":
            max_tokens = params.pop("max_tokens", None)
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
        return params

    def _merge_tools(self, *tool_groups: Any):
        merged_tools = []
        for tools in tool_groups:
            if tools is None:
                continue
            if isinstance(tools, list):
                merged_tools.extend(tools)
            else:
                merged_tools.append(tools)
        if not merged_tools:
            return None
        return merged_tools

    def _adapt_responses_tools(self, tools: Any):
        if not isinstance(tools, list):
            return tools

        adapted_tools = []
        for tool in tools:
            if (
                isinstance(tool, Mapping)
                and tool.get("type") == "function"
                and isinstance(tool.get("function"), Mapping)
            ):
                function_schema = tool.get("function")
                adapted_tool = {"type": "function"}
                for key in (
                    "name",
                    "description",
                    "parameters",
                    "strict",
                ):
                    value = function_schema.get(key)
                    if value is not None:
                        adapted_tool[key] = value
                adapted_tools.append(adapted_tool)
            else:
                adapted_tools.append(tool)
        return adapted_tools

    def _adapt_responses_tool_choice(self, tool_choice: Any):
        if not isinstance(tool_choice, Mapping):
            return tool_choice

        if tool_choice.get("type") != "function":
            return tool_choice

        if isinstance(tool_choice.get("function"), Mapping):
            name = tool_choice["function"].get("name")
            if name:
                return {"type": "function", "name": name}

        if tool_choice.get("name"):
            return {"type": "function", "name": tool_choice["name"]}

        return tool_choice

    def _adapt_response_format_to_text_format(self, response_format: Any):
        if not isinstance(response_format, Mapping):
            return response_format

        format_type = response_format.get("type")
        if format_type == "json_schema":
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, Mapping):
                adapted_format = {"type": "json_schema"}
                for key in ("name", "schema", "strict"):
                    value = json_schema.get(key)
                    if value is not None:
                        adapted_format[key] = value
                return adapted_format

        if format_type in {"json_object", "text"}:
            return {"type": format_type}

        return dict(response_format)

    def _adapt_responses_stream_options(self, stream_options: Any):
        if not isinstance(stream_options, Mapping):
            return stream_options

        adapted_stream_options = dict(stream_options)
        # chat.completions compatibility flag; responses returns usage in
        # `response.completed`, so `include_usage` has no effect here.
        adapted_stream_options.pop("include_usage", None)
        if not adapted_stream_options:
            return None
        return adapted_stream_options

    def _adapt_responses_web_search_tool(self, web_search_options: Any):  # noqa: C901
        web_search_tool = {"type": "web_search"}
        if not isinstance(web_search_options, Mapping):
            return web_search_tool

        options = dict(web_search_options)
        tool_type = options.pop("tool_type", None) or options.pop("type", None)
        if isinstance(tool_type, str) and tool_type:
            web_search_tool["type"] = tool_type

        search_context_size = options.get("search_context_size")
        if search_context_size is not None:
            web_search_tool["search_context_size"] = search_context_size

        user_location = options.get("user_location")
        if isinstance(user_location, Mapping):
            adapted_user_location = dict(user_location)
            approximate_location = adapted_user_location.get("approximate")
            if isinstance(approximate_location, Mapping):
                adapted_user_location = dict(approximate_location)
                location_type = user_location.get("type")
                if location_type is not None:
                    adapted_user_location["type"] = location_type
                else:
                    adapted_user_location["type"] = "approximate"
            web_search_tool["user_location"] = adapted_user_location

        # `web_search` supports filters while
        # `web_search_preview` does not.
        if "preview" not in web_search_tool["type"]:
            filters = options.get("filters")
            if isinstance(filters, Mapping):
                web_search_tool["filters"] = dict(filters)

            allowed_domains = options.get("allowed_domains")
            if allowed_domains is not None:
                current_filters = web_search_tool.get("filters")
                if isinstance(current_filters, Mapping):
                    adapted_filters = dict(current_filters)
                    adapted_filters["allowed_domains"] = allowed_domains
                    web_search_tool["filters"] = adapted_filters
                else:
                    web_search_tool["filters"] = {"allowed_domains": allowed_domains}

        return web_search_tool

    def _get_responses_supported_params(self) -> set[str]:
        cached = getattr(self, "_responses_supported_params_cache", None)
        if cached is not None:
            return cached

        # Fallback for mocked clients or signature introspection
        # failures.
        supported_params = {
            "background",
            "context_management",
            "conversation",
            "include",
            "input",
            "instructions",
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "model",
            "parallel_tool_calls",
            "previous_response_id",
            "prompt",
            "prompt_cache_key",
            "prompt_cache_retention",
            "reasoning",
            "safety_identifier",
            "service_tier",
            "store",
            "stream",
            "stream_options",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }
        try:
            if self.client is not None:
                signature = inspect.signature(self.client.responses.create)
                candidate_params = set(signature.parameters)
                # Ignore generic mock signatures like (*args, **kwargs)
                if not candidate_params.issubset({"args", "kwargs"}):
                    supported_params = candidate_params
        except Exception:  # noqa: S110
            pass

        self._responses_supported_params_cache = supported_params
        return supported_params

    def _adapt_responses_params(  # noqa: C901
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        supported_params = self._get_responses_supported_params()

        # Parameter adaptation between /chat/completions and /responses
        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        reasoning_effort = params.pop("reasoning_effort", None)
        if reasoning_effort is not None:
            reasoning = params.get("reasoning")
            if isinstance(reasoning, Mapping):
                adapted_reasoning = dict(reasoning)
                adapted_reasoning.setdefault("effort", reasoning_effort)
                params["reasoning"] = adapted_reasoning
            elif reasoning is None:
                params["reasoning"] = {"effort": reasoning_effort}

        provider_tools = params.pop("provider_tools", None)
        merged_tools = self._merge_tools(params.get("tools"), provider_tools)
        tools = self._adapt_responses_tools(merged_tools)
        if tools is None:
            params.pop("tools", None)
        else:
            params["tools"] = tools

        tool_choice = self._adapt_responses_tool_choice(params.get("tool_choice"))
        if tool_choice is None:
            params.pop("tool_choice", None)
        else:
            params["tool_choice"] = tool_choice

        response_format = params.pop("response_format", None)
        verbosity = params.pop("verbosity", None)
        if response_format is not None or verbosity is not None:
            if "text" in supported_params:
                current_text = params.get("text")
                text_config = (
                    dict(current_text) if isinstance(current_text, Mapping) else {}
                )
                if response_format is not None:
                    text_config["format"] = self._adapt_response_format_to_text_format(
                        response_format
                    )
                if verbosity is not None:
                    text_config["verbosity"] = verbosity
                params["text"] = text_config
            else:
                # Forward original names for providers/SDKs that
                # expose them.
                if response_format is not None:
                    params["response_format"] = response_format
                if verbosity is not None:
                    params["verbosity"] = verbosity

        web_search_options = params.pop("web_search_options", None)
        if web_search_options is not None:
            if "tools" in supported_params:
                web_search_tool = self._adapt_responses_web_search_tool(
                    web_search_options
                )
                existing_tools = params.get("tools")
                if isinstance(existing_tools, list):
                    params["tools"] = [
                        *existing_tools,
                        web_search_tool,
                    ]
                elif existing_tools is None:
                    params["tools"] = [web_search_tool]
                else:
                    params["tools"] = [
                        existing_tools,
                        web_search_tool,
                    ]
            elif "web_search_options" in supported_params:
                params["web_search_options"] = web_search_options
            else:
                # No known compatible field.
                pass

        stream_options = params.get("stream_options")
        if stream_options is not None:
            adapted_stream_options = self._adapt_responses_stream_options(
                stream_options
            )
            if adapted_stream_options is None:
                params.pop("stream_options", None)
            else:
                params["stream_options"] = adapted_stream_options

        # Keep only params that are either supported or mapped to
        # equivalents.
        for field in ("audio", "modalities", "stop", "stream_options"):
            if field in supported_params:
                continue
            params.pop(field, None)

        return params

    def _ensure_chat_messages(
        self,
        messages: Union[ChatMessages, List[Dict[str, Any]], None],
    ) -> ChatMessages:
        if isinstance(messages, ChatMessages):
            return messages.copy()
        if messages is None:
            return ChatMessages()
        return ChatMessages.from_chatml(messages)

    def _build_usage_metadata(self, model_output) -> dotdict:
        metadata = dotdict()
        usage = getattr(model_output, "usage", None)
        if usage is not None:
            if hasattr(usage, "to_dict"):
                metadata.update({"usage": usage.to_dict()})
            elif hasattr(usage, "model_dump"):
                metadata.update({"usage": usage.model_dump()})
            elif isinstance(usage, Mapping):
                metadata.update({"usage": dict(usage)})
        return metadata

    def _set_stop_metadata(
        self,
        metadata: dotdict,
        *,
        finish_reason: Optional[str] = None,
        stop_reason: Optional[str] = None,
    ) -> None:
        if finish_reason is None and stop_reason is not None:
            finish_reason = stop_reason
        if stop_reason is None and finish_reason is not None:
            stop_reason = finish_reason
        if finish_reason is not None:
            metadata.finish_reason = finish_reason
        if stop_reason is not None:
            metadata.stop_reason = stop_reason

    @staticmethod
    def _extract_reasoning(message) -> Optional[str]:
        return (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or getattr(message, "thinking", None)
        )

    @staticmethod
    def _extract_finish_reason(choice) -> Optional[str]:
        return getattr(choice, "finish_reason", None)

    @staticmethod
    def _extract_annotations(message) -> Optional[list]:
        annotations = getattr(message, "annotations", None)
        if annotations:
            return [item.model_dump() for item in annotations]
        return None

    @staticmethod
    def _process_stream_tool_calls(delta, stream_response, aggregator):
        tool_call = delta.tool_calls[0]
        if stream_response.response_type is None:
            stream_response.set_response_type("tool_call")
        aggregator.process(
            tool_call.index, tool_call.id,
            tool_call.function.name, tool_call.function.arguments,
        )

    @staticmethod
    def _stream_add_chunk(stream_response, chunk, response_type):
        if stream_response.response_type is None:
            stream_response.set_response_type(response_type)
            stream_response.first_chunk_event.set()
        stream_response.add(chunk)

    def _set_reasoning_fields(self, response_content: Any, reasoning_content: str):
        if response_content is None or isinstance(response_content, str):
            return
        response_content.reasoning = reasoning_content

    def _extract_responses_stop_reason(self, model_output) -> Optional[str]:
        incomplete_details = getattr(model_output, "incomplete_details", None)
        if isinstance(incomplete_details, Mapping):
            reason = incomplete_details.get("reason")
            if isinstance(reason, str):
                return reason
        elif incomplete_details is not None:
            reason = getattr(incomplete_details, "reason", None)
            if isinstance(reason, str):
                return reason

        status = getattr(model_output, "status", None)
        if isinstance(status, str):
            return status
        return None

    def _extract_responses_reasoning(self, model_output) -> Optional[str]:
        reasoning_chunks = []
        for item in getattr(model_output, "output", []):
            if getattr(item, "type", None) != "reasoning":
                continue
            for content_part in getattr(item, "content", []) or []:
                text = getattr(content_part, "text", None)
                if text:
                    reasoning_chunks.append(text)
            if not reasoning_chunks:
                for summary_part in getattr(item, "summary", []) or []:
                    text = getattr(summary_part, "text", None)
                    if text:
                        reasoning_chunks.append(text)

        if not reasoning_chunks:
            return None
        return "".join(reasoning_chunks)

    def _extract_responses_text_and_annotations(self, model_output):  # noqa: C901
        text_parts = []
        annotations = []

        for item in getattr(model_output, "output", []):
            if getattr(item, "type", None) != "message":
                continue
            for content_part in getattr(item, "content", []) or []:
                content_type = getattr(content_part, "type", None)
                if content_type == "output_text":
                    text = getattr(content_part, "text", None)
                    if text:
                        text_parts.append(text)
                    for ann in getattr(content_part, "annotations", []) or []:
                        if hasattr(ann, "model_dump"):
                            annotations.append(ann.model_dump())
                        elif isinstance(ann, Mapping):
                            annotations.append(dict(ann))
                elif content_type == "refusal":
                    refusal = getattr(content_part, "refusal", None)
                    if refusal:
                        text_parts.append(refusal)

        response_text = getattr(model_output, "output_text", None)
        if response_text:
            return response_text, annotations

        return "\n".join(text_parts).strip(), annotations

    def _execute_model(self, **kwargs):
        prefilling = kwargs.pop("prefilling")
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)

        params = {
            "model": kwargs.pop("model"),
            **kwargs,
            **self.sampling_run_params,
        }

        if self.provider_tools and self.api_mode != "response":
            raise ValueError(
                "`provider_tools` requires `api_mode='response'` for OpenAI."
            )

        if self.api_mode == "response":
            params["provider_tools"] = self.provider_tools
            params["input"] = messages.to_responses_input()
            adapted_params = self._adapt_responses_params(params)
            model_output = self.client.responses.create(**adapted_params)
            return model_output

        params["messages"] = messages.to_chatml()
        adapted_params = self._adapt_params(params)
        model_output = self.client.chat.completions.create(**adapted_params)

        return model_output

    async def _aexecute_model(self, **kwargs):
        prefilling = kwargs.pop("prefilling")
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)

        params = {
            "model": kwargs.pop("model"),
            **kwargs,
            **self.sampling_run_params,
        }

        if self.provider_tools and self.api_mode != "response":
            raise ValueError(
                "`provider_tools` requires `api_mode='response'` for OpenAI."
            )

        if self.api_mode == "response":
            params["provider_tools"] = self.provider_tools
            params["input"] = messages.to_responses_input()
            adapted_params = self._adapt_responses_params(params)
            model_output = await self.aclient.responses.create(**adapted_params)
            return model_output

        params["messages"] = messages.to_chatml()
        adapted_params = self._adapt_params(params)
        model_output = await self.aclient.chat.completions.create(**adapted_params)

        return model_output

    def _process_completion_model_output(  # noqa: C901
        self, model_output, typed_parser=None, generation_schema=None
    ):
        response = ModelResponse()
        metadata = self._build_usage_metadata(model_output)

        choice = model_output.choices[0]
        self._set_stop_metadata(
            metadata, finish_reason=self._extract_finish_reason(choice)
        )

        reasoning = self._extract_reasoning(choice.message)

        reasoning_tool_call = reasoning if self.reasoning_in_tool_call else None

        prefix_response_type = ""
        reasoning_content = None
        if self.return_reasoning is True and reasoning is not None:
            reasoning_content = reasoning
            prefix_response_type = "reasoning_"

        annotations = self._extract_annotations(choice.message)
        if annotations:
            metadata.annotations = annotations

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator(reasoning_tool_call)
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                tool_id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif choice.message.content:
            if (typed_parser or generation_schema) and self.verbose:
                repr_str = f"[{self.model_id}][raw_response] {choice.message.content}"
                cprint(repr_str, lc="r", ls="b")
            if typed_parser is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.decode(choice.message.content))
                if generation_schema and self.validate_typed_parser_output:
                    decoder = self._get_decoder(generation_schema)
                    decoder.decode(self._encoder.encode(response_content))
            elif generation_schema is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                decoder = self._get_decoder(generation_schema)
                struct = decoder.decode(choice.message.content)
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type(f"{prefix_response_type}text_generation")
                if reasoning_content is not None:
                    response_content = dotdict({"answer": choice.message.content})
                else:
                    response_content = choice.message.content
        elif choice.message.audio:
            response_content = dotdict(
                {
                    "id": choice.message.audio.id,
                    "audio": base64.b64decode(choice.message.audio.data),
                }
            )
            if choice.message.audio.transcript:
                response.set_response_type("audio_text_generation")
                response_content.text = choice.message.audio.transcript
            else:
                response.set_response_type("audio_generation")
        else:
            response.set_response_type("text_generation")
            response_content = ""

        if reasoning_content is not None:
            if isinstance(response_content, str):
                response_content = dotdict({"answer": response_content})
            self._set_reasoning_fields(response_content, reasoning_content)

        response.add(response_content)
        response.set_metadata(metadata)
        return response

    def _process_responses_model_output(  # noqa: C901
        self,
        model_output,
        typed_parser=None,
        generation_schema=None,
    ):
        response = ModelResponse()
        metadata = self._build_usage_metadata(model_output)

        for field in ("id", "status", "previous_response_id"):
            value = getattr(model_output, field, None)
            if value is not None:
                metadata[field] = value
        self._set_stop_metadata(
            metadata,
            stop_reason=self._extract_responses_stop_reason(model_output),
        )

        conversation = getattr(model_output, "conversation", None)
        if conversation is not None and getattr(conversation, "id", None):
            metadata.conversation_id = conversation.id

        reasoning = self._extract_responses_reasoning(model_output)
        reasoning_tool_call = reasoning if self.reasoning_in_tool_call else None

        prefix_response_type = ""
        reasoning_content = None
        if self.return_reasoning and reasoning is not None:
            reasoning_content = reasoning
            prefix_response_type = "reasoning_"

        function_calls = []
        for item in getattr(model_output, "output", []):
            if getattr(item, "type", None) == "function_call":
                function_calls.append(item)

        (
            response_text,
            annotations,
        ) = self._extract_responses_text_and_annotations(model_output)
        if annotations:
            metadata.annotations = annotations

        if function_calls:
            aggregator = ToolCallAggregator(reasoning_tool_call)
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(function_calls):
                tool_id = getattr(tool_call, "call_id", None) or getattr(
                    tool_call, "id", None
                )
                name = getattr(tool_call, "name", None)
                arguments = getattr(tool_call, "arguments", None) or "{}"
                aggregator.process(call_index, tool_id, name, arguments)
            response_content = aggregator
        elif response_text:
            if (typed_parser or generation_schema) and self.verbose:
                repr_str = f"[{self.model_id}][raw_response] {response_text}"
                cprint(repr_str, lc="r", ls="b")

            if typed_parser is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.decode(response_text))
                if generation_schema and self.validate_typed_parser_output:
                    decoder = self._get_decoder(generation_schema)
                    decoder.decode(self._encoder.encode(response_content))
            elif generation_schema is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                decoder = self._get_decoder(generation_schema)
                struct = decoder.decode(response_text)
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type(f"{prefix_response_type}text_generation")
                if reasoning_content is not None:
                    response_content = dotdict({"answer": response_text})
                else:
                    response_content = response_text
        else:
            response.set_response_type("text_generation")
            response_content = ""

        if reasoning_content is not None:
            if isinstance(response_content, str):
                response_content = dotdict({"answer": response_content})
            self._set_reasoning_fields(response_content, reasoning_content)

        response.add(response_content)
        response.set_metadata(metadata)
        return response

    def _process_model_output(
        self, model_output, typed_parser=None, generation_schema=None
    ):
        if self.api_mode == "response":
            return self._process_responses_model_output(
                model_output, typed_parser, generation_schema
            )
        return self._process_completion_model_output(
            model_output, typed_parser, generation_schema
        )

    def _generate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        typed_parser = kwargs.get("typed_parser")
        generation_schema = kwargs.get("generation_schema")

        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        # Pop after cache check to avoid modifying kwargs during
        # cache key generation
        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        if generation_schema is not None and typed_parser is None:
            response_format = response_format_from_msgspec_struct(generation_schema)
            if self.api_mode == "response":
                json_schema = response_format["json_schema"]
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema["name"],
                        "schema": json_schema["schema"],
                        "strict": json_schema.get("strict", True),
                    }
                }
            else:
                kwargs["response_format"] = response_format

        model_output = self._execute_model(**kwargs)
        response = self._process_model_output(
            model_output, typed_parser, generation_schema
        )

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            # Re-add popped values for cache key
            cache_kwargs = {
                **kwargs,
                "typed_parser": typed_parser,
                "generation_schema": generation_schema,
            }
            cache_key = generate_cache_key(**cache_kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        typed_parser = kwargs.get("typed_parser")
        generation_schema = kwargs.get("generation_schema")

        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        # Pop after cache check to avoid modifying kwargs during
        # cache key generation
        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        if generation_schema is not None and typed_parser is None:
            response_format = response_format_from_msgspec_struct(generation_schema)
            if self.api_mode == "response":
                json_schema = response_format["json_schema"]
                kwargs["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema["name"],
                        "schema": json_schema["schema"],
                        "strict": json_schema.get("strict", True),
                    }
                }
            else:
                kwargs["response_format"] = response_format

        model_output = await self._aexecute_model(**kwargs)
        response = self._process_model_output(
            model_output, typed_parser, generation_schema
        )

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            # Re-add popped values for cache key
            cache_kwargs = {
                **kwargs,
                "typed_parser": typed_parser,
                "generation_schema": generation_schema,
            }
            cache_key = generate_cache_key(**cache_kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def _stream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> ModelStreamResponse:
        stream_response = kwargs.pop("stream_response")
        metadata = dotdict()
        reasoning_tool_call = ""

        try:
            if self.api_mode == "response":
                model_output = self._execute_model(**kwargs)
                completed_response = None

                for event in model_output:
                    event_type = getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("text_generation")
                                stream_response.first_chunk_event.set()
                            stream_response.add(delta)
                        continue

                    if event_type in {
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = getattr(event, "delta", "")
                        if self.reasoning_in_tool_call and delta:
                            reasoning_tool_call += delta
                        if self.return_reasoning and delta:
                            if stream_response.response_type is None:
                                stream_response.set_response_type(
                                    "reasoning_text_generation"
                                )
                                stream_response.first_chunk_event.set()
                            stream_response.add(delta)
                        continue

                    if event_type == "response.completed":
                        completed_response = getattr(event, "response", None)

                if completed_response is not None:
                    metadata.update(self._build_usage_metadata(completed_response))
                    for field in (
                        "id",
                        "status",
                        "previous_response_id",
                    ):
                        value = getattr(completed_response, field, None)
                        if value is not None:
                            metadata[field] = value
                    self._set_stop_metadata(
                        metadata,
                        stop_reason=self._extract_responses_stop_reason(
                            completed_response
                        ),
                    )

                    aggregator = ToolCallAggregator(
                        reasoning_tool_call if self.reasoning_in_tool_call else None
                    )
                    function_calls = []
                    for item in getattr(completed_response, "output", []):
                        if getattr(item, "type", None) == "function_call":
                            function_calls.append(item)

                    if function_calls:
                        if stream_response.response_type is None:
                            stream_response.set_response_type("tool_call")
                        for call_index, tool_call in enumerate(function_calls):
                            tool_id = getattr(tool_call, "call_id", None) or getattr(
                                tool_call, "id", None
                            )
                            name = getattr(tool_call, "name", None)
                            arguments = getattr(tool_call, "arguments", None) or "{}"
                            aggregator.process(
                                call_index,
                                tool_id,
                                name,
                                arguments,
                            )
                        stream_response.data = aggregator
                        stream_response.first_chunk_event.set()
                    elif stream_response.response_type is None:
                        (
                            response_text,
                            _,
                        ) = self._extract_responses_text_and_annotations(
                            completed_response
                        )
                        stream_response.set_response_type("text_generation")
                        if response_text:
                            stream_response.add(response_text)
                        stream_response.first_chunk_event.set()
            else:
                aggregator = ToolCallAggregator()
                model_output = self._execute_model(**kwargs)
                finish_reason = None

                for chunk in model_output:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta
                        fr = self._extract_finish_reason(choice)
                        if fr is not None:
                            finish_reason = fr

                        reasoning_chunk = self._extract_reasoning(delta)

                        if self.reasoning_in_tool_call and reasoning_chunk:
                            reasoning_tool_call += reasoning_chunk

                        if self.return_reasoning and reasoning_chunk:
                            self._stream_add_chunk(
                                stream_response, reasoning_chunk,
                                "reasoning_text_generation",
                            )
                            continue

                        if getattr(delta, "content", None):
                            self._stream_add_chunk(
                                stream_response, delta.content, "text_generation",
                            )
                            continue

                        if getattr(delta, "tool_calls", None):
                            self._process_stream_tool_calls(
                                delta, stream_response, aggregator,
                            )
                            continue

                        annotations = self._extract_annotations(delta)
                        if annotations:
                            metadata.annotations = annotations
                            continue

                    elif chunk.usage:
                        usage = chunk.usage.to_dict()
                        metadata.update(usage)
                        metadata.usage = usage

                if aggregator.tool_calls:
                    if reasoning_tool_call:
                        aggregator.reasoning = reasoning_tool_call
                    stream_response.data = aggregator
                    stream_response.first_chunk_event.set()
                self._set_stop_metadata(metadata, finish_reason=finish_reason)
        finally:
            if not stream_response.first_chunk_event.is_set():
                stream_response.first_chunk_event.set()
            stream_response.set_metadata(metadata)
            stream_response.add(None)

    async def _astream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> ModelStreamResponse:
        stream_response = kwargs.pop("stream_response")
        metadata = dotdict()
        reasoning_tool_call = ""

        try:
            if self.api_mode == "response":
                model_output = await self._aexecute_model(**kwargs)
                completed_response = None

                async for event in model_output:
                    event_type = getattr(event, "type", None)

                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("text_generation")
                                stream_response.first_chunk_event.set()
                            stream_response.add(delta)
                        continue

                    if event_type in {
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = getattr(event, "delta", "")
                        if self.reasoning_in_tool_call and delta:
                            reasoning_tool_call += delta
                        if self.return_reasoning and delta:
                            if stream_response.response_type is None:
                                stream_response.set_response_type(
                                    "reasoning_text_generation"
                                )
                                stream_response.first_chunk_event.set()
                            stream_response.add(delta)
                        continue

                    if event_type == "response.completed":
                        completed_response = getattr(event, "response", None)

                if completed_response is not None:
                    metadata.update(self._build_usage_metadata(completed_response))
                    for field in (
                        "id",
                        "status",
                        "previous_response_id",
                    ):
                        value = getattr(completed_response, field, None)
                        if value is not None:
                            metadata[field] = value
                    self._set_stop_metadata(
                        metadata,
                        stop_reason=self._extract_responses_stop_reason(
                            completed_response
                        ),
                    )

                    aggregator = ToolCallAggregator(
                        reasoning_tool_call if self.reasoning_in_tool_call else None
                    )
                    function_calls = []
                    for item in getattr(completed_response, "output", []):
                        if getattr(item, "type", None) == "function_call":
                            function_calls.append(item)

                    if function_calls:
                        if stream_response.response_type is None:
                            stream_response.set_response_type("tool_call")
                        for call_index, tool_call in enumerate(function_calls):
                            tool_id = getattr(tool_call, "call_id", None) or getattr(
                                tool_call, "id", None
                            )
                            name = getattr(tool_call, "name", None)
                            arguments = getattr(tool_call, "arguments", None) or "{}"
                            aggregator.process(
                                call_index,
                                tool_id,
                                name,
                                arguments,
                            )
                        stream_response.data = aggregator
                        stream_response.first_chunk_event.set()
                    elif stream_response.response_type is None:
                        (
                            response_text,
                            _,
                        ) = self._extract_responses_text_and_annotations(
                            completed_response
                        )
                        stream_response.set_response_type("text_generation")
                        if response_text:
                            stream_response.add(response_text)
                        stream_response.first_chunk_event.set()
            else:
                aggregator = ToolCallAggregator()
                model_output = await self._aexecute_model(**kwargs)
                finish_reason = None

                async for chunk in model_output:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta
                        fr = self._extract_finish_reason(choice)
                        if fr is not None:
                            finish_reason = fr

                        reasoning_chunk = self._extract_reasoning(delta)

                        if self.reasoning_in_tool_call and reasoning_chunk:
                            reasoning_tool_call += reasoning_chunk

                        if self.return_reasoning and reasoning_chunk:
                            self._stream_add_chunk(
                                stream_response, reasoning_chunk,
                                "reasoning_text_generation",
                            )
                            continue

                        if getattr(delta, "content", None):
                            self._stream_add_chunk(
                                stream_response, delta.content, "text_generation",
                            )
                            continue

                        if getattr(delta, "tool_calls", None):
                            self._process_stream_tool_calls(
                                delta, stream_response, aggregator,
                            )
                            continue

                        annotations = self._extract_annotations(delta)
                        if annotations:
                            metadata.annotations = annotations
                            continue

                    elif chunk.usage:
                        usage = chunk.usage.to_dict()
                        metadata.update(usage)
                        metadata.usage = usage

                if aggregator.tool_calls:
                    if reasoning_tool_call:
                        aggregator.reasoning = reasoning_tool_call
                    stream_response.data = aggregator
                    stream_response.first_chunk_event.set()
                self._set_stop_metadata(metadata, finish_reason=finish_reason)
        finally:
            if not stream_response.first_chunk_event.is_set():
                stream_response.first_chunk_event.set()
            stream_response.set_metadata(metadata)
            stream_response.add(None)

    def __call__(  # noqa: C901
        self,
        messages: Union[str, List[Dict[str, Any]], ChatMessages],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
            messages:
                Conversation history. Can be simple string, list of messages,
                or ChatMessages container.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message
                it will continue its response from there.
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
                    2. required:
                        Call one or more functions.
                    3. Forced Tool:
                        Call exactly one specific tool e.g: "get_weather".
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_xml=True` and `stream=True`.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        elif isinstance(messages, ChatMessages):
            messages = self._ensure_chat_messages(messages)
        if isinstance(system_prompt, str):
            messages.insert(0, ChatBlock.system(system_prompt))

        if isinstance(tool_choice, str):
            if tool_choice not in ["auto", "required", "none"]:
                if self.api_mode == "response":
                    tool_choice = {
                        "type": "function",
                        "name": tool_choice,
                    }
                else:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "tool_choice": tool_choice,
            "tools": tool_schemas,
            "model": self.model_id,
        }

        if self._merge_tools(tool_schemas, self.provider_tools):
            generation_params["parallel_tool_calls"] = self.parallel_tool_calls

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse(mode="sync")
            stream_kwargs = {
                "stream": stream,
                "stream_response": stream_response,
            }
            if self.api_mode != "response":
                stream_kwargs["stream_options"] = {"include_usage": True}
            F.fire_and_forget(
                self._stream_generate,
                **generation_params,
                **stream_kwargs,
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

    async def acall(  # noqa: C901
        self,
        messages: Union[str, List[Dict[str, Any]], ChatMessages],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        typed_parser: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
            messages:
                Conversation history. Can be simple string, list of messages,
                or ChatMessages container.
            system_prompt:
                A set of instructions that defines the overarching behavior
                and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message
                it will continue its response from there.
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
                    2. required:
                        Call one or more functions.
                    3. Forced Tool:
                        Call exactly one specific tool e.g: "get_weather".
            typed_parser:
                Converts the model raw output into a typed-dict. Supported parser:
                `typed_xml`.

        Raises:
            ValueError:
                Raised if `generation_schema` and `stream=True`.
            ValueError:
                Raised if `typed_xml=True` and `stream=True`.
        """
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        elif isinstance(messages, ChatMessages):
            messages = self._ensure_chat_messages(messages)
        if isinstance(system_prompt, str):
            messages.insert(0, ChatBlock.system(system_prompt))

        if isinstance(tool_choice, str):
            if tool_choice not in ["auto", "required", "none"]:
                if self.api_mode == "response":
                    tool_choice = {
                        "type": "function",
                        "name": tool_choice,
                    }
                else:
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "tool_choice": tool_choice,
            "tools": tool_schemas,
            "model": self.model_id,
        }

        if self._merge_tools(tool_schemas, self.provider_tools):
            generation_params["parallel_tool_calls"] = self.parallel_tool_calls

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse(mode="async")
            stream_kwargs = {
                "stream": stream,
                "stream_response": stream_response,
            }
            if self.api_mode != "response":
                stream_kwargs["stream_options"] = {"include_usage": True}
            await F.afire_and_forget(
                self._astream_generate,
                **generation_params,
                **stream_kwargs,
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


@register_model
class OpenAITextToSpeech(_BaseOpenAI, TextToSpeechModel):
    """OpenAI Text to Speech."""

    def __init__(
        self,
        model_id: str,
        voice: Optional[str] = "alloy",
        speed: Optional[float] = 1.0,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        voice:
            The voice to use when generating the audio.
        speed:
            the speed of the generated audio. Select a value
            from 0.25 to 4.0. 1.0 is the default.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {
            "voice": voice,
            "speed": speed,
        }
        self.retry = retry
        self._initialize()
        self._get_api_key()

    @contextmanager
    def _execute_model(self, **kwargs):
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    @asynccontextmanager
    async def _aexecute_model(self, **kwargs):
        async with self.aclient.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        with self._execute_model(**kwargs) as model_output:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                model_output.stream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()

        async with self._aexecute_model(**kwargs) as model_output:
            with tempfile.NamedTemporaryFile(
                suffix=f".{kwargs.get('response_format')}", delete=False
            ) as temp_file:
                temp_file_path = temp_file.name
                await model_output.astream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        with self._execute_model(**kwargs) as model_output:
            for chunk in model_output.iter_bytes(chunk_size=1024):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        async with self._aexecute_model(**kwargs) as model_output:
            async for chunk in model_output.aiter_bytes(chunk_size=1024):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    def __call__(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            The text to generate audio for.
        stream:
            Whether generation should be in streaming mode.
        prompt:
            Control the voice of your generated audio with additional instructions.
        response_format:
            The format to audio in.
        """
        params = dotdict({"input": data, "response_format": response_format})
        if prompt:
            params.instructions = prompt
        if stream:
            stream_response = ModelStreamResponse(mode="sync")
            params.stream_response = stream_response
            F.fire_and_forget(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        prompt: Optional[str] = None,
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
        data:
            The text to generate audio for.
        stream:
            Whether generation should be in streaming mode.
        prompt:
            Control the voice of your generated audio with additional instructions.
        response_format:
            The format to audio in.
        """
        params = dotdict({"input": data, "response_format": response_format})
        if prompt:
            params.instructions = prompt
        if stream:
            stream_response = ModelStreamResponse(mode="async")
            params.stream_response = stream_response
            await F.afire_and_forget(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextToImage(_BaseOpenAI, TextToImageModel):
    """OpenAI Image Generation."""

    def __init__(
        self,
        *,
        model_id: str,
        moderation: Optional[Literal["auto", "low"]] = None,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        moderation:
            Control the content-moderation level for images generated.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        sampling_run_params = {}
        if moderation:
            sampling_run_params["moderation"] = moderation
        self.sampling_run_params = sampling_run_params
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.images.generate(**kwargs, **self.sampling_run_params)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.images.generate(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _get_metadata(self, model_output):
        metadata = dotdict(
            usage=(
                model_output.usage.to_dict() if model_output.usage is not None else {}
            ),
            details={
                "size": getattr(model_output, "size", None),
                "quality": getattr(model_output, "quality", None),
                "output_format": getattr(model_output, "output_format", None),
                "background": getattr(model_output, "background", None),
            },
        )
        return metadata

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_generation")

        model_output = self._execute_model(**kwargs)

        metadata = self._get_metadata(model_output)

        images = []
        for item in model_output.data:
            if item.url:
                images.append(item.url)
            if item.b64_json:
                images.append(item.b64_json)

        if len(images) == 1:
            images = images[0]

        response.add(images)
        response.set_metadata(metadata)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("image_generation")

        model_output = await self._aexecute_model(**kwargs)

        metadata = self._get_metadata(model_output)

        images = []
        for item in model_output.data:
            if item.url:
                images.append(item.url)
            if item.b64_json:
                images.append(item.b64_json)

        if len(images) == 1:
            images = images[0]

        response.add(images)
        response.set_metadata(metadata)

        return response

    def __call__(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = self._generate(**generation_params)
        return response

    async def acall(
        self,
        prompt: str,
        *,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[Literal["transparent", "opaque", "auto"]] = None,
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        size:
            The size of the generated images.
        quality:
            The quality of the image that will be generated.
        background:
            Allows to set transparency for the background of the generated image(s).
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if size is not None:
            generation_params.size = size
        if quality is not None:
            generation_params.quality = quality
        if background is not None:
            generation_params.background = background
        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        response = await self._agenerate(**generation_params)
        return response


@register_model
class OpenAIImageTextToImage(ImageTextToImageModel, OpenAITextToImage):
    """OpenAI Image Edit."""

    def _execute_model(self, **kwargs):
        model_output = self.client.images.edit(**kwargs, **self.sampling_run_params)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.images.edit(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _prepare_inputs(self, image, mask):
        inputs = {}
        if isinstance(image, (str, bytes)):
            image = [image]
        inputs["image"] = [encode_data_to_bytes(item) for item in image]
        if mask:
            inputs["mask"] = encode_data_to_bytes(mask)
        return inputs

    def __call__(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = self._generate(**generation_params, **inputs)
        return response

    async def acall(
        self,
        prompt: str,
        image: Union[str, List[str]],
        *,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = None,
        n: Optional[int] = 1,
    ) -> ModelResponse:
        """Async version of __call__. Args:
        prompt:
            A text description of the desired image(s).
        image:
            The image(s) to edit. Can be a path, an url or base64 string.
        mask:
            An additional image whose fully transparent areas
            (e.g. where alpha is zero) indicate where image
            should be edited. If there are multiple images provided,
            the mask will be applied on the first image.
        response_format:
            Format in which images are returned.
        n:
            The number of images to generate.
        """
        generation_params = dotdict(prompt=prompt, n=n, model=self.model_id)

        if response_format is not None:
            if response_format == "base64":
                response_format = "b64_json"
            generation_params.response_format = response_format

        inputs = self._prepare_inputs(image, mask)
        response = await self._agenerate(**generation_params, **inputs)
        return response


@register_model
class OpenAISpeechToText(_BaseOpenAI, SpeechToTextModel):
    """OpenAI Speech to Text."""

    def __init__(
        self,
        *,
        model_id: str,
        temperature: Optional[float] = 0.0,
        base_url: Optional[str] = None,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        temperature:
            The sampling temperature, between 0 and 1.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"temperature": temperature}
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.audio.transcriptions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.audio.transcriptions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("transcript")

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            if model_output.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in model_output.words
                ]
                transcript["words"] = words
            if model_output.segment:
                segments = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in model_output.segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    async def _agenerate(self, **kwargs):
        response = ModelResponse()

        model_output = await self._aexecute_model(**kwargs)

        response.set_response_type("transcript")

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            if model_output.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in model_output.words
                ]
                transcript["words"] = words
            if model_output.segment:
                segments = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in model_output.segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = self._execute_model(**kwargs)

        for event in model_output:
            chunk = event.transcript.text.delta
            if chunk:
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()
            elif event.transcript.text.done:
                stream_response.add(None)

        return stream_response

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = await self._aexecute_model(**kwargs)

        async for event in model_output:
            chunk = event.transcript.text.delta
            if chunk:
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()
            elif event.transcript.text.done:
                stream_response.add(None)

        return stream_response

    def __call__(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Args:
        data:
            Url, path, base64 to audio.
        stream:
            Whether generation should be in streaming mode.
        response_format:
            The format of the output, in one of these options:
            json, text, srt, verbose_json, or vtt.
        timestamp_granularities:
            The timestamp granularities to populate for this
            transcription. `response_format` must be set `verbose_json`
            to use timestamp granularities. Either or both of these
            options are supported: word, or segment. Note: There is no
            additional latency for segment timestamps, but generating
            word timestamps incurs additional latency.
        prompt:
            An optional text to guide the model's style or continue a
            previous audio segment. The prompt should match the audio language.
        language:
            The language of the input audio. Supplying the input language in
            ISO-639-1 (e.g. en) format will improve accuracy and latency.
        """
        file = encode_data_to_bytes(data)
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="sync")
            params["stream_response"] = stream_response
            params["stream"] = stream
            F.fire_and_forget(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**params)
            return response

    async def acall(
        self,
        data: str,
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """Async version of __call__. Args:
        data:
            Url, path, base64 to audio.
        stream:
            Whether generation should be in streaming mode.
        response_format:
            The format of the output, in one of these options:
            json, text, srt, verbose_json, or vtt.
        timestamp_granularities:
            The timestamp granularities to populate for this
            transcription. `response_format` must be set `verbose_json`
            to use timestamp granularities. Either or both of these
            options are supported: word, or segment. Note: There is no
            additional latency for segment timestamps, but generating
            word timestamps incurs additional latency.
        prompt:
            An optional text to guide the model's style or continue a
            previous audio segment. The prompt should match the audio language.
        language:
            The language of the input audio. Supplying the input language in
            ISO-639-1 (e.g. en) format will improve accuracy and latency.
        """
        file = encode_data_to_bytes(data)
        params = {
            "file": file,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
            "model": self.model_id,
        }
        if stream:
            stream_response = ModelStreamResponse(mode="async")
            params["stream_response"] = stream_response
            params["stream"] = stream
            await F.afire_and_forget(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextEmbedder(_BaseOpenAI, TextEmbedderModel):
    """OpenAI Text Embedder."""

    batch_support: bool = True

    def __init__(
        self,
        *,
        model_id: str,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.sampling_run_params = {"dimensions": dimensions}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.embeddings.create(
            **kwargs,
            **self.sampling_run_params,
        )
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.embeddings.create(
            **kwargs,
            **self.sampling_run_params,
        )
        return model_output

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict({"usage": model_output.usage.to_dict()})
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = await self._aexecute_model(**kwargs)
        embeddings = [item.embedding for item in model_output.data]
        metadata = dotdict({"usage": model_output.usage.to_dict()})
        response.add(embeddings)
        response.set_metadata(metadata)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[str]],
    ):
        """Args:
        data:
            Input text to embed.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[str]],
    ):
        """Async version of __call__. Args:
        data:
            Input text to embed.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response


@register_model
class OpenAIModeration(_BaseOpenAI, ModerationModel):
    """OpenAI Moderation."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: Optional[str] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        """Args:
        model_id:
            Model ID in provider.
        base_url:
            URL to model provider.
        enable_cache:
            If True, enables response caching to avoid redundant API calls.
        cache_size:
            Maximum number of responses to cache (default: 128).
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"base_url": base_url or self._get_base_url()}
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.retry = retry
        self._initialize()
        self._get_api_key()

    def _execute_model(self, **kwargs):
        model_output = self.client.moderations.create(**kwargs)
        return model_output

    async def _aexecute_model(self, **kwargs):
        model_output = await self.aclient.moderations.create(**kwargs)
        return model_output

    def _generate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = self._execute_model(**kwargs)
        moderation = dotdict({"results": model_output.results[0].model_dump()})
        moderation.safe = not moderation.results.flagged
        response.add(moderation)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    async def _agenerate(self, **kwargs):
        # Check cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = await self._aexecute_model(**kwargs)
        moderation = dotdict({"results": model_output.results[0].model_dump()})
        moderation.safe = not moderation.results.flagged
        response.add(moderation)

        # Store in cache if enabled
        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            self._response_cache.set(cache_key, response)

        return response

    def __call__(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = self._generate(input=data, model=self.model_id)
        return response

    async def acall(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ) -> ModelResponse:
        """Async version of __call__. Args:
        data:
            Input (or inputs) to classify. Can be a single string,
            an array of strings, or an array of multi-modal input
            objects similar to other models.
        """
        response = await self._agenerate(input=data, model=self.model_id)
        return response
