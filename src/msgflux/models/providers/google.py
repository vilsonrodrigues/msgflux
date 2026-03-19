import json
from os import getenv
from typing import Any, Dict, List, Mapping, Optional, Union

import msgspec

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

from msgflux.chat_messages import ChatMessages
from msgflux.dotdict import dotdict
from msgflux.dsl.typed_parsers import typed_parser_registry
from msgflux.exceptions import TypedParserNotFoundError
from msgflux.models.base import BaseModel
from msgflux.models.cache import ResponseCache, generate_cache_key
from msgflux.models.profiles import get_model_profile
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.types import ChatCompletionModel
from msgflux.nn import functional as F
from msgflux.utils.chat import ChatBlock, response_format_from_msgspec_struct
from msgflux.utils.console import cprint
from msgflux.utils.msgspec import struct_to_dict
from msgflux.utils.tenacity import apply_retry, default_model_retry


def _openai_tools_to_genai(
    tool_schemas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert OpenAI tool schemas to Google GenAI function declarations.

    OpenAI format:
        [{"type": "function", "function": {"name": ..., "description": ...,
          "parameters": ...}}]

    GenAI format:
        [{"function_declarations": [{"name": ..., "description": ...,
          "parameters": ...}]}]
    """
    declarations = []
    for tool in tool_schemas:
        func = tool.get("function", {})
        decl: Dict[str, Any] = {"name": func.get("name", "")}
        if func.get("description"):
            decl["description"] = func["description"]
        params = func.get("parameters")
        if params:
            decl["parameters"] = _convert_json_schema_to_genai(params)
        declarations.append(decl)
    return [{"function_declarations": declarations}]


def _convert_json_schema_to_genai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON Schema to Google GenAI Schema format.

    Removes unsupported keys like 'additionalProperties' and recursively
    converts nested schemas.
    """
    result: Dict[str, Any] = {}

    type_val = schema.get("type")
    if type_val:
        result["type"] = type_val.upper()

    for key in ("description", "enum", "required"):
        if key in schema:
            result[key] = schema[key]

    if "properties" in schema:
        result["properties"] = {
            k: _convert_json_schema_to_genai(v) for k, v in schema["properties"].items()
        }

    if "items" in schema:
        result["items"] = _convert_json_schema_to_genai(schema["items"])

    return result


def _genai_tool_choice(
    tool_choice: Optional[Union[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Convert OpenAI tool_choice to Google GenAI FunctionCallingConfig."""
    if tool_choice is None or tool_choice == "auto":
        return {"function_calling_config": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"function_calling_config": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"function_calling_config": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        if name:
            return {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": [name],
                }
            }
    if isinstance(tool_choice, str):
        return {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tool_choice],
            }
        }
    return None


class _BaseGoogle(BaseModel):
    provider: str = "google"

    def _initialize(self):
        if genai is None:
            raise ImportError(
                "`google-genai` client is not available. "
                "Install with `pip install google-genai`."
            )
        self.client = genai.Client(api_key=self._get_api_key())
        self.aclient = genai.Client(api_key=self._get_api_key())

        cache_size = getattr(self, "cache_size", 128)
        enable_cache = getattr(self, "enable_cache", None)
        self._response_cache = (
            ResponseCache(maxsize=cache_size) if enable_cache else None
        )

        retry_config = getattr(self, "retry", None)
        self.__call__ = apply_retry(
            self.__call__, retry_config, default=default_model_retry
        )
        self.acall = apply_retry(self.acall, retry_config, default=default_model_retry)

    def _get_api_key(self):
        key = getenv("GEMINI_API_KEY") or getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "The Google API key is not available. "
                "Please set `GEMINI_API_KEY` or `GOOGLE_API_KEY`."
            )
        return key

    @property
    def profile(self):
        return get_model_profile(self.model_id, provider_id=self.provider)


@register_model
class GoogleChatCompletion(_BaseGoogle, ChatCompletionModel):
    """Google Gemini Chat Completion."""

    def __init__(
        self,
        model_id: str,
        *,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        return_reasoning: Optional[bool] = True,
        reasoning_in_tool_call: Optional[bool] = True,
        validate_typed_parser_output: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        stop: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        context_length: Optional[int] = None,
        enable_cache: Optional[bool] = False,
        cache_size: Optional[int] = 128,
        retry: Optional[Any] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.context_length = context_length
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.return_reasoning = return_reasoning
        self.reasoning_in_tool_call = reasoning_in_tool_call
        self.validate_typed_parser_output = validate_typed_parser_output
        self.verbose = verbose
        self.retry = retry

        self.sampling_run_params: Dict[str, Any] = {}
        if max_tokens is not None:
            self.sampling_run_params["max_output_tokens"] = max_tokens
        if temperature is not None:
            self.sampling_run_params["temperature"] = temperature
        if top_p is not None:
            self.sampling_run_params["top_p"] = top_p
        if top_k is not None:
            self.sampling_run_params["top_k"] = top_k
        if stop:
            self.sampling_run_params["stop_sequences"] = stop

        self._initialize()

    def _ensure_chat_messages(
        self, messages: Union[ChatMessages, List[Dict[str, Any]], None]
    ) -> ChatMessages:
        if isinstance(messages, ChatMessages):
            return messages.copy()
        if messages is None:
            return ChatMessages()
        return ChatMessages.from_chatml(messages)

    def _build_config(
        self,
        *,
        system_instruction: Optional[str] = None,
        generation_schema: Optional[type] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build GenerateContentConfig dict."""
        config: Dict[str, Any] = {**self.sampling_run_params}

        if system_instruction:
            config["system_instruction"] = system_instruction

        # Thinking / reasoning
        if self.enable_thinking:
            thinking_config: Dict[str, Any] = {"include_thoughts": True}
            if self.thinking_budget is not None:
                thinking_config["thinking_budget"] = self.thinking_budget
            config["thinking_config"] = thinking_config

        # Structured output — reuse the shared msgspec→JSON Schema converter
        if generation_schema is not None:
            config["response_mime_type"] = "application/json"
            fmt = response_format_from_msgspec_struct(generation_schema)
            config["response_json_schema"] = fmt["json_schema"]["schema"]

        # Tools
        if tool_schemas:
            config["tools"] = _openai_tools_to_genai(tool_schemas)
            tc = _genai_tool_choice(tool_choice)
            if tc:
                config["tool_config"] = tc

        return config

    def _execute_model(self, **kwargs):
        prefilling = kwargs.pop("prefilling", None)
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)
        system_prompt = kwargs.pop("system_instruction", None)
        generation_schema = kwargs.pop("generation_schema", None)
        tool_schemas = kwargs.pop("tool_schemas", None)
        tool_choice = kwargs.pop("tool_choice", None)

        system_from_messages, contents = messages.to_genai()

        # system_prompt goes directly as config param (not in contents).
        # If messages also contained system/developer roles, merge them.
        if system_prompt and system_from_messages:
            system_instruction = f"{system_prompt}\n\n{system_from_messages}"
        else:
            system_instruction = system_prompt or system_from_messages

        config = self._build_config(
            system_instruction=system_instruction,
            generation_schema=generation_schema,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
        )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        return response

    async def _aexecute_model(self, **kwargs):
        prefilling = kwargs.pop("prefilling", None)
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)
        system_prompt = kwargs.pop("system_instruction", None)
        generation_schema = kwargs.pop("generation_schema", None)
        tool_schemas = kwargs.pop("tool_schemas", None)
        tool_choice = kwargs.pop("tool_choice", None)

        system_from_messages, contents = messages.to_genai()

        if system_prompt and system_from_messages:
            system_instruction = f"{system_prompt}\n\n{system_from_messages}"
        else:
            system_instruction = system_prompt or system_from_messages

        config = self._build_config(
            system_instruction=system_instruction,
            generation_schema=generation_schema,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
        )

        response = await self.aclient.aio.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=config,
        )
        return response

    def _process_model_output(  # noqa: C901
        self, model_output, typed_parser=None, generation_schema=None
    ):
        response = ModelResponse()
        metadata = dotdict()

        # Usage metadata
        usage = model_output.usage_metadata
        if usage:
            metadata.usage = {
                "prompt_tokens": usage.prompt_token_count or 0,
                "completion_tokens": usage.candidates_token_count or 0,
                "total_tokens": usage.total_token_count or 0,
            }
            if usage.thoughts_token_count:
                metadata.usage["reasoning_tokens"] = usage.thoughts_token_count

        candidate = model_output.candidates[0]
        content = candidate.content

        # Extract reasoning (thought parts), regular parts, and function calls
        reasoning_parts = []
        text_parts = []
        function_calls = []
        # Preserve full parts list for thought_signature round-trip
        raw_parts = []

        if content and content.parts:
            for part in content.parts:
                if getattr(part, "thought", False) and part.text:
                    reasoning_parts.append(part.text)
                elif part.function_call:
                    function_calls.append(part)
                elif part.text is not None:
                    text_parts.append(part.text)
                # Keep all parts (including thought_signature) for round-trip
                raw_parts.append(part)

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        reasoning_tool_call = None
        if self.reasoning_in_tool_call is True:
            reasoning_tool_call = reasoning

        prefix_response_type = ""
        reasoning_content = None
        if self.return_reasoning is True:
            reasoning_content = reasoning
            if reasoning_content is not None:
                prefix_response_type = "reasoning_"

        if function_calls:
            aggregator = ToolCallAggregator(reasoning_tool_call)
            response.set_response_type("tool_call")
            for call_index, part in enumerate(function_calls):
                fc = part.function_call
                tool_id = getattr(fc, "id", None) or f"call_{call_index}"
                name = fc.name or ""
                arguments = json.dumps(fc.args) if fc.args else "{}"
                aggregator.process(call_index, tool_id, name, arguments)
                # Preserve thought_signature for Gemini round-trip
                sig = getattr(part, "thought_signature", None)
                if sig is not None:
                    aggregator.tool_calls[call_index]["thought_signature"] = sig
            # Store raw model parts for exact round-trip in ChatMessages
            aggregator.genai_raw_parts = raw_parts
            response_content = aggregator
        elif text_parts:
            text = "".join(text_parts)
            if (typed_parser or generation_schema) and self.verbose:
                repr_str = f"[{self.model_id}][raw_response] {text}"
                cprint(repr_str, lc="r", ls="b")
            if typed_parser is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                parser = typed_parser_registry[typed_parser]
                response_content = dotdict(parser.decode(text))
                if generation_schema and self.validate_typed_parser_output:
                    encoded = msgspec.json.encode(response_content)
                    msgspec.json.decode(encoded, type=generation_schema)
            elif generation_schema is not None:
                response.set_response_type(f"{prefix_response_type}structured")
                struct = msgspec.json.decode(text.encode(), type=generation_schema)
                response_content = dotdict(struct_to_dict(struct))
            else:
                response.set_response_type(f"{prefix_response_type}text_generation")
                if reasoning_content is not None:
                    response_content = dotdict({"answer": text})
                else:
                    response_content = text
        else:
            response.set_response_type("text_generation")
            response_content = ""

        if reasoning_content is not None:
            response_content.think = reasoning_content

        response.add(response_content)
        response.set_metadata(metadata)
        return response

    def _generate(self, **kwargs: Mapping[str, Any]) -> ModelResponse:
        typed_parser = kwargs.get("typed_parser")
        generation_schema = kwargs.get("generation_schema")

        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        # generation_schema must reach _execute_model for response_json_schema
        model_output = self._execute_model(
            **kwargs, generation_schema=generation_schema
        )
        response = self._process_model_output(
            model_output, typed_parser, generation_schema
        )

        if self.enable_cache and self._response_cache:
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

        if self.enable_cache and self._response_cache:
            cache_key = generate_cache_key(**kwargs)
            hit, cached_response = self._response_cache.get(cache_key)
            if hit:
                return cached_response

        typed_parser = kwargs.pop("typed_parser")
        generation_schema = kwargs.pop("generation_schema")

        model_output = await self._aexecute_model(
            **kwargs, generation_schema=generation_schema
        )
        response = self._process_model_output(
            model_output, typed_parser, generation_schema
        )

        if self.enable_cache and self._response_cache:
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
    ) -> None:
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")
        prefilling = kwargs.pop("prefilling", None)
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)
        system_prompt = kwargs.pop("system_instruction", None)
        tool_schemas = kwargs.pop("tool_schemas", None)
        tool_choice = kwargs.pop("tool_choice", None)

        system_from_messages, contents = messages.to_genai()
        if system_prompt and system_from_messages:
            system_instruction = f"{system_prompt}\n\n{system_from_messages}"
        else:
            system_instruction = system_prompt or system_from_messages

        config = self._build_config(
            system_instruction=system_instruction,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
        )

        reasoning_tool_call = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.model_id,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, "thought", False) and part.text:
                            if self.reasoning_in_tool_call:
                                reasoning_tool_call += part.text
                            if self.return_reasoning:
                                if stream_response.response_type is None:
                                    stream_response.set_response_type(
                                        "reasoning_text_generation"
                                    )
                                    stream_response.first_chunk_event.set()
                                stream_response.add(part.text)
                        elif part.function_call:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("tool_call")
                            fc = part.function_call
                            idx = len(aggregator.tool_calls)
                            tid = getattr(fc, "id", None) or f"call_{idx}"
                            args = json.dumps(fc.args) if fc.args else "{}"
                            aggregator.process(idx, tid, fc.name or "", args)
                            sig = getattr(part, "thought_signature", None)
                            if sig is not None:
                                aggregator.tool_calls[idx]["thought_signature"] = sig
                        elif part.text:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("text_generation")
                                stream_response.first_chunk_event.set()
                            stream_response.add(part.text)

            if chunk.usage_metadata:
                usage = chunk.usage_metadata
                metadata.usage = {
                    "prompt_tokens": usage.prompt_token_count or 0,
                    "completion_tokens": usage.candidates_token_count or 0,
                    "total_tokens": usage.total_token_count or 0,
                }
                if usage.thoughts_token_count:
                    metadata.usage["reasoning_tokens"] = usage.thoughts_token_count

        if aggregator.tool_calls:
            if reasoning_tool_call:
                aggregator.reasoning = reasoning_tool_call
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    async def _astream_generate(  # noqa: C901
        self, **kwargs: Mapping[str, Any]
    ) -> None:
        aggregator = ToolCallAggregator()
        metadata = dotdict()

        stream_response = kwargs.pop("stream_response")
        prefilling = kwargs.pop("prefilling", None)
        messages = self._ensure_chat_messages(kwargs.pop("messages", None))
        if prefilling:
            messages.add_assistant(prefilling)
        system_prompt = kwargs.pop("system_instruction", None)
        tool_schemas = kwargs.pop("tool_schemas", None)
        tool_choice = kwargs.pop("tool_choice", None)

        system_from_messages, contents = messages.to_genai()
        if system_prompt and system_from_messages:
            system_instruction = f"{system_prompt}\n\n{system_from_messages}"
        else:
            system_instruction = system_prompt or system_from_messages

        config = self._build_config(
            system_instruction=system_instruction,
            tool_schemas=tool_schemas,
            tool_choice=tool_choice,
        )

        reasoning_tool_call = ""

        async for chunk in await self.aclient.aio.models.generate_content_stream(
            model=self.model_id,
            contents=contents,
            config=config,
        ):
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if getattr(part, "thought", False) and part.text:
                            if self.reasoning_in_tool_call:
                                reasoning_tool_call += part.text
                            if self.return_reasoning:
                                if stream_response.response_type is None:
                                    stream_response.set_response_type(
                                        "reasoning_text_generation"
                                    )
                                    stream_response.first_chunk_event.set()
                                stream_response.add(part.text)
                        elif part.function_call:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("tool_call")
                            fc = part.function_call
                            idx = len(aggregator.tool_calls)
                            tid = getattr(fc, "id", None) or f"call_{idx}"
                            args = json.dumps(fc.args) if fc.args else "{}"
                            aggregator.process(idx, tid, fc.name or "", args)
                            sig = getattr(part, "thought_signature", None)
                            if sig is not None:
                                aggregator.tool_calls[idx]["thought_signature"] = sig
                        elif part.text:
                            if stream_response.response_type is None:
                                stream_response.set_response_type("text_generation")
                                stream_response.first_chunk_event.set()
                            stream_response.add(part.text)

            if chunk.usage_metadata:
                usage = chunk.usage_metadata
                metadata.usage = {
                    "prompt_tokens": usage.prompt_token_count or 0,
                    "completion_tokens": usage.candidates_token_count or 0,
                    "total_tokens": usage.total_token_count or 0,
                }
                if usage.thoughts_token_count:
                    metadata.usage["reasoning_tokens"] = usage.thoughts_token_count

        if aggregator.tool_calls:
            if reasoning_tool_call:
                aggregator.reasoning = reasoning_tool_call
            stream_response.data = aggregator
            stream_response.first_chunk_event.set()

        stream_response.set_metadata(metadata)
        stream_response.add(None)

    def __call__(
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
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        elif isinstance(messages, ChatMessages):
            messages = messages.to_chatml()

        if isinstance(tool_choice, str):
            if tool_choice not in ["auto", "required", "none"]:
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "model": self.model_id,
        }

        if system_prompt:
            generation_params["system_instruction"] = system_prompt

        if tool_schemas:
            generation_params["tool_schemas"] = tool_schemas
            generation_params["tool_choice"] = tool_choice

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse()
            F.fire_and_forget(
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

    async def acall(
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
        if isinstance(messages, str):
            messages = [ChatBlock.user(messages)]
        elif isinstance(messages, ChatMessages):
            messages = messages.to_chatml()

        if isinstance(tool_choice, str):
            if tool_choice not in ["auto", "required", "none"]:
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }

        generation_params = {
            "messages": messages,
            "prefilling": prefilling,
            "model": self.model_id,
        }

        if system_prompt:
            generation_params["system_instruction"] = system_prompt

        if tool_schemas:
            generation_params["tool_schemas"] = tool_schemas
            generation_params["tool_choice"] = tool_choice

        if stream is True:
            if typed_parser is not None:
                raise ValueError("`typed_parser` is not `stream=True` compatible")

            stream_response = ModelStreamResponse()
            await F.afire_and_forget(
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
