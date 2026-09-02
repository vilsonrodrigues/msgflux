import tempfile
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import msgflux.nn.functional as F
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.models.cache import generate_cache_key
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion as _OpenAICompatibleChatCompletion,
)
from msgflux.models.openai_compatible import (
    OpenAICompatibleModel,
)
from msgflux.models.reasoning import (
    OpenAIReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.types import (
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextToImageModel,
    TextToSpeechModel,
)
from msgflux.tools.catalog import ToolCatalogView
from msgflux.utils.encode import encode_data_to_bytes


@register_model
class OpenAIChatCompletion(_OpenAICompatibleChatCompletion):
    """OpenAI Chat Completions provider."""

    hosted_tool_search_model_families = (
        "gpt-5.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    )
    provider = "openai"
    chat_transport = HTTPChatTransport
    default_api_mode = "responses"
    default_reasoning_codec = OpenAIReasoningCodec()
    supported_api_modes = ("responses", "chat_completions")
    reasoning_codecs = {"responses": OpenAIResponsesReasoningCodec()}
    supports_init_logprobs = True
    supports_prompt_cache_retention = True
    uses_max_completion_tokens = True
    responses_supports_reasoning_summary = True
    responses_supports_encrypted_reasoning = True

    def supports_native_compaction(self) -> bool:
        return self.api_mode == "responses"

    def _responses_compaction_input(
        self,
        messages: Union[str, List[Dict[str, Any]], ChatMessages],
    ) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, ChatMessages):
            messages = ChatMessages(messages)
        return messages.to_responses_input(
            provider=self.provider,
            api_mode=self.api_mode,
            reasoning_codec=self.reasoning_codec,
        )

    def count_context_tokens(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> ContextTokenEstimate:
        if self.api_mode != "responses":
            return super().count_context_tokens(
                messages,
                system_prompt=system_prompt,
                tool_catalog=tool_catalog,
            )
        params = {
            "model": self.model_id,
            "input": self._responses_compaction_input(messages),
            "instructions": system_prompt,
        }
        if tool_catalog and self._catalog_tool_entries(tool_catalog):
            params["tools"] = self._tools_to_responses(tool_catalog)
        result = self.client.responses.input_tokens.count(**params)
        return ContextTokenEstimate(
            input_tokens=int(result.input_tokens),
            source="provider",
        )

    async def acount_context_tokens(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> ContextTokenEstimate:
        if self.api_mode != "responses":
            return await super().acount_context_tokens(
                messages,
                system_prompt=system_prompt,
                tool_catalog=tool_catalog,
            )
        params = {
            "model": self.model_id,
            "input": self._responses_compaction_input(messages),
            "instructions": system_prompt,
        }
        if tool_catalog and self._catalog_tool_entries(tool_catalog):
            params["tools"] = self._tools_to_responses(tool_catalog)
        result = await self.aclient.responses.input_tokens.count(**params)
        return ContextTokenEstimate(
            input_tokens=int(result.input_tokens),
            source="provider",
        )

    def compact_context(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        system_prompt: str | None = None,
        native: bool = True,
    ) -> ModelCompaction:
        if self.api_mode != "responses" or not native:
            return super().compact_context(
                messages,
                system_prompt=system_prompt,
                native=False,
            )
        output = self.client.responses.compact(
            model=self.model_id,
            input=self._responses_compaction_input(messages),
            instructions=system_prompt,
        )
        return self._native_model_compaction(output)

    async def acompact_context(
        self,
        messages: Union[ChatMessages, List[Mapping[str, Any]]],
        *,
        system_prompt: str | None = None,
        native: bool = True,
    ) -> ModelCompaction:
        if self.api_mode != "responses" or not native:
            return await super().acompact_context(
                messages,
                system_prompt=system_prompt,
                native=False,
            )
        output = await self.aclient.responses.compact(
            model=self.model_id,
            input=self._responses_compaction_input(messages),
            instructions=system_prompt,
        )
        return self._native_model_compaction(output)

    def _native_model_compaction(self, output: Any) -> ModelCompaction:
        serialized_items = self._serialize_openai_value(output.output)
        usage = self.usage_codec.normalize(getattr(output, "usage", None))
        return ModelCompaction(
            format="provider",
            items=serialized_items,
            provider=self.provider,
            api_mode=self.api_mode,
            model_id=self.model_id,
            usage=dict(usage) if isinstance(usage, Mapping) else None,
        )


@register_model
class OpenAITextToSpeech(OpenAICompatibleModel, TextToSpeechModel):
    """OpenAI Text to Speech."""

    def __init__(
        self,
        model_id: str,
        voice: Optional[str] = "alloy",
        speed: Optional[float] = 1.0,
        stream_chunk_size: int = 1024,
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
        stream_chunk_size:
            Number of bytes yielded per streaming audio chunk.
        base_url:
            URL to model provider.
        retry:
            Retry config. A tenacity decorator, False to disable, or None for default.
        """
        super().__init__()
        if not isinstance(stream_chunk_size, int) or stream_chunk_size <= 0:
            raise ValueError("`stream_chunk_size` must be a positive integer")
        self.model_id = model_id
        self.stream_chunk_size = stream_chunk_size
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
                await model_output.stream_to_file(temp_file_path)

            response.set_response_type("audio_generation")
            response.add(temp_file_path)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        try:
            with self._execute_model(**kwargs) as model_output:
                for chunk in model_output.iter_bytes(chunk_size=self.stream_chunk_size):
                    stream_response.add(chunk)
                    if not stream_response.first_chunk_event.is_set():
                        stream_response.first_chunk_event.set()
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("audio_generation")

        try:
            async with self._aexecute_model(**kwargs) as model_output:
                async for chunk in model_output.iter_bytes(
                    chunk_size=self.stream_chunk_size
                ):
                    stream_response.add(chunk)
                    if not stream_response.first_chunk_event.is_set():
                        stream_response.first_chunk_event.set()
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

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
            F.detached(self._stream_generate, **params)
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
            await F.adetached(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextToImage(OpenAICompatibleModel, TextToImageModel):
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
class OpenAISpeechToText(OpenAICompatibleModel, SpeechToTextModel):
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
            words = getattr(model_output, "words", None)
            if words:
                transcript["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end} for w in words
                ]
            segments = getattr(model_output, "segments", None)
            if segments:
                transcript["segments"] = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]

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
            words = getattr(model_output, "words", None)
            if words:
                transcript["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end} for w in words
                ]
            segments = getattr(model_output, "segments", None)
            if segments:
                transcript["segments"] = [
                    {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        try:
            model_output = self._execute_model(**kwargs)

            for event in model_output:
                if event.type == "transcript.text.delta":
                    chunk = event.delta
                    if chunk:
                        stream_response.add(chunk)
                        if not stream_response.first_chunk_event.is_set():
                            stream_response.first_chunk_event.set()
                elif event.type == "transcript.text.done":
                    break
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

        return stream_response

    async def _astream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        try:
            model_output = await self._aexecute_model(**kwargs)

            async for event in model_output:
                if event.type == "transcript.text.delta":
                    chunk = event.delta
                    if chunk:
                        stream_response.add(chunk)
                        if not stream_response.first_chunk_event.is_set():
                            stream_response.first_chunk_event.set()
                elif event.type == "transcript.text.done":
                    break
        except Exception as exc:
            stream_response.finish(error=exc, status="failed")
        else:
            stream_response.finish(status="completed")

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
            F.detached(self._stream_generate, **params)
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
            await F.adetached(self._astream_generate, **params)
            await F.await_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = await self._agenerate(**params)
            return response


@register_model
class OpenAITextEmbedder(OpenAICompatibleModel, TextEmbedderModel):
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
class OpenAIModeration(OpenAICompatibleModel, ModerationModel):
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
