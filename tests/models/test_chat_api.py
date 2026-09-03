from unittest.mock import patch

import msgspec
import pytest

from msgflux.models import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
    OpenAIResponsesContextAdapter,
)
from msgflux.models.chat_api import (
    ChatAPIAdapter,
    ChatTransport,
    PreparedChatRequest,
    ResolvedChatCredentials,
)
from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.reasoning import OpenAICompatibleReasoningCodec


class RecordingAPI(ChatAPIAdapter):
    name = "recording"
    endpoint = "/recording"
    canonical_history = True

    def __init__(self):
        self.calls = []

    def prepare_request(self, owner, params):
        self.calls.append(("prepare", dict(params)))
        return PreparedChatRequest(
            api=self.name,
            endpoint=self.endpoint,
            params={**params, "prepared": True},
        )

    def build_generation_params(self, owner, messages, *args, **kwargs):
        self.calls.append(("build", messages))
        return {"model": owner.model_id, "input": messages}

    def process_output(self, owner, output, *args, **kwargs):
        self.calls.append(("process", output))
        return {"decoded": output}

    def stream(self, owner, **kwargs):
        self.calls.append(("stream", kwargs))
        return "streamed"

    async def astream(self, owner, **kwargs):
        self.calls.append(("astream", kwargs))
        return "astreamed"


class RecordingTransport(ChatTransport):
    def __init__(self, calls):
        self.calls = calls

    def create(self, owner, request):
        self.calls.append(("create", dict(request.params)))
        return {"raw": request.params, "endpoint": request.endpoint}

    async def acreate(self, owner, request):
        self.calls.append(("acreate", dict(request.params)))
        return {"raw": request.params, "endpoint": request.endpoint}


def _build_recording_model():
    adapter = RecordingAPI()
    transport = RecordingTransport(adapter.calls)

    class RecordingChatCompletion(OpenAICompatibleChatCompletion):
        provider = "recording"
        capabilities = ChatProviderCapabilities(
            default_api_mode="recording",
            api_modes=(ChatAPIModeCapabilities(name="recording", adapter=adapter),),
            default_reasoning_codec=OpenAICompatibleReasoningCodec(),
        )
        chat_transport = transport

        def _get_api_key(self):
            return "test"

    model = RecordingChatCompletion(model_id="test-model")
    return model, adapter


def test_custom_api_adapter_owns_the_protocol_boundary():
    model, adapter = _build_recording_model()

    params = model._build_generation_params("hello", None, None, None)
    raw = model._execute_model(**params)
    decoded = model._process_model_output(raw)
    streamed = model._stream_generate(model=model.model_id)

    assert model._uses_canonical_history is True
    assert params == {"model": "test-model", "input": "hello"}
    assert raw["raw"]["prepared"] is True
    assert raw["endpoint"] == "/recording"
    assert decoded == {"decoded": raw}
    assert streamed == "streamed"
    assert [name for name, _ in adapter.calls] == [
        "build",
        "prepare",
        "create",
        "process",
        "stream",
    ]


@pytest.mark.asyncio
async def test_custom_api_adapter_supports_async_transport_and_streaming():
    model, adapter = _build_recording_model()

    raw = await model._aexecute_model(model=model.model_id, input="hello")
    streamed = await model._astream_generate(model=model.model_id)

    assert raw["raw"]["prepared"] is True
    assert streamed == "astreamed"
    assert [name for name, _ in adapter.calls] == [
        "prepare",
        "acreate",
        "astream",
    ]


def test_chat_runtime_does_not_create_sdk_clients():
    from msgflux.models.chat_transport import HTTPChatTransport

    class DirectChatCompletion(OpenAICompatibleChatCompletion):
        provider = "direct"

        def _get_api_key(self):
            return "test"

    model = DirectChatCompletion(model_id="test-model")

    assert isinstance(model.chat_transport, HTTPChatTransport)
    assert not hasattr(model, "client")
    assert not hasattr(model, "aclient")


def test_chat_runtime_strategies_are_not_serialized():
    model, _ = _build_recording_model()

    state = model.serialize()["state"]

    assert "api_adapter" not in state
    assert "api_mode_capabilities" not in state
    assert "reasoning_codec" not in state
    msgspec.json.encode(model.serialize())


def test_chat_capabilities_require_a_declared_default_mode():
    with pytest.raises(ValueError, match="Default API mode 'missing'"):
        ChatProviderCapabilities(
            default_api_mode="missing",
            api_modes=(
                ChatAPIModeCapabilities(
                    name="recording",
                    adapter=RecordingAPI(),
                ),
            ),
            default_reasoning_codec=OpenAICompatibleReasoningCodec(),
        )


def test_chat_capabilities_reject_duplicate_api_modes():
    with pytest.raises(ValueError, match="API mode names must be unique"):
        ChatProviderCapabilities(
            default_api_mode="recording",
            api_modes=(
                ChatAPIModeCapabilities(
                    name="recording",
                    adapter=RecordingAPI(),
                ),
                ChatAPIModeCapabilities(
                    name="recording",
                    adapter=RecordingAPI(),
                ),
            ),
            default_reasoning_codec=OpenAICompatibleReasoningCodec(),
        )


def test_chat_capabilities_require_a_typed_context_adapter():
    with pytest.raises(TypeError, match="ChatContextAdapter instance"):
        ChatAPIModeCapabilities(
            name="recording",
            adapter=RecordingAPI(),
            context_adapter=object(),
        )


def test_chat_capabilities_reject_context_adapter_for_another_api_mode():
    with pytest.raises(ValueError, match="cannot be attached"):
        ChatAPIModeCapabilities(
            name="chat_completions",
            adapter=RecordingAPI(),
            context_adapter=OpenAIResponsesContextAdapter(),
        )


def test_responses_context_adapter_is_reusable_by_compatible_providers():
    calls = []

    class ContextTransport(ChatTransport):
        def create(self, owner, request):
            calls.append(request)
            if request.endpoint == "/responses/input_tokens":
                return {"input_tokens": 37}
            return {
                "output": [{"type": "compaction", "encrypted_content": "opaque"}],
                "usage": {"input_tokens": 37, "output_tokens": 4},
            }

    class CompatibleResponsesModel(OpenAICompatibleChatCompletion):
        provider = "compatible_context"
        capabilities = ChatProviderCapabilities(
            default_api_mode="responses",
            api_modes=(
                ChatAPIModeCapabilities(
                    name="responses",
                    adapter=OpenAIResponsesAPI(),
                    context_adapter=OpenAIResponsesContextAdapter(),
                ),
            ),
            default_reasoning_codec=OpenAICompatibleReasoningCodec(),
        )
        chat_transport = ContextTransport()

        def _get_api_key(self):
            return "test"

    model = CompatibleResponsesModel(model_id="compatible-model")
    estimate = model.count_context_tokens(
        [{"role": "user", "content": "Remember this"}],
        system_prompt="Preserve facts.",
    )
    compacted = model.compact_context(
        [{"role": "user", "content": "Remember this"}],
        system_prompt="Preserve facts.",
    )

    assert model.supports_native_compaction() is True
    assert estimate.input_tokens == 37
    assert compacted.provider == "compatible_context"
    assert compacted.model_id == "compatible-model"
    assert compacted.items == [{"type": "compaction", "encrypted_content": "opaque"}]
    assert [request.endpoint for request in calls] == [
        "/responses/input_tokens",
        "/responses/compact",
    ]
    assert all(request.api == "responses" for request in calls)


def test_chat_model_requires_typed_capabilities():
    class InvalidChatCompletion(OpenAICompatibleChatCompletion):
        capabilities = {"default_api_mode": "chat_completions"}

    with pytest.raises(TypeError, match="ChatProviderCapabilities instance"):
        InvalidChatCompletion(model_id="test-model")


def test_prepared_request_expands_sdk_extensions_for_direct_http():
    request = PreparedChatRequest(
        api="chat_completions",
        endpoint="/chat/completions",
        params={
            "model": "test-model",
            "messages": [],
            "extra_body": {"reasoning": {"effort": "high"}},
            "extra_headers": {"X-Test": "value"},
        },
    )

    assert request.params["extra_body"] == {"reasoning": {"effort": "high"}}
    assert request.json == {
        "model": "test-model",
        "messages": [],
        "reasoning": {"effort": "high"},
    }
    assert request.headers == {"X-Test": "value"}
    assert "test-model" not in repr(request)


def test_resolved_credentials_do_not_expose_headers_in_repr():
    credentials = ResolvedChatCredentials(
        headers={"Authorization": "Bearer secret-token"},
        base_url="https://secret-account.example.com",
    )

    assert "secret-token" not in repr(credentials)
    assert "secret-account" not in repr(credentials)
