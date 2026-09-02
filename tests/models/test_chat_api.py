from unittest.mock import patch

import pytest

from msgflux.models.chat_api import (
    ChatAPIAdapter,
    ChatTransport,
    PreparedChatRequest,
    ResolvedChatCredentials,
)
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion


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
        default_api_mode = "recording"
        supported_api_modes = ("recording",)
        api_adapters = {"recording": adapter}
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


def test_chat_runtime_does_not_load_or_create_openai_sdk_clients():
    from msgflux.models.chat_transport import HTTPChatTransport

    class DirectChatCompletion(OpenAICompatibleChatCompletion):
        provider = "direct"

        def _get_api_key(self):
            return "test"

    with patch("msgflux.models.openai_sdk._load_openai_sdk") as load_sdk:
        model = DirectChatCompletion(model_id="test-model")

    load_sdk.assert_not_called()
    assert isinstance(model.chat_transport, HTTPChatTransport)
    assert not hasattr(model, "client")
    assert not hasattr(model, "aclient")


def test_declared_api_mode_requires_an_adapter():
    class InvalidChatCompletion(OpenAICompatibleChatCompletion):
        default_api_mode = "missing"
        supported_api_modes = ("missing",)

    with pytest.raises(ValueError, match="without a ChatAPIAdapter"):
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
        headers={"Authorization": "Bearer secret-token"}
    )

    assert "secret-token" not in repr(credentials)
