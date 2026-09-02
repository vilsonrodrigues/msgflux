from unittest.mock import patch

import pytest

from msgflux.models.chat_api import ChatAPIAdapter, ChatTransport
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion


class RecordingAPI(ChatAPIAdapter):
    name = "recording"
    endpoint = "/recording"
    canonical_history = True

    def __init__(self):
        self.calls = []

    def prepare_request(self, owner, params):
        self.calls.append(("prepare", dict(params)))
        return {**params, "prepared": True}

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

    def create(self, owner, api, params):
        self.calls.append(("create", dict(params)))
        return {"raw": params, "endpoint": api.endpoint}

    async def acreate(self, owner, api, params):
        self.calls.append(("acreate", dict(params)))
        return {"raw": params, "endpoint": api.endpoint}


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

    with (
        patch("msgflux.models.openai_compatible.OpenAI"),
        patch("msgflux.models.openai_compatible.AsyncOpenAI"),
    ):
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


def test_declared_api_mode_requires_an_adapter():
    class InvalidChatCompletion(OpenAICompatibleChatCompletion):
        default_api_mode = "missing"
        supported_api_modes = ("missing",)

    with pytest.raises(ValueError, match="without a ChatAPIAdapter"):
        InvalidChatCompletion(model_id="test-model")
