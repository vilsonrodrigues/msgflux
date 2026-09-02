"""OpenAI SDK runtime for model capabilities that still require the SDK."""

from importlib import import_module
from os import getenv
from typing import Any

import httpx2

from msgflux.models.openai_compatible import OpenAICompatibleModel


def _load_openai_sdk():
    try:
        openai_module = import_module("openai")
    except ImportError as exc:
        raise ImportError(
            "`openai` client is not available. "
            "Install with `pip install msgflux[openai]`."
        ) from exc

    try:
        instrumentation = import_module("opentelemetry.instrumentation.openai")
    except ImportError:
        instrumentor_type = None
    else:
        instrumentor_type = instrumentation.OpenAIInstrumentor

    if instrumentor_type is not None and not getattr(
        openai_module, "_otel_instrumented", False
    ):
        instrumentor_type().instrument()
        openai_module._otel_instrumented = True
    return (
        openai_module,
        openai_module.OpenAI,
        openai_module.AsyncOpenAI,
    )


def create_openai_sdk_client(owner: Any, *, async_client: bool = False):
    """Create an OpenAI SDK client for a model that explicitly needs one."""
    openai_module, sync_client_type, async_client_type = _load_openai_sdk()
    client_type = async_client_type if async_client else sync_client_type
    max_retries = getenv("OPENAI_MAX_RETRIES", openai_module.DEFAULT_MAX_RETRIES)
    timeout = getenv("OPENAI_TIMEOUT")
    verify_ssl = getenv("OPENAI_SSL_VERIFY", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    http_client_type = httpx2.AsyncClient if async_client else httpx2.Client
    return client_type(
        **owner.sampling_params,
        api_key=owner._get_api_key(),
        timeout=timeout,
        max_retries=max_retries,
        http_client=http_client_type(
            limits=httpx2.Limits(
                max_connections=1000,
                max_keepalive_connections=100,
            ),
            verify=verify_ssl,
        ),
    )


class OpenAISDKModel(OpenAICompatibleModel):
    """Base for non-chat capabilities implemented through the OpenAI SDK."""

    def _initialize(self):
        self.current_key_index = 0
        self.client = create_openai_sdk_client(self)
        self.aclient = create_openai_sdk_client(self, async_client=True)
        self._initialize_runtime()
