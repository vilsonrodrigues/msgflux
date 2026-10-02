"""Custom credentials do not require an unrelated environment API key."""

import pytest

from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)
from msgflux.models.providers.openai import OpenAIChatCompletion


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
def test_explicit_resolver_initializes_without_environment_key_and_resolves_lazily(
    monkeypatch, api_mode
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resolutions = []

    class Credentials(ModelCredentialResolver):
        def resolve(self, _model):
            resolutions.append("resolved")
            return ResolvedModelCredentials(
                headers={"Authorization": "Bearer selected-account"}
            )

    resolver = Credentials()
    model = OpenAIChatCompletion(
        model_id="gpt-4.1-mini", api_mode=api_mode, credential_resolver=resolver
    )
    try:
        assert resolutions == []
        assert model.credential_resolver is resolver
        assert (
            resolver.resolve(model).headers["Authorization"]
            == "Bearer selected-account"
        )
    finally:
        model.close()
