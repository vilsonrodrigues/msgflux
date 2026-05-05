import pytest
from argparse import Namespace

from msgflux.channels import SocialHttpClient
from msgflux.channels.social.slack import cli as slack_cli


@pytest.mark.asyncio
async def test_slack_auth_info_returns_bot_identity(monkeypatch):
    calls = []

    async def fake_post(token, method, payload, http_client):
        calls.append((token, method, payload, http_client))
        return {
            "ok": True,
            "url": "https://example.slack.com/",
            "team": "msgflux",
            "team_id": "T123",
            "user": "msgflux-bot",
            "user_id": "U123",
            "bot_id": "B123",
        }

    monkeypatch.setattr(slack_cli, "_post_slack_api", fake_post)
    args = Namespace(
        slack_action="auth-info",
        bot_token="xoxb-token",
        bot_token_env="SLACK_BOT_TOKEN",
        timeout_s=3,
    )

    result = await slack_cli._slack_auth_info(
        slack_cli._slack_bot_token(args), args.timeout_s
    )

    assert len(calls) == 1
    token, method, payload, http_client = calls[0]
    assert (token, method, payload) == ("xoxb-token", "auth.test", {})
    assert isinstance(http_client, SocialHttpClient)
    assert http_client.config.timeout_s == 3
    assert result == {
        "ok": True,
        "url": "https://example.slack.com/",
        "team": "msgflux",
        "team_id": "T123",
        "user": "msgflux-bot",
        "user_id": "U123",
        "bot_id": "B123",
    }
