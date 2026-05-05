import pytest
from argparse import Namespace

from msgflux.channels.social.discord import cli as discord_cli


@pytest.mark.asyncio
async def test_discord_create_ask_command_returns_summary(monkeypatch):
    calls = []

    async def fake_post(token, path, payload, timeout_s):
        calls.append((token, path, payload, timeout_s))
        return {
            "id": "C123",
            "application_id": "A123",
            "name": payload["name"],
            "description": payload["description"],
            "version": "V123",
        }

    monkeypatch.setattr(discord_cli, "_post_discord_api", fake_post)

    result = await discord_cli._create_ask_command(
        "A123",
        "bot-token",
        "G123",
        "ask",
        "Ask a question.",
        "prompt",
        "Question text.",
        3,
    )

    assert calls == [
        (
            "bot-token",
            "/applications/A123/guilds/G123/commands",
            {
                "name": "ask",
                "type": 1,
                "description": "Ask a question.",
                "options": [
                    {
                        "type": 3,
                        "name": "prompt",
                        "description": "Question text.",
                        "required": True,
                    }
                ],
            },
            3,
        )
    ]
    assert result == {
        "ok": True,
        "scope": "guild",
        "guild_id": "G123",
        "id": "C123",
        "application_id": "A123",
        "name": "ask",
        "description": "Ask a question.",
        "version": "V123",
    }
