import asyncio
import json
import os
import sys
from argparse import Namespace
from typing import Any, Dict, Optional

import httpx

from msgflux.channels.env import load_env_file
from msgflux.channels.exceptions import ChannelError
from msgflux.channels.social.discord.adapter import (
    DEFAULT_DISCORD_BOT_TOKEN_ENV,
    DISCORD_API_BASE_URL,
)
from msgflux.channels.social.http import SocialHttpClient, SocialHttpConfig

DEFAULT_DISCORD_APPLICATION_ID_ENV = "DISCORD_APPLICATION_ID"


def run_discord(args: Namespace) -> int:
    load_env_file(getattr(args, "env_file", None))

    result = asyncio.run(_run_discord_action(args))
    sys.stdout.write(f"{json.dumps(result, indent=2, sort_keys=True)}\n")
    return 0


async def _run_discord_action(args: Namespace) -> Dict[str, Any]:
    action = args.discord_action
    if action == "create-ask-command":
        return await _create_ask_command(
            _discord_application_id(args),
            _discord_bot_token(args),
            getattr(args, "guild_id", None),
            getattr(args, "name", "ask"),
            getattr(args, "description", "Ask the msgFlux assistant a question."),
            getattr(args, "option_name", "prompt"),
            getattr(args, "option_description", None)
            or "Question or instruction for the assistant.",
            getattr(args, "timeout_s", 10.0),
        )
    raise ValueError(f"Unsupported Discord action `{action}`")


async def _create_ask_command(
    application_id: str,
    bot_token: str,
    guild_id: Optional[str],
    name: str,
    description: str,
    option_name: str,
    option_description: str,
    timeout_s: float,
) -> Dict[str, Any]:
    payload = {
        "name": name,
        "type": 1,
        "description": description,
        "options": [
            {
                "type": 3,
                "name": option_name,
                "description": option_description,
                "required": True,
            }
        ],
    }
    path = f"/applications/{application_id}/commands"
    scope = "global"
    if guild_id:
        path = f"/applications/{application_id}/guilds/{guild_id}/commands"
        scope = "guild"

    result = await _post_discord_api(bot_token, path, payload, timeout_s)
    return {
        "ok": True,
        "scope": scope,
        "guild_id": guild_id,
        "id": result.get("id"),
        "application_id": result.get("application_id"),
        "name": result.get("name"),
        "description": result.get("description"),
        "version": result.get("version"),
    }


async def _post_discord_api(
    bot_token: str,
    path: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    http_client = SocialHttpClient(SocialHttpConfig(timeout_s=timeout_s))
    try:
        result = await http_client.post_json(
            f"{DISCORD_API_BASE_URL}{path}",
            payload,
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "msgflux",
            },
        )
    except httpx.HTTPStatusError as e:
        raise ChannelError(
            f"Discord API request failed with HTTP "
            f"{e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.HTTPError as e:
        raise ChannelError(f"Discord API request failed: {e}") from e

    if not isinstance(result, dict):
        raise ChannelError("Discord API returned an invalid response")
    return result


def _discord_application_id(args: Namespace) -> str:
    application_id = getattr(args, "application_id", None)
    application_id_env = (
        getattr(args, "application_id_env", None) or DEFAULT_DISCORD_APPLICATION_ID_ENV
    )
    application_id = application_id or os.getenv(application_id_env, "")
    if not application_id:
        raise ChannelError("Discord application id is not configured")
    return application_id


def _discord_bot_token(args: Namespace) -> str:
    token = getattr(args, "bot_token", None)
    token_env = getattr(args, "bot_token_env", None) or DEFAULT_DISCORD_BOT_TOKEN_ENV
    token = token or os.getenv(token_env, "")
    if not token:
        raise ChannelError("Discord bot token is not configured")
    return token
