"""Configuration behavior that affects the coding host."""

import pytest
import msgspec

from msgflux.coding.config import ToolSelection, load_config, select_tools


def test_select_tools_preserves_profile_when_no_cli_values():
    profile = ToolSelection(active=("workspace",), deferred=("web_search",))
    assert select_tools(profile) is profile


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"active": " workspace, agents "},
            ToolSelection(active=("workspace", "agents")),
        ),
        ({"deferred": " web_search "}, ToolSelection(deferred=("web_search",))),
        (
            {"active": "workspace", "deferred": "web_search"},
            ToolSelection(active=("workspace",), deferred=("web_search",)),
        ),
        ({"active": ""}, ToolSelection()),
        ({"deferred": "   "}, ToolSelection()),
    ],
)
def test_select_tools_replaces_both_profile_lists(kwargs, expected):
    profile = ToolSelection(active=("from_profile",), deferred=("also_profile",))
    assert select_tools(profile, **kwargs) == expected


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"active": "workspace,,agents"}, "empty entries"),
        ({"deferred": "workspace,"}, "empty entries"),
        ({"active": "workspace,workspace"}, "duplicate tools"),
        ({"active": "workspace", "deferred": "workspace"}, "active and deferred"),
    ],
)
def test_select_tools_rejects_invalid_lists(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_tools(ToolSelection(), **kwargs)


def test_load_user_profiles_and_subagent_models(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        'default_model = "openai/gpt-6-sol"\n'
        'reasoning_effort = "high"\n'
        'default_profile = "full"\n'
        '[active_accounts]\nopenai = "personal"\n'
        "[profiles.full.tools]\n"
        'active = ["workspace", "agents"]\n'
        'deferred = ["web_search"]\n'
        "[agents.explorer]\n"
        'models = ["openai/gpt-6-luna", "anthropic/opus-5.5"]\n'
        'description = "Explore the repository"\n'
    )

    config = load_config(
        path, ('profiles.full.tools.deferred=["web_search", "other"]',)
    )

    assert config.default_model == "openai/gpt-6-sol"
    assert config.reasoning_effort == "high"
    assert config.active_accounts["openai"] == "personal"
    assert config.profile().tools.active == ("workspace", "agents")
    assert config.profile().tools.deferred == ("web_search", "other")
    assert config.agents["explorer"].models == (
        "openai/gpt-6-luna",
        "anthropic/opus-5.5",
    )


def test_builtin_lite_profile_without_file(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.profile().tools.active == ("workspace",)
    assert config.default_model is None


@pytest.mark.parametrize(
    "content, message",
    [
        ('default_profile = "missing"\n', "Unknown coding profile"),
        ("unknown = 1\n", "unknown"),
        (
            '[profiles.lite.tools]\nactive = ["workspace"]\ndeferred = ["workspace"]\n',
            "active and deferred",
        ),
        ("[agents.explorer]\nmodels = []\n", "at least one model"),
    ],
)
def test_invalid_config_fails_before_agent_starts(tmp_path, content, message):
    path = tmp_path / "config.toml"
    path.write_text(content)
    with pytest.raises((ValueError, msgspec.ValidationError), match=message):
        load_config(path)
