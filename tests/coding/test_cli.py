import sqlite3
import sys
from types import ModuleType

import pytest

from msgflux.coding.accounts import AccountStorage
from msgflux.coding.cli import (
    _parser,
    _run,
    _legacy_thread_stores,
    _load_extensions,
    _make_model,
    _make_subagents,
    main,
)
from msgflux.coding.config import CodingConfig, SubagentConfig
from msgflux.data.stores import InMemoryCheckpointStore


def test_explicit_extension_entry_point_registers_command(monkeypatch):
    module = ModuleType("sample_coding_extension")

    def register(extensions):
        extensions.register_command("echo", lambda text: text)

    module.register = register
    monkeypatch.setitem(sys.modules, module.__name__, module)

    extensions = _load_extensions(["sample_coding_extension:register"])

    assert extensions.commands()[0].id == "echo"
    with pytest.raises(ValueError, match="MODULE:FUNCTION"):
        _load_extensions(["invalid-entry-point"])


def test_account_command_adds_named_key_without_echoing_secret(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("TEST_CODING_ACCOUNT_KEY", "test-secret")

    main(
        [
            "account",
            "--state-dir",
            str(tmp_path),
            "add",
            "openai",
            "work",
            "--key-env",
            "TEST_CODING_ACCOUNT_KEY",
            "--model",
            "gpt-test",
        ]
    )
    main(["account", "--state-dir", str(tmp_path), "list", "openai"])

    output = capsys.readouterr().out
    assert "openai:work" in output
    assert "gpt-test" in output
    assert "test-secret" not in output


def test_model_uses_only_the_selected_account(tmp_path):
    accounts = AccountStorage(tmp_path)
    accounts.add("openai", "personal", "first-key")
    accounts.add("openai", "work", "second-key")
    config = CodingConfig(active_accounts={"openai": "work"})

    model = _make_model("openai/gpt-4.1-mini", None, config, accounts)

    assert model.credential_resolver.resolve(model).headers == {
        "Authorization": "Bearer second-key"
    }
    with pytest.raises(FileNotFoundError):
        _make_model(
            "openai/gpt-4.1-mini",
            None,
            CodingConfig(active_accounts={"openai": "missing"}),
            accounts,
        )


def test_subagent_model_gateway_disables_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = CodingConfig(
        agents={
            "explorer": SubagentConfig(models=("openai/gpt-4.1-mini", "openai/gpt-4.1"))
        }
    )

    [explorer] = _make_subagents(
        config, AccountStorage(tmp_path), InMemoryCheckpointStore()
    )

    assert explorer.model.fallback is False
    assert len(explorer.model._deployments) == 2


def test_legacy_thread_detection_keeps_shared_database_available(tmp_path):
    legacy_dir = tmp_path / "coding"
    legacy_dir.mkdir()
    checkpoint_path = legacy_dir / "checkpoints.sqlite3"
    with sqlite3.connect(checkpoint_path) as connection:
        connection.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        connection.execute("INSERT INTO checkpoints VALUES ('old-thread')")

    assert _legacy_thread_stores(tmp_path, "old-thread") == (
        checkpoint_path,
        legacy_dir / "approvals.sqlite3",
    )
    assert _legacy_thread_stores(tmp_path, "another-thread") is None


@pytest.mark.asyncio
async def test_cli_builds_agent_from_profile_and_thread_store(tmp_path, monkeypatch):
    from msgflux.coding.tui import CodingApp

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "module.py").write_text("value = 1")
    state = tmp_path / "state"
    state.mkdir()
    (state / "config.toml").write_text(
        'default_model = "openai/gpt-5"\n'
        '[profiles.lite.tools]\nactive = ["workspace"]\n'
    )
    seen = {}

    from types import SimpleNamespace

    async def fake_stream(self, prompt):
        yield SimpleNamespace(type="message.end", data={"content": "verified"})

    monkeypatch.setattr("msgflux.coding.cli.CodingSession.stream", fake_stream)

    async def fake_run(self):
        seen["thread"] = self.coding_session.thread_id
        assert not (state / "threads").exists()
        assert self.coding_session.agent is None
        async for _event in self.coding_session.stream("inspect project"):
            pass
        seen["tools"] = self.coding_session.agent.tool_library.get_tool_names()

    monkeypatch.setattr(CodingApp, "run_async", fake_run)
    args = _parser().parse_args(
        ["--workspace", str(workspace), "--state-dir", str(state)]
    )

    await _run(args)

    assert set(seen["tools"]) >= {"read", "bash", "apply_patch", "task"}
    assert not set(seen["tools"]) & {"edit", "write", "ls", "glob", "grep"}
    thread_dir = state / "threads" / seen["thread"]
    assert (thread_dir / "checkpoints.sqlite3").is_file()
    assert (thread_dir / "metadata.json").is_file()


@pytest.mark.asyncio
async def test_print_mode_omits_progress_tools_and_renders_final_output(
    tmp_path, monkeypatch, capsys
):
    from types import SimpleNamespace
    from msgflux.coding import cli

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    project = tmp_path / "project"
    project.mkdir()
    observed = {}

    async def fake_stream(self, prompt):
        observed["name"] = self.agent.get_module_name()
        observed["tools"] = self.agent.tool_library.get_tool_names()
        observed["prompt"] = self.agent.system_prompt.data
        yield SimpleNamespace(
            type="commentary.delta", data={"delta": "hidden progress"}
        )
        yield SimpleNamespace(type="message.end", data={"content": "final result"})

    monkeypatch.setattr(cli.CodingSession, "stream", fake_stream)
    args = _parser().parse_args(
        [
            "--workspace",
            str(project),
            "--state-dir",
            str(tmp_path / "state"),
            "--model",
            "openai/gpt-5",
            "--read-only",
            "-p",
            "inspect",
        ]
    )
    await _run(args)
    assert capsys.readouterr().out == "final result\n"
    assert observed["name"] == "main"
    assert "send_user_message" not in observed["tools"]
    assert "bash" not in observed["tools"]
    assert "noninteractive" in observed["prompt"]
