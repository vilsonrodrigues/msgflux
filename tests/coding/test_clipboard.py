"""Clipboard parsing, export, and native helper behavior."""

import os
from pathlib import Path

import pytest

from msgflux.coding.tui import clipboard


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ("", ("selection", None)),
        ("all", ("all", None)),
        ("last --file ~/answer.txt", ("last", Path.home() / "answer.txt")),
        ('--file "a b.txt" selection', ("selection", Path("a b.txt"))),
    ],
)
def test_parse_copy_arguments(arguments, expected):
    assert clipboard.parse_copy_arguments(arguments) == expected


@pytest.mark.parametrize(
    "arguments", ["unknown", "all last", "--file", "--file x junk"]
)
def test_parse_copy_arguments_rejects_invalid_forms(arguments):
    with pytest.raises(ValueError):
        clipboard.parse_copy_arguments(arguments)


def test_export_creates_private_file_and_refuses_overwrite(tmp_path):
    path = tmp_path / "export.txt"
    clipboard.export_text(path, "private text")
    assert path.read_text(encoding="utf-8") == "private text"
    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        clipboard.export_text(path, "replacement")
    assert path.read_text(encoding="utf-8") == "private text"


@pytest.mark.asyncio
async def test_native_clipboard_returns_false_when_helpers_fail(monkeypatch):
    class Process:
        returncode = 1

        async def communicate(self, _payload):
            return None

    monkeypatch.setattr(clipboard, "_clipboard_commands", lambda: [("helper",)])
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: "/helper")

    async def create(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr(clipboard.asyncio, "create_subprocess_exec", create)
    assert await clipboard.copy_system_clipboard("text") is False


@pytest.mark.asyncio
async def test_native_clipboard_returns_true_on_success(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self, payload):
            assert payload == "héllo".encode()
            return None

    monkeypatch.setattr(clipboard, "_clipboard_commands", lambda: [("helper", "--arg")])
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: "/helper")
    calls = []

    async def create(executable, *args, **kwargs):
        calls.append((executable, args, kwargs))
        return Process()

    monkeypatch.setattr(clipboard.asyncio, "create_subprocess_exec", create)
    assert await clipboard.copy_system_clipboard("héllo") is True
    assert calls[0][0:2] == ("/helper", ("--arg",))
