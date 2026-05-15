import asyncio

import pytest

from msgflux.context import ExecutionScope, execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    EventStream,
    EventType,
    PermissionDeniedError,
    PermissionManager,
)
from msgflux.tools.builtin import FILE_EDIT_TOOL_NAME, FileEdit


@pytest.mark.asyncio
async def test_file_edit_bypass_applies_edit_and_emits_events(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    tool = FileEdit()

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        with EventStream() as stream:
            result = await tool.acall(
                file_path=str(file_path),
                old_string="hello",
                new_string="goodbye",
            )
            stream.close()
            events = stream.events

    assert result == f"Edit applied to {file_path.resolve()}."
    assert file_path.read_text(encoding="utf-8") == "goodbye world\n"
    names = [event.name for event in events]
    assert EventType.FILE_EDIT_PROPOSED in names
    assert EventType.PERMISSION_GRANTED in names
    assert EventType.FILE_EDIT_APPLIED in names

    proposed = next(
        event for event in events if event.name == EventType.FILE_EDIT_PROPOSED
    )
    assert proposed.attributes["path"] == str(file_path.resolve())
    assert proposed.attributes["operation"] == "replace"
    assert "-hello world" in proposed.attributes["diff"]
    assert "+goodbye world" in proposed.attributes["diff"]


@pytest.mark.asyncio
async def test_file_edit_deny_rejects_without_writing(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    tool = FileEdit()

    with execution_context(permission_manager=PermissionManager(default_mode="deny")):
        with EventStream() as stream:
            with pytest.raises(PermissionDeniedError):
                await tool.acall(
                    file_path=str(file_path),
                    old_string="hello",
                    new_string="goodbye",
                )
            stream.close()
            events = stream.events

    assert file_path.read_text(encoding="utf-8") == "hello world\n"
    names = [event.name for event in events]
    assert EventType.FILE_EDIT_PROPOSED in names
    assert EventType.PERMISSION_DENIED in names
    assert EventType.FILE_EDIT_REJECTED in names
    assert EventType.FILE_EDIT_APPLIED not in names


@pytest.mark.asyncio
async def test_file_edit_ask_user_waits_for_approval(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    manager = PermissionManager(default_mode="ask_user")
    tool = FileEdit()

    async def approve_pending():
        while not manager.list_pending():
            await asyncio.sleep(0)
        request = manager.list_pending()[0]
        assert request.action == "file.edit"
        assert request.resource == str(file_path.resolve())
        assert "preview" in request.metadata
        manager.approve(request.request_id)

    with execution_context(permission_manager=manager):
        task = asyncio.create_task(approve_pending())
        result = await tool.acall(
            file_path=str(file_path),
            old_string="hello",
            new_string="goodbye",
        )
        await task

    assert result == f"Edit applied to {file_path.resolve()}."
    assert file_path.read_text(encoding="utf-8") == "goodbye world\n"


@pytest.mark.asyncio
async def test_file_edit_scope_policy_overrides_manager_default(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    manager = PermissionManager(default_mode="deny")
    tool = FileEdit()

    with execution_context(
        scope=ExecutionScope(permission_mode="bypass"),
        permission_manager=manager,
    ):
        await tool.acall(
            file_path=str(file_path),
            old_string="hello",
            new_string="goodbye",
        )

    assert file_path.read_text(encoding="utf-8") == "goodbye world\n"


@pytest.mark.asyncio
async def test_file_edit_tool_library_executes_async(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    library = ToolLibrary("agent", [FileEdit()])

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        responses = await library.acall(
            [
                (
                    "call_1",
                    FILE_EDIT_TOOL_NAME,
                    {
                        "file_path": str(file_path),
                        "old_string": "hello",
                        "new_string": "goodbye",
                    },
                )
            ]
        )

    response = responses.get_by_name(FILE_EDIT_TOOL_NAME)
    assert response is not None
    assert response.error is None
    assert response.result == f"Edit applied to {file_path.resolve()}."
    assert library.library[FILE_EDIT_TOOL_NAME].display_name == "Edit"
    assert library.library[FILE_EDIT_TOOL_NAME].description == FileEdit.description


def test_file_edit_sync_call_raises_clear_error(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    tool = FileEdit()

    with pytest.raises(RuntimeError, match="Edit is async-only"):
        tool(
            file_path=str(file_path),
            old_string="hello",
            new_string="goodbye",
        )


@pytest.mark.asyncio
async def test_file_edit_rejects_ambiguous_old_string(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello hello\n", encoding="utf-8")
    tool = FileEdit()

    with pytest.raises(ValueError, match="appears multiple times"):
        await tool.acall(
            file_path=str(file_path),
            old_string="hello",
            new_string="goodbye",
        )
