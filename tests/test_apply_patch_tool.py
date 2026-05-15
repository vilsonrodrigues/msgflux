import asyncio

import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType, PermissionManager
from msgflux.tools.builtin import APPLY_PATCH_TOOL_NAME, ApplyPatch, FileRead


def _read_file(file_path):
    FileRead()(str(file_path))


@pytest.mark.asyncio
async def test_apply_patch_updates_multiple_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("hello world\n", encoding="utf-8")
    second.write_text("alpha beta\n", encoding="utf-8")
    _read_file(first)
    _read_file(second)
    patch = f"""*** Begin Patch
*** Update File: {first}
@@
-hello world
+goodbye world
*** Update File: {second}
@@
-alpha beta
+alpha gamma
*** End Patch"""

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        with EventStream() as stream:
            result = await ApplyPatch().acall(patch)
            stream.close()
            events = stream.events

    assert result == "Patch applied to 2 file(s)."
    assert first.read_text(encoding="utf-8") == "goodbye world\n"
    assert second.read_text(encoding="utf-8") == "alpha gamma\n"
    names = [event.name for event in events]
    assert names.count(EventType.FILE_EDIT_PROPOSED) == 2
    assert names.count(EventType.FILE_EDIT_APPLIED) == 2


@pytest.mark.asyncio
async def test_apply_patch_adds_and_deletes_files(tmp_path):
    existing = tmp_path / "delete.txt"
    created = tmp_path / "created.txt"
    existing.write_text("remove me\n", encoding="utf-8")
    _read_file(existing)
    patch = f"""*** Begin Patch
*** Delete File: {existing}
*** Add File: {created}
+created
+content
*** End Patch"""

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        result = await ApplyPatch().acall(patch)

    assert result == "Patch applied to 2 file(s)."
    assert not existing.exists()
    assert created.read_text(encoding="utf-8") == "created\ncontent\n"


@pytest.mark.asyncio
async def test_apply_patch_requires_prior_read_for_update(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** End Patch"""

    with pytest.raises(ValueError, match="must be read with the Read tool"):
        await ApplyPatch().acall(patch)


@pytest.mark.asyncio
async def test_apply_patch_waits_for_user_approval(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    manager = PermissionManager(default_mode="ask_user")
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** End Patch"""

    async def approve_pending():
        while not manager.list_pending():
            await asyncio.sleep(0)
        request = manager.list_pending()[0]
        assert request.tool_name == APPLY_PATCH_TOOL_NAME
        assert request.metadata["preview"]["operation"] == "update"
        manager.approve(request.request_id)

    with execution_context(permission_manager=manager):
        task = asyncio.create_task(approve_pending())
        result = await ApplyPatch().acall(patch)
        await task

    assert result == "Patch applied to 1 file(s)."
    assert file_path.read_text(encoding="utf-8") == "goodbye world\n"


@pytest.mark.asyncio
async def test_apply_patch_rejects_external_change_after_proposal(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    manager = PermissionManager(default_mode="ask_user")
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** End Patch"""

    async def mutate_and_approve_pending():
        while not manager.list_pending():
            await asyncio.sleep(0)
        request = manager.list_pending()[0]
        file_path.write_text("hello changed\n", encoding="utf-8")
        manager.approve(request.request_id)

    with execution_context(permission_manager=manager):
        task = asyncio.create_task(mutate_and_approve_pending())
        with pytest.raises(ValueError, match="changed after the patch was proposed"):
            await ApplyPatch().acall(patch)
        await task


@pytest.mark.asyncio
async def test_apply_patch_tool_library_executes_async(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** End Patch"""
    library = ToolLibrary("agent", [ApplyPatch()])

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        responses = await library.acall(
            [("call_1", APPLY_PATCH_TOOL_NAME, {"patch": patch})]
        )

    response = responses.get_by_name(APPLY_PATCH_TOOL_NAME)
    assert response is not None
    assert response.error is None
    assert response.result == "Patch applied to 1 file(s)."


def test_apply_patch_sync_call_raises_clear_error():
    with pytest.raises(RuntimeError, match="apply_patch is async-only"):
        ApplyPatch()("*** Begin Patch\n*** End Patch")


@pytest.mark.asyncio
async def test_apply_patch_rejects_duplicate_file_operations(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** Update File: {file_path}
@@
-hello world
+hello team
*** End Patch"""

    with pytest.raises(ValueError, match="multiple operations for file"):
        await ApplyPatch().acall(patch)


@pytest.mark.asyncio
async def test_apply_patch_rejects_update_hunk_without_context(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
+inserted
*** End Patch"""

    with pytest.raises(ValueError, match="must include context or removed lines"):
        await ApplyPatch().acall(patch)


@pytest.mark.asyncio
async def test_apply_patch_rolls_back_when_later_write_fails(tmp_path):
    file_path = tmp_path / "example.txt"
    blocking_parent = tmp_path / "not_a_directory"
    blocking_parent.write_text("blocks child creation\n", encoding="utf-8")
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+goodbye world
*** Add File: {blocking_parent / "child.txt"}
+created
*** End Patch"""

    with execution_context(permission_manager=PermissionManager(default_mode="bypass")):
        with pytest.raises(FileExistsError):
            await ApplyPatch().acall(patch)

    assert file_path.read_text(encoding="utf-8") == "hello world\n"
    assert blocking_parent.read_text(encoding="utf-8") == "blocks child creation\n"


@pytest.mark.asyncio
async def test_parallel_apply_patch_revalidates_same_file_after_approval(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello world\n", encoding="utf-8")
    _read_file(file_path)
    manager = PermissionManager(default_mode="ask_user")
    first_patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+first writer
*** End Patch"""
    second_patch = f"""*** Begin Patch
*** Update File: {file_path}
@@
-hello world
+second writer
*** End Patch"""

    async def wait_for_pending_count(count: int) -> None:
        while len(manager.list_pending()) < count:
            await asyncio.sleep(0)

    with execution_context(permission_manager=manager):
        first = asyncio.create_task(ApplyPatch().acall(first_patch))
        second = asyncio.create_task(ApplyPatch().acall(second_patch))
        await wait_for_pending_count(2)
        for request in manager.list_pending():
            manager.approve(request.request_id)
        results = await asyncio.gather(first, second, return_exceptions=True)

    successes = [result for result in results if isinstance(result, str)]
    failures = [result for result in results if isinstance(result, ValueError)]
    assert successes == ["Patch applied to 1 file(s)."]
    assert len(failures) == 1
    assert "changed after the patch was proposed" in str(failures[0])
    assert file_path.read_text(encoding="utf-8") in {
        "first writer\n",
        "second writer\n",
    }
