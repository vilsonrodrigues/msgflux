from pathlib import Path

import pytest

from msgflux.nn.modules.agent import Agent
from msgflux.sandbox import BaseShellSandbox, Sandbox


class MockModel:
    model_type = "chat_completion"

    def __call__(self, **kwargs):
        return kwargs

    async def acall(self, **kwargs):
        return kwargs


def test_shell_factory_creates_just_bash_when_available():
    pytest.importorskip("just_bash")

    shell = Sandbox.shell("just-bash")

    assert isinstance(shell, BaseShellSandbox)
    assert shell.name == "shell"
    assert shell.capabilities.filesystem is True


@pytest.mark.asyncio
async def test_just_bash_shell_executes_commands_when_available():
    pytest.importorskip("just_bash")
    shell = Sandbox.shell("just-bash", files={"/workspace/input.txt": "hello\nworld\n"})

    result = await shell.acall("cat input.txt | wc -l")

    assert "exit_code=0" in result
    assert "stdout:\n2" in result


@pytest.mark.asyncio
async def test_just_bash_shell_persists_files_between_calls_when_available():
    pytest.importorskip("just_bash")
    shell = Sandbox.shell("just-bash")

    await shell.acall("echo runtime > /workspace/state.txt")
    result = await shell.acall("cat /workspace/state.txt")

    assert "stdout:\nruntime" in result


@pytest.mark.asyncio
async def test_just_bash_shell_mounts_file_when_available(tmp_path: Path):
    pytest.importorskip("just_bash")
    source = tmp_path / "notes.txt"
    source.write_text("alpha\nbeta\n", encoding="utf-8")
    shell = Sandbox.shell("just-bash", mounts={"/workspace/notes.txt": source})

    result = await shell.acall("grep beta notes.txt")

    assert "exit_code=0" in result
    assert "stdout:\nbeta" in result


@pytest.mark.asyncio
async def test_just_bash_shell_mounts_directory_when_available(tmp_path: Path):
    pytest.importorskip("just_bash")
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("msgFlux\n", encoding="utf-8")
    src = root / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')\n", encoding="utf-8")
    shell = Sandbox.shell("just-bash", mounts={"/workspace/project": root})

    result = await shell.acall("find project -type f | sort")

    assert "project/README.md" in result
    assert "project/src/main.py" in result


@pytest.mark.asyncio
async def test_just_bash_shell_overlay_workspace_does_not_write_host_when_available(
    tmp_path: Path,
):
    pytest.importorskip("just_bash")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    readme = workspace / "README.md"
    readme.write_text("original\n", encoding="utf-8")
    shell = Sandbox.shell("just-bash", fs_mode="overlay", workspace=workspace)

    first = await shell.acall("cat README.md")
    await shell.acall("echo changed > README.md")
    second = await shell.acall("cat README.md")

    assert "stdout:\noriginal" in first
    assert "stdout:\nchanged" in second
    assert readme.read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_just_bash_shell_mountable_fs_combines_files_and_overlay_mounts(
    tmp_path: Path,
):
    pytest.importorskip("just_bash")
    project = tmp_path / "project"
    project.mkdir()
    readme = project / "README.md"
    readme.write_text("mounted\n", encoding="utf-8")
    shell = Sandbox.shell(
        "just-bash",
        fs_mode="mountable",
        files={"/workspace/base.txt": "base\n"},
        mounts={"/workspace/project": project},
    )

    result = await shell.acall("cat base.txt; cat project/README.md")
    await shell.acall("echo changed > project/README.md")
    changed = await shell.acall("cat project/README.md")

    assert "base\nmounted" in result
    assert "stdout:\nchanged" in changed
    assert readme.read_text(encoding="utf-8") == "mounted\n"


def test_just_bash_shell_fs_mode_can_default_from_env_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("just_bash")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("env overlay\n", encoding="utf-8")
    monkeypatch.setenv("MSGFLUX_SHELL_FS_MODE", "overlay")
    monkeypatch.setenv("MSGFLUX_SHELL_WORKSPACE", str(workspace))

    shell = Sandbox.shell("just-bash")

    assert shell.fs_mode == "overlay"
    assert shell.workspace == workspace.resolve()


def test_just_bash_shell_rejects_symlink_mounts_when_available(tmp_path: Path):
    pytest.importorskip("just_bash")
    target = tmp_path / "target.txt"
    target.write_text("secret\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        Sandbox.shell("just-bash", mounts={"/workspace/link.txt": link})


@pytest.mark.asyncio
async def test_just_bash_shell_can_run_as_agent_tool_when_available():
    pytest.importorskip("just_bash")
    shell = Sandbox.shell("just-bash", files={"/workspace/input.txt": "alpha\nbeta\n"})
    agent = Agent(name="agent", model=MockModel(), tools=[shell])

    responses = await agent.tool_library.acall(
        [("call_1", "shell", {"command": "grep beta input.txt"})]
    )

    response = responses.get_by_name("shell")
    assert response is not None
    assert response.error is None
    assert "stdout:\nbeta" in response.result
