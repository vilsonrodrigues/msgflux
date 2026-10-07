"""Clipboard delivery and an explicit, private file export fallback."""

import asyncio
import os
import shlex
import shutil
import sys
from pathlib import Path


def parse_copy_arguments(arguments: str) -> tuple[str, Path | None]:
    tokens = shlex.split(arguments)
    destination = None
    if "--file" in tokens:
        index = tokens.index("--file")
        if index + 1 >= len(tokens):
            raise ValueError("Use /copy [all|last|selection] [--file PATH]")
        destination = Path(tokens[index + 1]).expanduser()
        del tokens[index : index + 2]
    if len(tokens) > 1 or (tokens and tokens[0] not in {"all", "last", "selection"}):
        raise ValueError("Use /copy [all|last|selection] [--file PATH]")
    return (tokens[0] if tokens else "selection"), destination


def export_text(path: Path, text: str) -> None:
    """Create a private export; never overwrite an existing file."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def _clipboard_commands():
    if sys.platform == "darwin":
        return [("pbcopy",)]
    if sys.platform == "win32":
        return [("clip",)]
    commands = []
    if os.environ.get("WAYLAND_DISPLAY"):
        commands.append(("wl-copy",))
    if os.environ.get("DISPLAY"):
        commands.extend(
            (("xclip", "-selection", "clipboard"), ("xsel", "--clipboard", "--input"))
        )
    return commands


async def copy_system_clipboard(text: str) -> bool:
    """Return true only when a native clipboard helper reports success."""
    for command in _clipboard_commands():
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *command[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            continue
        try:
            await asyncio.wait_for(process.communicate(text.encode("utf-8")), timeout=2)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.communicate()
            if asyncio.current_task().cancelling():
                raise
            continue
        if process.returncode == 0:
            return True
    return False
