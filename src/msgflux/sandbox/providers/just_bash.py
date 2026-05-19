from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Mapping

import msgflux.nn.functional as F
from msgflux.sandbox.base import BaseShellSandbox, SandboxCapabilities

try:
    from just_bash import Bash
    from just_bash.fs import (
        InMemoryFs,
        MountableFs,
        MountableFsOptions,
        MountConfig,
        OverlayFs,
        OverlayFsOptions,
    )
    from just_bash.types import ExecResult, NetworkConfig
except ImportError as exc:  # pragma: no cover - exercised by factory import path.
    raise ImportError(
        "`Sandbox.shell('just-bash')` requires the optional dependency "
        "`just-bash`. Install it with `msgflux[shell]`."
    ) from exc


class JustBashSandbox(BaseShellSandbox):
    name = "shell"
    display_name = "Shell"
    description = (
        "Execute a bash command in a just-bash virtual Linux environment. "
        "The filesystem is sandboxed and persists across command calls."
    )
    usage_guidance = (
        "Use `shell` for grep, sed, awk, jq, pipes, redirections and quick file "
        "inspection inside the sandbox filesystem. Mounted files and directories "
        "are copied into the virtual filesystem; host files are not modified."
    )

    def __init__(
        self,
        *,
        files: Mapping[str, str | bytes] | None = None,
        mounts: Mapping[str, str | Path] | None = None,
        workspace: str | Path | None = None,
        fs_mode: Literal["memory", "overlay", "mountable"] | None = None,
        read_only: bool = False,
        cwd: str = "/workspace",
        env: Mapping[str, str] | None = None,
        network: bool | NetworkConfig | None = None,
    ) -> None:
        super().__init__()
        self.cwd = _normalize_virtual_path(cwd)
        self.env = dict(env or {})
        self.files = _build_initial_files(files=files, mounts=mounts)
        self.workspace = _resolve_optional_path(
            workspace or os.getenv("MSGFLUX_SHELL_WORKSPACE")
        )
        self.fs_mode = _normalize_fs_mode(fs_mode)
        self.read_only = read_only
        self.network = _normalize_network(network=network)
        self.capabilities = SandboxCapabilities(
            language="shell",
            network=_network_enabled(self.network),
            filesystem=True,
            persistence=True,
            limitations=(
                "Runs through just-bash command implementations, so behavior may "
                "differ from system bash.",
                "Mounted host files are copied or overlaid into a virtual "
                "filesystem; host writes are not performed by default.",
            ),
        )
        filesystem = _build_filesystem(
            files=self.files,
            mounts=mounts,
            workspace=self.workspace,
            cwd=self.cwd,
            fs_mode=self.fs_mode,
            read_only=read_only,
        )
        self._bash = Bash(
            fs=filesystem,
            files=None if filesystem is not None else self.files,
            cwd=self.cwd,
            env=self.env,
            network=self.network,
        )

    def __call__(self, command: str) -> str:
        return F.wait_for(self.acall, command)

    async def acall(self, command: str) -> str:
        result = await self._bash.exec(command)
        return _format_exec_result(result)

    @property
    def fs(self):
        return self._bash.fs


def _normalize_network(
    *,
    network: bool | NetworkConfig | None,
) -> NetworkConfig | None:
    if isinstance(network, NetworkConfig) or network is None:
        return network
    if network is True:
        return NetworkConfig(dangerously_allow_full_internet_access=True)
    return NetworkConfig(dangerously_allow_full_internet_access=False)


def _network_enabled(network: NetworkConfig | None) -> bool:
    return bool(
        network is not None and network.dangerously_allow_full_internet_access
    )


def _normalize_fs_mode(
    fs_mode: Literal["memory", "overlay", "mountable"] | None,
) -> Literal["memory", "overlay", "mountable"]:
    value = fs_mode or os.getenv("MSGFLUX_SHELL_FS_MODE", "memory")
    if value not in {"memory", "overlay", "mountable"}:
        raise ValueError(
            "`fs_mode` must be one of: 'memory', 'overlay', 'mountable'."
        )
    return value


def _resolve_optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Workspace path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Workspace path must be a directory: {path}")
    return path


def _build_filesystem(
    *,
    files: Mapping[str, str | bytes],
    mounts: Mapping[str, str | Path] | None,
    workspace: Path | None,
    cwd: str,
    fs_mode: Literal["memory", "overlay", "mountable"],
    read_only: bool,
):
    if fs_mode == "memory":
        return None
    if fs_mode == "overlay":
        if workspace is None:
            raise ValueError("`fs_mode='overlay'` requires `workspace`.")
        return OverlayFs(
            OverlayFsOptions(
                root=str(workspace),
                mount_point=cwd,
                read_only=read_only,
            )
        )

    mount_configs = []
    if workspace is not None:
        mount_configs.append(
            MountConfig(
                mount_point=cwd,
                filesystem=OverlayFs(
                    OverlayFsOptions(
                        root=str(workspace),
                        mount_point="/",
                        read_only=read_only,
                    )
                ),
            )
        )
    for mount_path, source in (mounts or {}).items():
        source_path = Path(source).expanduser().resolve()
        if source_path.is_dir():
            mount_configs.append(
                MountConfig(
                    mount_point=_normalize_virtual_path(mount_path),
                    filesystem=OverlayFs(
                        OverlayFsOptions(
                            root=str(source_path),
                            mount_point="/",
                            read_only=read_only,
                        )
                    ),
                )
            )
    return MountableFs(
        MountableFsOptions(
            base=InMemoryFs(dict(files)),
            mounts=mount_configs,
        )
    )


def _build_initial_files(
    *,
    files: Mapping[str, str | bytes] | None,
    mounts: Mapping[str, str | Path] | None,
) -> dict[str, str | bytes]:
    initial: dict[str, str | bytes] = {
        _normalize_virtual_path(path): content
        for path, content in (files or {}).items()
    }
    for mount_path, source in (mounts or {}).items():
        raw_source_path = Path(source).expanduser()
        if raw_source_path.is_symlink():
            raise ValueError(f"Mount source cannot be a symlink: {raw_source_path}")
        source_path = raw_source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Mount source does not exist: {source_path}")
        virtual_path = _normalize_virtual_path(mount_path)
        if source_path.is_file():
            initial[_virtual_file_path(virtual_path, source_path)] = (
                source_path.read_bytes()
            )
            continue
        if source_path.is_dir():
            for file_path in source_path.rglob("*"):
                if file_path.is_symlink():
                    continue
                if not file_path.is_file():
                    continue
                relative = file_path.relative_to(source_path).as_posix()
                initial[f"{virtual_path.rstrip('/')}/{relative}"] = (
                    file_path.read_bytes()
                )
            continue
        raise ValueError(f"Mount source must be a file or directory: {source_path}")
    return initial


def _normalize_virtual_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    if not normalized:
        raise ValueError("Virtual paths must be non-empty.")
    if not normalized.startswith("/"):
        normalized = f"/workspace/{normalized}"
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized


def _virtual_file_path(virtual_path: str, source_path: Path) -> str:
    if virtual_path.endswith("/"):
        return f"{virtual_path.rstrip('/')}/{source_path.name}"
    return virtual_path


def _format_exec_result(result: ExecResult) -> str:
    parts = [f"exit_code={result.exit_code}"]
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr:
        parts.append(f"stderr:\n{result.stderr.rstrip()}")
    return "\n".join(parts)
