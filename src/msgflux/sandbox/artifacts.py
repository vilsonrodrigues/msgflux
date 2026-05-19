from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import aiofiles


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    path: Path
    filename: str
    size: int
    unit: str = "bytes"


class ArtifactNamespace:
    """Lazy, paginated access to files mounted for sandbox code."""

    def __init__(
        self,
        artifacts: Mapping[str, ArtifactRef] | None = None,
        *,
        max_read_bytes: int = 12000,
    ) -> None:
        self._artifacts = dict(artifacts or {})
        self._max_read_bytes = max_read_bytes

    def list(self) -> list[str]:
        return sorted(self._artifacts)

    def __getitem__(self, name: str):
        helpers = {
            "list": self.alist,
            "info": self.ainfo,
            "read": self.aread,
            "search": self.asearch,
            "help": self.ahelp,
        }
        helper = helpers.get(name)
        if helper is None:
            raise KeyError(f"Artifact helper `{name}` is not available.")
        return helper

    def info(self, name: str) -> dict[str, str | int]:
        artifact = self._get(name)
        return {
            "name": artifact.name,
            "filename": artifact.filename,
            "size": artifact.size,
            "unit": artifact.unit,
        }

    async def alist(self) -> list[str]:
        return self.list()

    async def ainfo(self, name: str) -> dict[str, str | int]:
        return self.info(name)

    def read(
        self,
        name: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> str:
        if offset < 0:
            raise ValueError("`offset` must be greater than or equal to zero.")
        if limit is None:
            raise ValueError("`limit` is required when reading artifacts.")
        if limit <= 0:
            raise ValueError("`limit` must be greater than zero.")
        if limit > self._max_read_bytes:
            raise ValueError(
                "`limit` exceeds max artifact read size "
                f"({self._max_read_bytes} bytes)."
            )

        artifact = self._get(name)
        with artifact.path.open("rb") as file:
            file.seek(offset)
            data = file.read(limit)
        return data.decode(encoding, errors="replace")

    async def aread(
        self,
        name: str,
        *,
        offset: int = 0,
        limit: int | None = None,
        encoding: str = "utf-8",
    ) -> str:
        if offset < 0:
            raise ValueError("`offset` must be greater than or equal to zero.")
        if limit is None:
            raise ValueError("`limit` is required when reading artifacts.")
        if limit <= 0:
            raise ValueError("`limit` must be greater than zero.")
        if limit > self._max_read_bytes:
            raise ValueError(
                "`limit` exceeds max artifact read size "
                f"({self._max_read_bytes} bytes)."
            )

        artifact = self._get(name)
        async with aiofiles.open(artifact.path, "rb") as file:
            await file.seek(offset)
            data = await file.read(limit)
        return data.decode(encoding, errors="replace")

    def search(
        self,
        name: str,
        query: str,
        *,
        limit: int = 10,
        chunk_size: int = 8192,
        encoding: str = "utf-8",
    ) -> list[dict[str, str | int]]:
        if not query:
            raise ValueError("`query` must be a non-empty string.")
        if limit <= 0:
            raise ValueError("`limit` must be greater than zero.")
        if chunk_size <= 0 or chunk_size > self._max_read_bytes:
            raise ValueError(
                "`chunk_size` must be greater than zero and less than or equal "
                f"to {self._max_read_bytes} bytes."
            )

        artifact = self._get(name)
        matches: list[dict[str, str | int]] = []
        offset = 0
        with artifact.path.open("rb") as file:
            while len(matches) < limit:
                data = file.read(chunk_size)
                if not data:
                    break
                text = data.decode(encoding, errors="replace")
                _append_text_matches(
                    matches,
                    artifact=artifact,
                    text=text,
                    query=query,
                    base_offset=offset,
                    limit=limit,
                    encoding=encoding,
                )
                offset += len(data)
        return matches

    async def asearch(
        self,
        name: str,
        query: str,
        *,
        limit: int = 10,
        chunk_size: int = 8192,
        encoding: str = "utf-8",
    ) -> list[dict[str, str | int]]:
        if not query:
            raise ValueError("`query` must be a non-empty string.")
        if limit <= 0:
            raise ValueError("`limit` must be greater than zero.")
        if chunk_size <= 0 or chunk_size > self._max_read_bytes:
            raise ValueError(
                "`chunk_size` must be greater than zero and less than or equal "
                f"to {self._max_read_bytes} bytes."
            )

        artifact = self._get(name)
        matches: list[dict[str, str | int]] = []
        offset = 0
        async with aiofiles.open(artifact.path, "rb") as file:
            while len(matches) < limit:
                data = await file.read(chunk_size)
                if not data:
                    break
                text = data.decode(encoding, errors="replace")
                _append_text_matches(
                    matches,
                    artifact=artifact,
                    text=text,
                    query=query,
                    base_offset=offset,
                    limit=limit,
                    encoding=encoding,
                )
                offset += len(data)
        return matches

    def help(self) -> str:
        return (
            "Use artifacts to inspect mounted files without loading them all into "
            'memory. Prefer await artifacts["read"](name, offset=0, limit=4000) '
            'and pass chunks directly to tools. Use await artifacts["info"](name) to '
            "check byte size."
        )

    async def ahelp(self) -> str:
        return self.help()

    def _get(self, name: str) -> ArtifactRef:
        artifact = self._artifacts.get(name)
        if artifact is None:
            raise KeyError(f"Artifact `{name}` is not available.")
        return artifact


def normalize_artifacts(
    artifacts: (
        Mapping[str, str | Path] | list[str | Path] | tuple[str | Path, ...] | None
    ),
) -> dict[str, ArtifactRef]:
    if artifacts is None:
        return {}
    if isinstance(artifacts, Mapping):
        items = artifacts.items()
    elif isinstance(artifacts, (list, tuple)):
        items = ((Path(path).name, path) for path in artifacts)
    else:
        raise TypeError(
            "`artifacts` must be a mapping, list, tuple or None, "
            f"given `{type(artifacts)}`"
        )

    normalized: dict[str, ArtifactRef] = {}
    for raw_name, raw_path in items:
        name = str(raw_name)
        if not name:
            raise ValueError("Artifact names must be non-empty strings.")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        stat = path.stat()
        normalized[name] = ArtifactRef(
            name=name,
            path=path,
            filename=path.name,
            size=stat.st_size,
        )
    return normalized


def _compact_preview(text: str, index: int, query_length: int) -> str:
    start = max(0, index - 80)
    end = min(len(text), index + query_length + 80)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].replace("\n", "\\n") + suffix


def _append_text_matches(
    matches: list[dict[str, str | int]],
    *,
    artifact: ArtifactRef,
    text: str,
    query: str,
    base_offset: int,
    limit: int,
    encoding: str,
) -> None:
    start = 0
    while len(matches) < limit:
        index = text.find(query, start)
        if index < 0:
            return
        matches.append(
            {
                "name": artifact.name,
                "offset": base_offset + len(text[:index].encode(encoding)),
                "preview": _compact_preview(text, index, len(query)),
            }
        )
        start = index + len(query)
