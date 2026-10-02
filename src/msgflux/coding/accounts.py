"""Local storage for named provider API key accounts."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import msgspec

from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise ValueError(
            f"{label} must contain 1-64 ASCII letters, digits, dots, underscores "
            "or hyphens"
        )
    if value in {".", ".."}:
        raise ValueError(f"invalid {label}")
    return value


class AccountMetadata(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Public account metadata, safe to show without revealing the API key."""

    provider: str
    alias: str
    models: tuple[str, ...] = ()


class _StoredAccount(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    provider: str
    alias: str
    api_key: str
    models: tuple[str, ...] = ()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"alias={self.alias!r}, api_key=<redacted>, models={self.models!r})"
        )


class AccountCredentialResolver(ModelCredentialResolver):
    """Resolve one explicitly selected API key as a Bearer token."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def resolve(self, owner: Any) -> ResolvedModelCredentials:
        del owner
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer {self._api_key}"}
        )


class AccountStorage:
    """Manage ``accounts/<provider>/<alias>.json`` under a private root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def accounts_dir(self) -> Path:
        return self.root / "accounts"

    def add(
        self, provider: str, alias: str, api_key: str, *, models: tuple[str, ...] = ()
    ) -> AccountMetadata:
        provider = _validate_name(provider, "provider")
        alias = _validate_name(alias, "alias")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        models = self._validate_models(models)
        directory = self._provider_dir(provider, create=True)
        path = directory / f"{alias}.json"
        self._reject_symlink(path, "account file")
        account = _StoredAccount(provider, alias, api_key, models)
        self._write_atomic(path, msgspec.json.encode(account), replace=False)
        return AccountMetadata(provider, alias, models)

    def list(self, provider: str | None = None) -> list[AccountMetadata]:
        if provider is not None:
            provider = _validate_name(provider, "provider")
            directories = [self._provider_dir(provider, create=False)]
        else:
            if not self.accounts_dir.exists():
                return []
            if self.accounts_dir.is_symlink():
                raise ValueError("accounts directory must not be a symlink")
            directories = sorted(p for p in self.accounts_dir.iterdir() if p.is_dir())
        result: list[AccountMetadata] = []
        for directory in directories:
            if not directory.exists():
                continue
            self._reject_symlink(directory, "provider directory")
            for path in sorted(directory.glob("*.json")):
                self._reject_symlink(path, "account file")
                account = self._read(path)
                result.append(
                    AccountMetadata(account.provider, account.alias, account.models)
                )
        return result

    def get(self, provider: str, alias: str) -> AccountMetadata:
        account = self._read_account(provider, alias)
        return AccountMetadata(account.provider, account.alias, account.models)

    def remove(self, provider: str, alias: str) -> None:
        path = self._account_path(provider, alias)
        self._reject_symlink(path, "account file")
        path.unlink()

    def record_model(self, provider: str, alias: str, model: str) -> AccountMetadata:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        account = self._read_account(provider, alias)
        models = account.models if model in account.models else (*account.models, model)
        updated = _StoredAccount(
            account.provider, account.alias, account.api_key, models
        )
        self._write_atomic(
            self._account_path(account.provider, account.alias),
            msgspec.json.encode(updated),
            replace=True,
        )
        return AccountMetadata(updated.provider, updated.alias, updated.models)

    def credential_resolver(
        self, provider: str, alias: str
    ) -> AccountCredentialResolver:
        """Build a resolver for this explicit account; never selects a fallback."""
        return AccountCredentialResolver(self._read_account(provider, alias).api_key)

    def _read_account(self, provider: str, alias: str) -> _StoredAccount:
        path = self._account_path(provider, alias)
        self._reject_symlink(path, "account file")
        return self._read(path)

    def _account_path(self, provider: str, alias: str) -> Path:
        provider = _validate_name(provider, "provider")
        alias = _validate_name(alias, "alias")
        return self._provider_dir(provider, create=False) / f"{alias}.json"

    def _provider_dir(self, provider: str, *, create: bool) -> Path:
        path = self.accounts_dir / _validate_name(provider, "provider")
        if create:
            self.accounts_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.accounts_dir.is_symlink():
            raise ValueError("accounts directory must not be a symlink")
        if create:
            path.mkdir(exist_ok=True, mode=0o700)
        if path.is_symlink():
            raise ValueError("provider directory must not be a symlink")
        return path

    @staticmethod
    def _validate_models(models: tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(models, (tuple, list)) or any(
            not isinstance(model, str) or not model.strip() for model in models
        ):
            raise ValueError("models must be a sequence of non-empty strings")
        return tuple(dict.fromkeys(models))

    @staticmethod
    def _reject_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink")

    @staticmethod
    def _read(path: Path) -> _StoredAccount:
        try:
            return msgspec.json.decode(path.read_bytes(), type=_StoredAccount)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"account not found: {path.stem}") from exc
        except msgspec.DecodeError as exc:
            raise ValueError(f"invalid account data: {path.stem}") from exc

    @staticmethod
    def _write_atomic(path: Path, payload: bytes, *, replace: bool) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".account-", dir=path.parent)
        temp_path = Path(temporary)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(temp_path, path)
            else:
                # link is atomic and fails if an account with this alias exists.
                os.link(temp_path, path)
                temp_path.unlink()
        finally:
            temp_path.unlink(missing_ok=True)


__all__ = ["AccountCredentialResolver", "AccountMetadata", "AccountStorage"]
