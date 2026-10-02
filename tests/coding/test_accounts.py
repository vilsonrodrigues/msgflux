"""Tests for named API key account storage."""

import os

import pytest

from msgflux.coding.accounts import AccountStorage


def test_add_list_and_get_never_expose_secret(tmp_path):
    storage = AccountStorage(tmp_path)
    storage.add("openai", "personal", "secret-value", models=("gpt-6-luna",))

    listed = storage.list()
    fetched = storage.get("openai", "personal")
    assert listed == [fetched]
    assert fetched.models == ("gpt-6-luna",)
    assert "secret-value" not in repr(fetched)
    assert "secret-value" not in repr(listed)


def test_add_same_provider_aliases_and_explicit_credential_resolver(tmp_path):
    storage = AccountStorage(tmp_path)
    storage.add("openai", "personal", "key-one")
    storage.add("openai", "work", "key-two")

    resolver = storage.credential_resolver("openai", "work")
    assert resolver.resolve(object()).headers == {"Authorization": "Bearer key-two"}
    assert "key-two" not in repr(resolver)
    assert [account.alias for account in storage.list("openai")] == ["personal", "work"]


def test_records_models_without_duplicate_entries(tmp_path):
    storage = AccountStorage(tmp_path)
    storage.add("anthropic", "main", "secret")

    storage.record_model("anthropic", "main", "opus-5.5")
    result = storage.record_model("anthropic", "main", "opus-5.5")
    assert result.models == ("opus-5.5",)


@pytest.mark.parametrize(
    "provider,alias", [("../outside", "x"), ("openai", "../x"), ("", "x")]
)
def test_rejects_path_components(tmp_path, provider, alias):
    storage = AccountStorage(tmp_path)
    with pytest.raises(ValueError):
        storage.add(provider, alias, "secret")
    assert not (tmp_path / "accounts").exists()


def test_account_file_is_private_and_add_is_not_overwrite(tmp_path):
    storage = AccountStorage(tmp_path)
    storage.add("openai", "main", "first-secret")
    path = tmp_path / "accounts" / "openai" / "main.json"

    assert os.stat(path).st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        storage.add("openai", "main", "replacement-secret")
    assert (
        storage.credential_resolver("openai", "main")
        .resolve(None)
        .headers["Authorization"]
        == "Bearer first-secret"
    )


def test_rejects_symlinked_account_file(tmp_path):
    storage = AccountStorage(tmp_path)
    storage.add("openai", "main", "secret")
    path = tmp_path / "accounts" / "openai" / "main.json"
    outside = tmp_path / "outside.json"
    outside.write_text(path.read_text())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        storage.get("openai", "main")
