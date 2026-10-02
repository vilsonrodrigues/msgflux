"""Tests for per-thread coding persistence layout."""

import json
import os

import pytest

from msgflux.coding.storage import ThreadStorage, validate_thread_id


def test_new_thread_creates_minimal_metadata(tmp_path):
    storage = ThreadStorage(tmp_path)

    metadata = storage.new_thread()

    assert metadata.thread_id.startswith("thd_")
    assert storage.thread_dir(metadata.thread_id).is_dir()
    assert metadata == storage.read_metadata(metadata.thread_id)
    contents = json.loads(
        (storage.thread_dir(metadata.thread_id) / "metadata.json").read_text()
    )
    assert set(contents) == {
        "thread_id",
        "created_at",
        "updated_at",
        "workspace",
        "title",
    }


@pytest.mark.parametrize(
    "thread_id",
    [
        "../escape",
        "",
        ".",
        "x" * 129,
        "thd_" + "0" * 32 + "/x",
    ],
)
def test_invalid_thread_id_rejected_before_path_use(tmp_path, thread_id):
    storage = ThreadStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.thread_dir(thread_id)
    assert not (tmp_path / "threads").exists()


def test_opens_separate_sqlite_stores_inside_thread_dir(tmp_path):
    storage = ThreadStorage(tmp_path)
    thread_id = "thd_" + "a" * 32

    checkpoints = storage.open_checkpoint_store(thread_id)
    approvals = storage.open_approval_store(thread_id)
    try:
        directory = storage.thread_dir(thread_id)
        assert checkpoints.path == str(directory / "checkpoints.sqlite3")
        assert approvals.path == str(directory / "approvals.sqlite3")
        assert (directory / "metadata.json").is_file()
        assert os.stat(directory).st_mode & 0o777 == 0o700
        assert os.stat(directory / "metadata.json").st_mode & 0o777 == 0o600
        assert os.stat(directory / "checkpoints.sqlite3").st_mode & 0o777 == 0o600
    finally:
        checkpoints.close()
        approvals.close()


def test_rejects_symlinked_thread_directory(tmp_path):
    storage = ThreadStorage(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    storage.threads_dir.mkdir()
    thread_id = "thd_" + "b" * 32
    (storage.threads_dir / thread_id).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        storage.open_checkpoint_store(thread_id)


def test_validate_returns_valid_id_unchanged():
    thread_id = "thd_" + "c" * 32
    assert validate_thread_id(thread_id) == thread_id
    assert validate_thread_id("my-thread") == "my-thread"
