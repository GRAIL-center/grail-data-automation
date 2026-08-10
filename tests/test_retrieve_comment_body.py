from pathlib import Path

import pytest

from src.collect_comments.retrieve_comment_body import replaceCommentFolder


def _write_complete_comment(folder: Path, text: str) -> None:
    folder.mkdir()
    (folder / "manifest.json").write_text('{"complete": true}', encoding="utf-8")
    (folder / "full_comment.txt").write_text(text, encoding="utf-8")


def test_replace_comment_folder_retries_permission_error(tmp_path, monkeypatch):
    staging = tmp_path / ".comment.staging"
    destination = tmp_path / "comment"
    _write_complete_comment(staging, "new")

    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        if source == staging and target == destination:
            attempts += 1
            if attempts < 3:
                raise PermissionError(5, "Access is denied")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("src.collect_comments.retrieve_comment_body.time.sleep", lambda _: None)

    replaceCommentFolder(staging, destination)

    assert attempts == 3
    assert (destination / "full_comment.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()


def test_replace_comment_folder_accepts_concurrent_completion(tmp_path, monkeypatch):
    staging = tmp_path / ".comment.staging"
    destination = tmp_path / "comment"
    _write_complete_comment(staging, "this run")

    original_replace = Path.replace

    def concurrent_replace(source, target):
        if source == staging and target == destination:
            _write_complete_comment(destination, "other run")
            raise PermissionError(5, "Access is denied")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", concurrent_replace)

    replaceCommentFolder(staging, destination)

    assert (destination / "full_comment.txt").read_text(encoding="utf-8") == "other run"
    assert not staging.exists()


def test_replace_comment_folder_restores_backup_after_retry_failure(
    tmp_path,
    monkeypatch,
):
    staging = tmp_path / ".comment.staging"
    destination = tmp_path / "comment"
    _write_complete_comment(staging, "new")
    _write_complete_comment(destination, "old")

    original_replace = Path.replace

    def locked_staging(source, target):
        if source == staging:
            raise PermissionError(5, "Access is denied")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", locked_staging)
    monkeypatch.setattr("src.collect_comments.retrieve_comment_body.time.sleep", lambda _: None)

    with pytest.raises(PermissionError):
        replaceCommentFolder(staging, destination)

    assert (destination / "full_comment.txt").read_text(encoding="utf-8") == "old"
