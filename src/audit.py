"""Append-only audit records and deterministic reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

AUDIT_SCHEMA_VERSION = "1.0"
ANALYSIS_SCHEMA_VERSION = "2.0"
PROMPT_VERSION = "2026-07-26.1"
REPRODUCIBILITY_PACKAGES = (
    "PyMuPDF",
    "Pillow",
    "PyYAML",
    "Flask",
    "beautifulsoup4",
    "gspread",
    "pytesseract",
    "python-docx",
    "python-dotenv",
    "requests",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def create_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must be 1-128 characters using only letters, numbers, "
            "periods, underscores, or hyphens"
        )
    if run_id in {".", ".."}:
        raise ValueError("run_id cannot be '.' or '..'")
    return run_id


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
    )


def atomic_write_text(
    path: str | Path,
    value: str,
    encoding: str = "utf-8",
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open(
            "w",
            encoding=encoding,
            newline="\n",
        ) as stream:
            stream.write(value)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


class AuditTrail:
    """Thread-safe, append-only JSONL audit writer."""

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        base_context: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.base_context = dict(base_context or {})
        self._lock = threading.Lock()
        self._sequence = 0
        if self.path.is_file():
            try:
                with self.path.open("r", encoding="utf-8") as stream:
                    self._sequence = sum(1 for line in stream if line.strip())
            except OSError:
                self._sequence = 0

    def record(
        self,
        event: str,
        *,
        status: str = "ok",
        **details: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record = {
                "audit_schema_version": AUDIT_SCHEMA_VERSION,
                "sequence": self._sequence,
                "timestamp": utc_now(),
                **self.base_context,
                "run_id": self.run_id,
                "event": event,
                "status": status,
                **details,
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_json(record))
                stream.write("\n")
        return record


def _run_git(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def source_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source_files = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("src/**/*.html"),
            *root.glob("src/**/*.css"),
            *root.glob("src/**/*.js"),
            *root.glob("*.py"),
            *root.glob("*.yaml"),
            *root.glob("*.yml"),
            *root.glob("requirements*.txt"),
        }
    )
    digest = hashlib.sha256()
    file_count = 0
    for path in source_files:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        file_count += 1

    status = _run_git(root, "status", "--porcelain")
    return {
        "repository_root": str(root),
        "git_commit": _run_git(root, "rev-parse", "HEAD"),
        "git_branch": _run_git(root, "branch", "--show-current"),
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": digest.hexdigest(),
        "source_file_count": file_count,
    }


def package_versions(
    package_names: Iterable[str] = REPRODUCIBILITY_PACKAGES,
) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def sanitized_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        return {}

    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            redacted = {}
            for key, nested in value.items():
                normalized_key = str(key).casefold()
                is_env_reference = normalized_key.endswith("_env")
                is_secret = any(
                    marker in normalized_key
                    for marker in [
                        "api_key",
                        "password",
                        "secret",
                        "token",
                        "credential",
                    ]
                )
                redacted[key] = (
                    "[REDACTED]"
                    if is_secret and not is_env_reference
                    else redact(nested)
                )
            return redacted
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    snapshot = redact(json.loads(json.dumps(data)))

    environment_names = {"REGULATION_API_KEY"}

    def collect_env_references(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if (
                    str(key).casefold().endswith("_env")
                    and isinstance(nested, str)
                    and nested.strip()
                ):
                    environment_names.add(nested.strip())
                collect_env_references(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_env_references(nested)

    collect_env_references(data)
    snapshot["environment"] = {
        f"{name}_present": bool(os.getenv(name))
        for name in sorted(environment_names)
    }
    return snapshot


def runtime_manifest(
    repo_root: str | Path,
    config_path: str | Path = "config.yaml",
) -> dict[str, Any]:
    tesseract_path = shutil.which("tesseract")
    tesseract_version = None
    if tesseract_path:
        try:
            completed = subprocess.run(
                [tesseract_path, "--version"],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
            tesseract_version = completed.stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            tesseract_version = "available; version query failed"
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "external_tools": {
            "tesseract": {
                "path": tesseract_path,
                "version": tesseract_version,
            }
        },
        "packages": package_versions(),
        "source": source_state(repo_root),
        "config": sanitized_config(config_path),
    }


def artifact_inventory(
    paths: Iterable[str | Path],
    relative_to: str | Path | None = None,
) -> list[dict[str, Any]]:
    base = Path(relative_to).resolve() if relative_to is not None else None
    inventory = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            display_path = (
                resolved.relative_to(base).as_posix() if base is not None else str(path)
            )
        except ValueError:
            display_path = str(path)
        inventory.append(
            {
                "path": display_path,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return sorted(inventory, key=lambda item: item["path"])


def snapshot_artifacts(
    paths: Iterable[str | Path],
    source_root: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    """Create an immutable per-run snapshot, hard-linking large binary sources."""
    source = Path(source_root).resolve()
    destination = Path(destination_root).resolve()
    if destination == source:
        raise ValueError("Artifact snapshot destination must differ from source")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_manifest_path = destination / "snapshot_manifest.json"
    if snapshot_manifest_path.exists():
        raise FileExistsError(
            f"Immutable artifact snapshot already exists: {destination}"
        )

    binary_suffixes = {
        ".pdf",
        ".doc",
        ".docx",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }
    records = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = resolved.relative_to(source)
        except ValueError as exc:
            raise ValueError(
                f"Cannot snapshot artifact outside source root: {resolved}"
            ) from exc

        snapshot_path = (destination / relative).resolve()
        try:
            snapshot_path.relative_to(destination)
        except ValueError as exc:
            raise ValueError("Artifact snapshot path escaped destination") from exc
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        source_hash = sha256_file(resolved)
        method = "reused"
        if snapshot_path.exists():
            if not snapshot_path.is_file() or sha256_file(snapshot_path) != source_hash:
                raise FileExistsError(
                    f"Immutable artifact snapshot already exists with different content: "
                    f"{snapshot_path}"
                )
        elif resolved.suffix.casefold() in binary_suffixes:
            try:
                os.link(resolved, snapshot_path)
                method = "hardlink"
            except OSError:
                shutil.copy2(resolved, snapshot_path)
                method = "copy"
        else:
            shutil.copy2(resolved, snapshot_path)
            method = "copy"

        records.append(
            {
                "source_path": str(resolved),
                "snapshot_path": str(snapshot_path),
                "relative_path": relative.as_posix(),
                "size_bytes": snapshot_path.stat().st_size,
                "sha256": source_hash,
                "method": method,
            }
        )

    manifest = {
        "created_at": utc_now(),
        "source_root": str(source),
        "snapshot_root": str(destination),
        "artifacts": sorted(records, key=lambda item: item["relative_path"]),
    }
    atomic_write_json(snapshot_manifest_path, manifest)
    return manifest
