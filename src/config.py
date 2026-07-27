"""Configuration loading, validation, migration, and safe persistence."""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.audit import atomic_write_text, sha256_text, utc_now

SUPPORTED_PROVIDERS = {"openrouter", "ollama"}
SUPPORTED_NOTICE_TYPES = {"RULE", "PRORULE", "NOTICE", "PRESDOCU"}
SUPPORTED_NOTICE_ORDERS = {
    "relevance",
    "newest",
    "oldest",
    "executive_order_number",
}
SUPPORTED_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
COMMENT_EXISTING_ACTIONS = {"update", "skip", "append"}

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 2,
    "application": {
        "artifact_root": "downloads",
        "log_dir": "logs",
        "service_account_file": "service_account.json",
        "sheet_url_env": "GRAIL_SHEET_URL",
    },
    "ai": {
        "provider": "openrouter",
        "model": "openai/gpt-4.1-mini",
        "timeout_seconds": 60,
        "temperature": 0.2,
        "validation_retries": 1,
        "fallback": {
            "provider": "ollama",
            "model": "llama3.1:latest",
            "timeout_seconds": 90,
        },
        "comparison": {
            "enabled": False,
            "models": [],
        },
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
            "max_concurrent_calls": 8,
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "max_concurrent_calls": 1,
        },
    },
    "pipeline": {
        "max_workers": 4,
        "review_threshold": 0.75,
        "log_level": "INFO",
    },
    "notices": {
        "search": {
            "query_terms": [
                "artificial intelligence",
                "machine learning",
                "automated decision-making",
                "algorithmic systems",
            ],
            "topic_terms": [
                "artificial intelligence",
                "machine learning",
                "automated decision-making",
                "algorithm",
                "foundation model",
                "generative AI",
            ],
            "action_terms": [
                "request for information",
                "request for comment",
                "comments requested",
                "proposed rule",
                "notice of proposed rulemaking",
                "seeks comment",
            ],
            "exclude_terms": [],
            "start_date": "2025-01-01",
            "end_date": None,
            "document_types": ["NOTICE", "PRORULE"],
            "order": "newest",
            "per_page": 50,
            "max_pages_per_query": 1,
            "max_results": 25,
            "minimum_topic_matches": 1,
            "require_action_term": False,
            "only_open_for_comment": False,
        },
        "processing": {
            "fetch_full_text": True,
            "ai_analysis": True,
            "max_text_characters": 60_000,
            "max_workers": 4,
            "request_timeout_seconds": 30,
            "request_retries": 3,
            "review_threshold": 0.75,
        },
        "output": {
            "sheet_url_env": "NOTICE_SHEET_URL",
            "export_to_sheet": False,
            "worksheet_name": "",
            "deduplicate_by_fr_document": True,
            "batch_size": 500,
        },
    },
    "comments": {
        "output": {
            "sheet_url_env": "COMMENT_SHEET_URL",
            "export_to_sheet": False,
            "worksheet_title_template": (
                "{fr_document_number} - {notice_title}"
            ),
            "existing_comment_action": "update",
            "batch_size": 500,
        },
    },
    "web": {
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False,
        "background_workers": 2,
        "recent_run_limit": 20,
        "secret_key_env": "GRAIL_WEB_SECRET",
    },
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _legacy_to_current(data: dict[str, Any]) -> dict[str, Any]:
    """Map the original notice keys into version-2 settings when necessary."""
    migrated = copy.deepcopy(data)
    comments = migrated.get("comments")
    if isinstance(comments, dict):
        output = comments.get("output")
        if isinstance(output, dict):
            has_legacy_deduplication = (
                "deduplicate_by_comment_id" in output
            )
            legacy_deduplication = output.pop(
                "deduplicate_by_comment_id",
                None,
            )
            if (
                has_legacy_deduplication
                and not isinstance(legacy_deduplication, bool)
            ):
                raise ValueError(
                    "comments.output.deduplicate_by_comment_id must be "
                    "true or false"
                )
            if (
                "existing_comment_action" not in output
                and isinstance(legacy_deduplication, bool)
            ):
                output["existing_comment_action"] = (
                    "skip" if legacy_deduplication else "append"
                )
    legacy_settings = migrated.get("notice_collection_settings")
    legacy_terms = migrated.get("notice_search_terms")
    if "notices" in migrated or (
        not isinstance(legacy_settings, dict)
        and not isinstance(legacy_terms, dict)
    ):
        return migrated

    notice_config: dict[str, Any] = {"search": {}}
    search = notice_config["search"]
    if isinstance(legacy_settings, dict):
        aliases = {
            "max_notices": "max_results",
            "docket_type": "document_types",
            "order": "order",
            "start_date": "start_date",
        }
        for legacy_key, current_key in aliases.items():
            value = legacy_settings.get(legacy_key)
            if value is None:
                continue
            if current_key == "document_types" and isinstance(value, str):
                value = [value]
            search[current_key] = value
    if isinstance(legacy_terms, dict):
        default_terms = legacy_terms.get("default_terms") or []
        action_terms = legacy_terms.get("search_terms") or []
        if default_terms:
            search["query_terms"] = list(default_terms)
            search["topic_terms"] = list(default_terms)
        if action_terms:
            search["action_terms"] = list(action_terms)
    migrated["notices"] = notice_config
    return migrated


def _expect_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _expect_string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {qualifier}")
    return value.strip()


def _expect_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _expect_int(
    value: Any,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be no greater than {maximum}")
    return value


def _expect_number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return float(value)


def _expect_string_list(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    normalized: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_value = _expect_string(item, f"{field}[{index}]")
        key = item_value.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(item_value)
    if not allow_empty and not normalized:
        raise ValueError(f"{field} must include at least one value")
    return normalized


def _validate_iso_date(value: Any, field: str, *, nullable: bool = False) -> Any:
    if value is None and nullable:
        return None
    normalized = _expect_string(value, field)
    try:
        date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD") from exc
    return normalized


def _validate_model_selection(
    selection: Any,
    field: str,
    *,
    require_timeout: bool = False,
) -> None:
    value = _expect_object(selection, field)
    provider = value.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"{field}.provider must be one of {sorted(SUPPORTED_PROVIDERS)}"
        )
    _expect_string(value.get("model"), f"{field}.model")
    timeout = value.get("timeout_seconds")
    if require_timeout or timeout is not None:
        _expect_int(timeout, f"{field}.timeout_seconds", minimum=1)


def validateConfig(data: Any) -> dict[str, Any]:
    """Validate known settings and return a normalized, defaults-applied copy."""
    root = _expect_object(data, "configuration")
    migrated = _legacy_to_current(root)
    merged = _deep_merge(DEFAULT_CONFIG, migrated)

    version = merged.get("version")
    _expect_int(version, "version", minimum=1)

    application = _expect_object(merged.get("application"), "application")
    for key in (
        "artifact_root",
        "log_dir",
        "service_account_file",
        "sheet_url_env",
    ):
        _expect_string(application.get(key), f"application.{key}")

    ai = _expect_object(merged.get("ai"), "ai")
    _validate_model_selection(ai, "ai", require_timeout=True)
    _expect_number(ai.get("temperature"), "ai.temperature", minimum=0, maximum=2)
    _expect_int(
        ai.get("validation_retries"),
        "ai.validation_retries",
        minimum=0,
        maximum=3,
    )
    fallback = ai.get("fallback")
    if fallback is not None:
        _validate_model_selection(fallback, "ai.fallback", require_timeout=True)
    comparison = _expect_object(ai.get("comparison"), "ai.comparison")
    _expect_bool(comparison.get("enabled"), "ai.comparison.enabled")
    comparison_models = comparison.get("models")
    if not isinstance(comparison_models, list):
        raise ValueError("ai.comparison.models must be a list")
    for index, model in enumerate(comparison_models):
        _validate_model_selection(model, f"ai.comparison.models[{index}]")
    openrouter = _expect_object(ai.get("openrouter"), "ai.openrouter")
    _expect_string(openrouter.get("api_key_env"), "ai.openrouter.api_key_env")
    _expect_string(openrouter.get("base_url"), "ai.openrouter.base_url")
    _expect_int(
        openrouter.get("max_concurrent_calls"),
        "ai.openrouter.max_concurrent_calls",
        minimum=1,
        maximum=64,
    )
    ollama = _expect_object(ai.get("ollama"), "ai.ollama")
    _expect_string(ollama.get("base_url"), "ai.ollama.base_url")
    _expect_int(
        ollama.get("max_concurrent_calls"),
        "ai.ollama.max_concurrent_calls",
        minimum=1,
        maximum=64,
    )

    pipeline = _expect_object(merged.get("pipeline"), "pipeline")
    _expect_int(pipeline.get("max_workers"), "pipeline.max_workers", minimum=1)
    _expect_number(
        pipeline.get("review_threshold"),
        "pipeline.review_threshold",
        minimum=0,
        maximum=1,
    )
    log_level = _expect_string(
        pipeline.get("log_level"),
        "pipeline.log_level",
    ).upper()
    if log_level not in SUPPORTED_LOG_LEVELS:
        raise ValueError(
            f"pipeline.log_level must be one of {sorted(SUPPORTED_LOG_LEVELS)}"
        )
    merged["pipeline"]["log_level"] = log_level

    notices = _expect_object(merged.get("notices"), "notices")
    search = _expect_object(notices.get("search"), "notices.search")
    for key in ("query_terms", "topic_terms", "action_terms"):
        search[key] = _expect_string_list(
            search.get(key),
            f"notices.search.{key}",
        )
    search["exclude_terms"] = _expect_string_list(
        search.get("exclude_terms"),
        "notices.search.exclude_terms",
        allow_empty=True,
    )
    search["start_date"] = _validate_iso_date(
        search.get("start_date"),
        "notices.search.start_date",
    )
    search["end_date"] = _validate_iso_date(
        search.get("end_date"),
        "notices.search.end_date",
        nullable=True,
    )
    if search["end_date"] and search["end_date"] < search["start_date"]:
        raise ValueError(
            "notices.search.end_date cannot be before notices.search.start_date"
        )
    document_types = _expect_string_list(
        search.get("document_types"),
        "notices.search.document_types",
    )
    normalized_types = [value.upper() for value in document_types]
    unsupported_types = set(normalized_types) - SUPPORTED_NOTICE_TYPES
    if unsupported_types:
        raise ValueError(
            "notices.search.document_types contains unsupported values: "
            + ", ".join(sorted(unsupported_types))
        )
    search["document_types"] = normalized_types
    order = _expect_string(search.get("order"), "notices.search.order")
    if order not in SUPPORTED_NOTICE_ORDERS:
        raise ValueError(
            f"notices.search.order must be one of {sorted(SUPPORTED_NOTICE_ORDERS)}"
        )
    _expect_int(search.get("per_page"), "notices.search.per_page", minimum=1, maximum=1000)
    _expect_int(
        search.get("max_pages_per_query"),
        "notices.search.max_pages_per_query",
        minimum=1,
        maximum=50,
    )
    _expect_int(search.get("max_results"), "notices.search.max_results", minimum=1)
    _expect_int(
        search.get("minimum_topic_matches"),
        "notices.search.minimum_topic_matches",
        minimum=0,
    )
    _expect_bool(
        search.get("require_action_term"),
        "notices.search.require_action_term",
    )
    _expect_bool(
        search.get("only_open_for_comment"),
        "notices.search.only_open_for_comment",
    )

    processing = _expect_object(notices.get("processing"), "notices.processing")
    for key in ("fetch_full_text", "ai_analysis"):
        _expect_bool(processing.get(key), f"notices.processing.{key}")
    _expect_int(
        processing.get("max_text_characters"),
        "notices.processing.max_text_characters",
        minimum=1_000,
    )
    _expect_int(
        processing.get("max_workers"),
        "notices.processing.max_workers",
        minimum=1,
    )
    _expect_int(
        processing.get("request_timeout_seconds"),
        "notices.processing.request_timeout_seconds",
        minimum=1,
    )
    _expect_int(
        processing.get("request_retries"),
        "notices.processing.request_retries",
        minimum=0,
        maximum=10,
    )
    _expect_number(
        processing.get("review_threshold"),
        "notices.processing.review_threshold",
        minimum=0,
        maximum=1,
    )
    notice_output = _expect_object(notices.get("output"), "notices.output")
    _expect_string(
        notice_output.get("sheet_url_env"),
        "notices.output.sheet_url_env",
    )
    _expect_bool(
        notice_output.get("export_to_sheet"),
        "notices.output.export_to_sheet",
    )
    _expect_string(
        notice_output.get("worksheet_name"),
        "notices.output.worksheet_name",
        allow_empty=True,
    )
    _expect_bool(
        notice_output.get("deduplicate_by_fr_document"),
        "notices.output.deduplicate_by_fr_document",
    )
    _expect_int(
        notice_output.get("batch_size"),
        "notices.output.batch_size",
        minimum=1,
        maximum=5_000,
    )

    comments = _expect_object(merged.get("comments"), "comments")
    comment_output = _expect_object(comments.get("output"), "comments.output")
    _expect_string(
        comment_output.get("sheet_url_env"),
        "comments.output.sheet_url_env",
    )
    _expect_bool(
        comment_output.get("export_to_sheet"),
        "comments.output.export_to_sheet",
    )
    worksheet_template = _expect_string(
        comment_output.get("worksheet_title_template"),
        "comments.output.worksheet_title_template",
    )
    if not worksheet_template.lstrip().startswith(
        "{fr_document_number}"
    ):
        raise ValueError(
            "comments.output.worksheet_title_template must start with "
            "{fr_document_number} so each notice keeps one stable tab"
        )
    existing_comment_action = _expect_string(
        comment_output.get("existing_comment_action"),
        "comments.output.existing_comment_action",
    )
    if existing_comment_action not in COMMENT_EXISTING_ACTIONS:
        raise ValueError(
            "comments.output.existing_comment_action must be one of "
            f"{sorted(COMMENT_EXISTING_ACTIONS)}"
        )
    _expect_int(
        comment_output.get("batch_size"),
        "comments.output.batch_size",
        minimum=1,
        maximum=5_000,
    )

    web = _expect_object(merged.get("web"), "web")
    _expect_string(web.get("host"), "web.host")
    _expect_int(web.get("port"), "web.port", minimum=1, maximum=65_535)
    _expect_bool(web.get("debug"), "web.debug")
    _expect_int(
        web.get("background_workers"),
        "web.background_workers",
        minimum=1,
        maximum=16,
    )
    _expect_int(web.get("recent_run_limit"), "web.recent_run_limit", minimum=1)
    _expect_string(web.get("secret_key_env"), "web.secret_key_env")

    # Legacy sections are accepted as migration input but are not retained in
    # normalized settings, so every consumer sees one canonical structure.
    merged.pop("notice_collection_settings", None)
    merged.pop("notice_search_terms", None)
    return merged


def loadConfig(
    configPath: str | Path = "config.yaml",
    *,
    applyDefaults: bool = True,
) -> dict[str, Any]:
    path = Path(configPath)
    if not path.is_file():
        if applyDefaults:
            return validateConfig({})
        return {}
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be an object: {path}")
    return validateConfig(data) if applyDefaults else data


def dumpConfig(data: dict[str, Any]) -> str:
    normalized = validateConfig(data)
    return yaml.safe_dump(
        normalized,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def saveConfigText(
    configPath: str | Path,
    yamlText: str,
    *,
    actor: str = "application",
    auditPath: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically save YAML, retaining a last-known-good backup."""
    path = Path(configPath)
    try:
        parsed = yaml.safe_load(yamlText) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc
    normalized = validateConfig(parsed)

    persisted_text = yamlText.rstrip() + "\n"
    before_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    before_hash = sha256_text(before_text) if before_text else None
    after_hash = sha256_text(persisted_text)
    if before_text and before_hash != after_hash:
        atomic_write_text(path.with_suffix(path.suffix + ".bak"), before_text)
    atomic_write_text(path, persisted_text)

    record = {
        "timestamp": utc_now(),
        "actor": actor,
        "config_path": str(path),
        "before_sha256": before_hash,
        "after_sha256": after_hash,
        "changed": before_hash != after_hash,
    }
    if auditPath is not None:
        audit_destination = Path(auditPath)
        audit_destination.parent.mkdir(parents=True, exist_ok=True)
        with audit_destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, sort_keys=True))
            stream.write("\n")
    return {"config": normalized, **record}


def loadSearchTerms(configPath: str | Path = "config.yaml") -> dict[str, list[str]]:
    """Compatibility adapter for the original notice collector."""
    search = loadConfig(configPath)["notices"]["search"]
    return {
        "default_terms": list(search["topic_terms"]),
        "search_terms": list(search["action_terms"]),
    }


def loadSettings(configPath: str | Path = "config.yaml") -> dict[str, Any]:
    """Compatibility adapter for the original notice collector."""
    search = loadConfig(configPath)["notices"]["search"]
    return {
        "maxNotices": search["max_results"],
        "docketType": search["document_types"][0],
        "order": search["order"],
        "startDate": search["start_date"],
    }


def loadPipelineSettings(configPath: str | Path = "config.yaml") -> dict[str, Any]:
    data = loadConfig(configPath)
    pipeline = data["pipeline"]
    comparison = data["ai"]["comparison"]
    return {
        "max_workers": pipeline["max_workers"],
        "review_threshold": float(pipeline["review_threshold"]),
        "log_level": pipeline["log_level"],
        "comparison_models": (
            list(comparison["models"]) if comparison["enabled"] else []
        ),
    }


def loadNoticeSettings(configPath: str | Path = "config.yaml") -> dict[str, Any]:
    data = loadConfig(configPath)
    return copy.deepcopy(data["notices"])


def loadApplicationSettings(configPath: str | Path = "config.yaml") -> dict[str, Any]:
    return copy.deepcopy(loadConfig(configPath)["application"])


def loadWebSettings(configPath: str | Path = "config.yaml") -> dict[str, Any]:
    return copy.deepcopy(loadConfig(configPath)["web"])


def resolveConfigPath(
    configPath: str | Path,
    configuredPath: str | Path,
) -> Path:
    """Resolve a configured filesystem path relative to its YAML file."""
    value = Path(configuredPath)
    if value.is_absolute():
        return value.resolve()
    return (Path(configPath).resolve().parent / value).resolve()


def validateFederalRegisterNumber(value: str) -> str:
    normalized = _expect_string(value, "Federal Register document number")
    if not re.fullmatch(r"\d{4}-\d{4,6}", normalized):
        raise ValueError(
            "Federal Register document number must look like 2026-08281"
        )
    return normalized
