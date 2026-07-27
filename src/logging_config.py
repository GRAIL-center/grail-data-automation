"""Application logging with human-readable console and structured JSONL output."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "grail_log_context",
    default={},
)
_STANDARD_LOG_RECORD_FIELDS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _LOG_CONTEXT.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class _JsonFormatter(logging.Formatter):
    """Serialize one log record per line for searching and ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _utc_timestamp(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if (
                key not in _STANDARD_LOG_RECORD_FIELDS
                and key not in {"message", "asctime"}
                and not key.startswith("_")
            ):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


@contextlib.contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Temporarily add structured context to every log record in this execution context."""
    merged = {**_LOG_CONTEXT.get(), **{key: value for key, value in values.items() if value is not None}}
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def configure_logging(
    log_dir: str | Path = "logs",
    level: str | int = "INFO",
    run_id: str | None = None,
    console: bool = True,
) -> Path:
    """Configure idempotent rotating logs and return the JSONL log path."""
    if isinstance(level, str):
        resolved_level = getattr(logging, level.upper(), None)
        if not isinstance(resolved_level, int):
            raise ValueError(f"Unsupported log level: {level}")
    elif isinstance(level, int):
        resolved_level = level
    else:
        raise TypeError("level must be a logging level name or integer")

    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_log_path = destination / "grail.jsonl"

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    for handler in list(root_logger.handlers):
        if getattr(handler, "_grail_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    context_filter = _ContextFilter()
    json_handler = logging.handlers.RotatingFileHandler(
        json_log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    json_handler._grail_handler = True  # type: ignore[attr-defined]
    json_handler.setLevel(resolved_level)
    json_handler.setFormatter(_JsonFormatter())
    json_handler.addFilter(context_filter)
    root_logger.addHandler(json_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler._grail_handler = True  # type: ignore[attr-defined]
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

    _LOG_CONTEXT.set({"run_id": run_id} if run_id else {})

    logging.getLogger(__name__).info(
        "Logging initialized",
        extra={
            "event": "logging_configured",
            "log_path": str(json_log_path),
            "log_level": logging.getLevelName(resolved_level),
        },
    )
    return json_log_path
