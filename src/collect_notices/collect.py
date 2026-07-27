"""Backward-compatible entry point for notice discovery."""

from __future__ import annotations

from typing import Any

from src.collect_notices.pipeline import runNoticePipeline


def collectNotices(
    configPath: str = "config.yaml",
    downloadRoot: str | None = None,
    spreadsheetUrl: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Run the integrated notice pipeline.

    Historically this function wrote rows straight to ``NOTICE_SHEET_URL``.
    It now preserves local evidence and run artifacts before exporting. Calling
    it without a URL retains the original configured-sheet behavior.
    """
    return runNoticePipeline(
        configPath=configPath,
        downloadRoot=downloadRoot,
        spreadsheetUrl=spreadsheetUrl,
        useConfiguredSheet=spreadsheetUrl is None,
        **options,
    )
