"""Top-level GRAIL collection and comment-analysis workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.audit import (
    ANALYSIS_SCHEMA_VERSION,
    PROMPT_VERSION,
    AuditTrail,
    atomic_write_json,
    create_run_id,
    runtime_manifest,
    utc_now,
    validate_run_id,
)
from src.analyze_comments.ai_analysis import configureAIClient
from src.analyze_comments.api_analysis import SHEET_COLUMNS, analyzeCollectedComments
from src.collect_comments import collectComments
from src.collect_notices.client import fetchFederalRegisterDocument
from src.collect_notices.pipeline import runNoticePipeline
from src.config import (
    loadApplicationSettings,
    loadConfig,
    loadPipelineSettings,
    loadWebSettings,
    resolveConfigPath,
    validateFederalRegisterNumber,
)
from src.export.sheets import (
    buildCommentWorksheetTitle,
    exportMappingRows,
)
from src.logging_config import configure_logging

logger = logging.getLogger(__name__)
FAILURE_PREVIEW_LIMIT = 25


def _configured_sheet_url(
    config: dict[str, Any],
    output: dict[str, Any],
    *,
    explicit_url: str | None,
    requested: bool,
    label: str,
) -> str | None:
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()
    if not requested:
        return None
    output_env = output["sheet_url_env"]
    shared_env = config["application"]["sheet_url_env"]
    resolved = (
        os.getenv(output_env, "").strip()
        or os.getenv(shared_env, "").strip()
    )
    if not resolved:
        raise ValueError(
            f"Google Sheet export is enabled for {label}, but neither "
            f"{output_env} nor shared fallback {shared_env} is set"
        )
    return resolved


def _resolve_notice_context(
    fr_document_number: str,
    comments: list[Any],
    supplied_title: str | None,
) -> dict[str, Any]:
    if supplied_title and supplied_title.strip():
        return {
            "fr_document_number": fr_document_number,
            "title": supplied_title.strip(),
            "title_source": "workflow_or_user_input",
        }
    try:
        federal_register = fetchFederalRegisterDocument(
            fr_document_number,
            timeout=10,
            retries=1,
            fields=["document_number", "title", "html_url"],
        )
    except Exception as exc:
        lookup_error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        title = federal_register.get("title")
        if isinstance(title, str) and title.strip():
            return {
                "fr_document_number": fr_document_number,
                "title": title.strip(),
                "title_source": "federal_register_api",
                "federal_register_url": federal_register.get("html_url"),
            }
        lookup_error = {
            "error_type": "ValueError",
            "error": "Federal Register metadata did not include a title",
        }
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        source = comment.get("source")
        if not isinstance(source, dict):
            continue
        title = source.get("document_title")
        if isinstance(title, str) and title.strip():
            return {
                "fr_document_number": fr_document_number,
                "title": title.strip(),
                "title_source": "regulations_gov_source_document",
                "federal_register_lookup_error": lookup_error,
            }
    return {
        "fr_document_number": fr_document_number,
        "title": "Comments",
        "title_source": "fallback",
        "federal_register_lookup_error": lookup_error,
    }


def parseModelSpec(value: str) -> dict[str, str]:
    """Parse PROVIDER:MODEL while preserving colons inside model tags."""
    provider, separator, model = value.strip().partition(":")
    if not separator or provider not in {"openrouter", "ollama"} or not model.strip():
        raise argparse.ArgumentTypeError(
            "model must use PROVIDER:MODEL (provider is openrouter or ollama)"
        )
    return {"provider": provider, "model": model.strip()}


def collectAndAnalyzeComments(
    frDocumentNumber: str,
    downloadRoot: str | None = None,
    spreadsheetUrl: str | None = None,
    comparisonModels: list[dict[str, Any]] | None = None,
    reviewThreshold: float = 0.75,
    maxWorkers: int = 4,
    runId: str | None = None,
    configPath: str = "config.yaml",
    useConfiguredSheet: bool = False,
    noticeTitle: str | None = None,
) -> dict[str, Any]:
    """Collect comments for an FR document and run the staged text pipeline."""
    fr_document_number = validateFederalRegisterNumber(frDocumentNumber)
    if (
        isinstance(maxWorkers, bool)
        or not isinstance(maxWorkers, int)
        or maxWorkers <= 0
    ):
        raise ValueError("maxWorkers must be a positive integer")
    if (
        isinstance(reviewThreshold, bool)
        or not isinstance(reviewThreshold, (int, float))
        or not 0 <= reviewThreshold <= 1
    ):
        raise ValueError("reviewThreshold must be between 0 and 1")

    config = loadConfig(configPath)
    resolved_download_root = str(
        Path(downloadRoot).resolve()
        if downloadRoot is not None
        else resolveConfigPath(
            configPath,
            config["application"]["artifact_root"],
        )
    )
    comment_output = config["comments"]["output"]
    spreadsheetUrl = _configured_sheet_url(
        config,
        comment_output,
        explicit_url=spreadsheetUrl,
        requested=(
            useConfiguredSheet
            or comment_output["export_to_sheet"]
            or bool(spreadsheetUrl)
        ),
        label="comments",
    )
    run_id = validate_run_id(runId or create_run_id())
    run_dir = Path(resolved_download_root) / "runs" / run_id
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        raise FileExistsError(
            f"Run ID already has a manifest; choose a unique run ID: {run_id}"
        )
    configureAIClient(configPath)
    pipeline_audit = AuditTrail(
        run_dir / "audit.jsonl",
        run_id,
        {"scope": "pipeline"},
    )
    pipeline_audit.record(
        "pipeline_started",
        fr_document_number=fr_document_number,
        sheet_export_requested=bool(spreadsheetUrl),
    )
    initial_manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "status": "collecting",
        "started_at": utc_now(),
        "input": {
            "fr_document_number": fr_document_number,
            "notice_title": noticeTitle,
            "download_root": resolved_download_root,
            "comparison_models": list(comparisonModels or []),
            "review_threshold": reviewThreshold,
            "max_workers": maxWorkers,
            "sheet_export_requested": bool(spreadsheetUrl),
        },
        "runtime": runtime_manifest(Path(__file__).resolve().parents[1], configPath),
        "audit_path": str(pipeline_audit.path),
    }
    atomic_write_json(run_manifest_path, initial_manifest)
    logger.info(
        "Collecting comments for FR document %s",
        fr_document_number,
        extra={
            "event": "comment_collection_started",
            "run_id": run_id,
            "fr_document_number": fr_document_number,
        },
    )
    try:
        comments = collectComments(fr_document_number)
    except Exception as exc:
        initial_manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "failure": {
                    "stage": "comment_collection",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        )
        atomic_write_json(run_manifest_path, initial_manifest)
        pipeline_audit.record(
            "comment_collection_finished",
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    notice_context = _resolve_notice_context(
        fr_document_number,
        comments,
        noticeTitle,
    )
    resolved_notice_title = notice_context["title"]
    atomic_write_json(run_dir / "notice_context.json", notice_context)
    pipeline_audit = AuditTrail(
        run_dir / "audit.jsonl",
        run_id,
        {"scope": "pipeline"},
    )
    pipeline_audit.record(
        "comment_collection_finished",
        fr_document_number=fr_document_number,
        collected_comment_count=len(comments),
        notice_title=resolved_notice_title,
        notice_title_source=notice_context["title_source"],
    )
    try:
        result = analyzeCollectedComments(
            comments,
            resolved_download_root,
            comparisonModels=comparisonModels,
            reviewThreshold=reviewThreshold,
            maxWorkers=maxWorkers,
            runId=run_id,
            configPath=configPath,
        )
    except Exception as exc:
        failure_manifest = (
            json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if run_manifest_path.is_file()
            else initial_manifest
        )
        failure_manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "failure": {
                    "stage": "comment_analysis",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        )
        atomic_write_json(run_manifest_path, failure_manifest)
        AuditTrail(
            pipeline_audit.path,
            run_id,
            {"scope": "pipeline"},
        ).record(
            "batch_analysis_finished",
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    result.setdefault("run_id", run_id)
    result.setdefault("run_manifest_path", str(run_manifest_path))
    result.setdefault("audit_path", str(run_dir / "audit.jsonl"))
    result["fr_document_number"] = fr_document_number
    result["notice_title"] = resolved_notice_title
    result["notice_context"] = notice_context
    result["collected_comment_count"] = len(comments)
    result["sheet_rows_appended"] = 0
    result["sheet_rows_updated"] = 0
    result["sheet_rows_skipped"] = 0
    result["sheet_worksheet_title"] = None
    result["sheet_export"] = {
        "requested": bool(spreadsheetUrl),
        "status": "not_requested",
    }
    if spreadsheetUrl:
        worksheet_title = buildCommentWorksheetTitle(
            fr_document_number,
            resolved_notice_title,
            template=comment_output["worksheet_title_template"],
        )
        try:
            credentials_path = Path(
                config["application"]["service_account_file"]
            )
            if not credentials_path.is_absolute():
                credentials_path = (
                    Path(configPath).resolve().parent / credentials_path
                )
            export_details = exportMappingRows(
                spreadsheetUrl,
                result["rows"],
                credentialsPath=str(credentials_path),
                worksheetTitle=worksheet_title,
                createWorksheet=True,
                reusePrefix=fr_document_number,
                worksheetRows=max(2, len(result["rows"]) + 1),
                worksheetCols=max(1, len(SHEET_COLUMNS)),
                canonicalHeaders=SHEET_COLUMNS,
                deduplicateBy=(
                    "Comment ID"
                    if comment_output["existing_comment_action"] == "skip"
                    else None
                ),
                upsertBy=(
                    "Comment ID"
                    if comment_output["existing_comment_action"] == "update"
                    else None
                ),
                batchSize=comment_output["batch_size"],
            )
        except Exception as exc:
            sheet_failure = {
                "stage": "sheet_export",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            result.setdefault("failures", []).append(sheet_failure)
            result["sheet_export"] = {
                "requested": True,
                "status": "failed",
                "worksheet_title": worksheet_title,
                **sheet_failure,
            }
            AuditTrail(
                result["audit_path"],
                run_id,
                {"scope": "pipeline"},
            ).record(
                "sheet_export_finished",
                status="failed",
                spreadsheet_url=spreadsheetUrl,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            logger.exception(
                "Google Sheets export failed; analysis artifacts were retained"
            )
        else:
            result["sheet_rows_appended"] = export_details["appended"]
            result["sheet_rows_updated"] = export_details.get("updated", 0)
            result["sheet_rows_skipped"] = export_details["skipped"]
            result["sheet_worksheet_title"] = export_details[
                "worksheet_title"
            ]
            result["sheet_export"] = {
                "requested": True,
                "status": "completed",
                **export_details,
            }
            pipeline_audit = AuditTrail(
                result["audit_path"],
                run_id,
                {"scope": "pipeline"},
            )
            pipeline_audit.record(
                "sheet_export_finished",
                spreadsheet_url=spreadsheetUrl,
                row_count=result["sheet_rows_appended"],
                updated_count=result["sheet_rows_updated"],
                skipped_count=result["sheet_rows_skipped"],
                worksheet_title=result["sheet_worksheet_title"],
            )

    run_manifest_path = result.get("run_manifest_path")
    result["status"] = (
        "completed_with_failures"
        if result.get("failures")
        else "completed"
    )
    if run_manifest_path and Path(run_manifest_path).is_file():
        run_manifest = json.loads(
            Path(run_manifest_path).read_text(encoding="utf-8")
        )
        run_manifest["status"] = result["status"]
        run_manifest["finished_at"] = utc_now()
        if isinstance(run_manifest.get("results"), dict):
            run_manifest["results"]["failure_count"] = len(
                result.get("failures", [])
            )
        run_manifest["pipeline"] = {
            "fr_document_number": fr_document_number,
            "notice_title": resolved_notice_title,
            "notice_context": notice_context,
            "collected_comment_count": len(comments),
            "sheet_export_requested": bool(spreadsheetUrl),
            "sheet_rows_appended": result["sheet_rows_appended"],
            "sheet_rows_updated": result["sheet_rows_updated"],
            "sheet_rows_skipped": result["sheet_rows_skipped"],
            "sheet_worksheet_title": result["sheet_worksheet_title"],
            "sheet_export": result["sheet_export"],
            "failure_count": len(result.get("failures", [])),
            "degraded_comment_count": result.get(
                "degraded_comment_count",
                0,
            ),
            "failed_comment_count": result.get(
                "failed_comment_count",
                0,
            ),
            "finished_at": utc_now(),
        }
        atomic_write_json(run_manifest_path, run_manifest)
    return result


def runConnectedWorkflow(
    *,
    analyzeTop: int,
    configPath: str = "config.yaml",
    downloadRoot: str | None = None,
    noticeSheetUrl: str | None = None,
    commentSheetUrl: str | None = None,
    queryTerms: list[str] | None = None,
    maxNotices: int | None = None,
    maxWorkers: int | None = None,
    reviewThreshold: float | None = None,
    comparisonModels: list[dict[str, Any]] | None = None,
    skipNoticeAI: bool = False,
    useConfiguredSheet: bool = False,
    runId: str | None = None,
) -> dict[str, Any]:
    """Discover notices and explicitly hand the top results to comment analysis."""
    if (
        isinstance(analyzeTop, bool)
        or not isinstance(analyzeTop, int)
        or analyzeTop <= 0
    ):
        raise ValueError("analyzeTop must be a positive integer")
    config = loadConfig(configPath)
    root = (
        Path(downloadRoot).resolve()
        if downloadRoot is not None
        else resolveConfigPath(
            configPath,
            config["application"]["artifact_root"],
        )
    )
    parent_run_id = validate_run_id(runId or create_run_id())
    parent_dir = root / "runs" / parent_run_id
    parent_manifest = parent_dir / "connected_workflow.json"
    if parent_manifest.exists():
        raise FileExistsError(
            f"Connected workflow ID already exists: {parent_run_id}"
        )
    parent_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(
        parent_dir / "audit.jsonl",
        parent_run_id,
        {"scope": "connected_workflow"},
    )
    started_at = utc_now()
    base_id = parent_run_id[:105]
    notice_run_id = f"{base_id}-notices"
    intended_notice_manifest = (
        root / "runs" / notice_run_id / "run_manifest.json"
    )
    parent_record: dict[str, Any] = {
        "run_id": parent_run_id,
        "workflow": "connected",
        "status": "running",
        "current_stage": "notice_discovery",
        "started_at": started_at,
        "finished_at": None,
        "input": {
            "analyze_top": analyzeTop,
            "query_terms": list(queryTerms or []),
            "max_notices": maxNotices,
            "max_workers": maxWorkers,
            "review_threshold": reviewThreshold,
            "comparison_models": list(comparisonModels or []),
            "skip_notice_ai": skipNoticeAI,
            "notice_sheet_export_requested": bool(
                noticeSheetUrl or useConfiguredSheet
            ),
            "comment_sheet_export_requested": bool(
                commentSheetUrl
                or noticeSheetUrl
                or useConfiguredSheet
            ),
        },
        "runtime": runtime_manifest(
            Path(__file__).resolve().parents[1],
            configPath,
        ),
        "notice_run": {
            "run_id": notice_run_id,
            "status": "pending",
            "run_manifest_path": str(intended_notice_manifest),
        },
        "selected_document_numbers": [],
        "comment_runs": [],
        "failures": [],
        "audit_path": str(audit.path),
        "manifest_path": str(parent_manifest),
    }
    atomic_write_json(parent_manifest, parent_record)
    audit.record(
        "connected_workflow_started",
        analyze_top=analyzeTop,
    )
    try:
        notice_result = runNoticePipeline(
            configPath=configPath,
            downloadRoot=str(root),
            spreadsheetUrl=noticeSheetUrl,
            useConfiguredSheet=useConfiguredSheet,
            queryTerms=queryTerms,
            maxResults=maxNotices,
            maxWorkers=maxWorkers,
            skipAI=skipNoticeAI,
            comparisonModels=comparisonModels,
            runId=notice_run_id,
        )
    except Exception as exc:
        failure = {
            "stage": "notice_pipeline",
            "child_run_id": notice_run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        parent_record.update(
            {
                "status": "failed",
                "current_stage": "notice_discovery",
                "finished_at": utc_now(),
                "notice_run": {
                    **parent_record["notice_run"],
                    "status": "failed",
                },
                "failures": [failure],
            }
        )
        atomic_write_json(parent_manifest, parent_record)
        audit.record(
            "connected_workflow_finished",
            status="failed",
            notice_count=0,
            comment_run_count=0,
            failure_count=1,
            **failure,
        )
        raise
    selected_records = notice_result["records"][:analyzeTop]
    selected_numbers = [
        record["document_number"]
        for record in selected_records
    ]
    comment_runs = []
    failures = [
        {
            "stage": "notice_processing",
            **failure,
        }
        for failure in notice_result.get("failures", [])
    ]
    parent_record.update(
        {
            "current_stage": "comment_analysis",
            "notice_run": {
                "run_id": notice_result["run_id"],
                "status": notice_result.get(
                    "status",
                    (
                        "completed_with_failures"
                        if notice_result.get("failures")
                        else "completed"
                    ),
                ),
                "selected_count": notice_result["selected_count"],
                "processed_count": len(notice_result["records"]),
                "failure_count": len(notice_result.get("failures", [])),
                "run_manifest_path": notice_result["run_manifest_path"],
            },
            "selected_document_numbers": selected_numbers,
            "failures": list(failures),
        }
    )
    atomic_write_json(parent_manifest, parent_record)
    # Keep one connected run on the settings snapshot loaded at its start.
    pipeline_settings = config["pipeline"]
    resolved_comment_sheet_url = commentSheetUrl or noticeSheetUrl
    for index, notice_record in enumerate(selected_records, start=1):
        document_number = notice_record["document_number"]
        notice_title = (
            notice_record.get("metadata", {}).get("title")
            or "Comments"
        )
        child_run_id = f"{base_id}-comments-{index:02d}"
        try:
            comment_result = collectAndAnalyzeComments(
                document_number,
                str(root),
                resolved_comment_sheet_url,
                comparisonModels=comparisonModels,
                reviewThreshold=(
                    reviewThreshold
                    if reviewThreshold is not None
                    else pipeline_settings["review_threshold"]
                ),
                maxWorkers=(
                    maxWorkers
                    if maxWorkers is not None
                    else pipeline_settings["max_workers"]
                ),
                runId=child_run_id,
                configPath=configPath,
                useConfiguredSheet=useConfiguredSheet,
                noticeTitle=notice_title,
            )
            child_summary = {
                "fr_document_number": document_number,
                "notice_title": notice_title,
                "run_id": comment_result["run_id"],
                "status": comment_result.get("status"),
                "collected_comment_count": comment_result[
                    "collected_comment_count"
                ],
                "failure_count": len(comment_result["failures"]),
                "sheet_worksheet_title": comment_result.get(
                    "sheet_worksheet_title"
                ),
                "sheet_rows_appended": comment_result.get(
                    "sheet_rows_appended",
                    0,
                ),
                "sheet_rows_updated": comment_result.get(
                    "sheet_rows_updated",
                    0,
                ),
                "sheet_rows_skipped": comment_result.get(
                    "sheet_rows_skipped",
                    0,
                ),
                "run_manifest_path": comment_result["run_manifest_path"],
            }
            comment_runs.append(child_summary)
            audit.record(
                "connected_comment_run_finished",
                status=comment_result.get("status", "completed"),
                fr_document_number=document_number,
                child_run_id=comment_result["run_id"],
                failure_count=len(comment_result.get("failures", [])),
                sheet_worksheet_title=comment_result.get(
                    "sheet_worksheet_title"
                ),
                sheet_rows_appended=comment_result.get(
                    "sheet_rows_appended",
                    0,
                ),
                sheet_rows_updated=comment_result.get(
                    "sheet_rows_updated",
                    0,
                ),
                sheet_rows_skipped=comment_result.get(
                    "sheet_rows_skipped",
                    0,
                ),
            )
            for child_failure in comment_result.get("failures", []):
                failures.append(
                    {
                        "stage": "comment_processing",
                        "fr_document_number": document_number,
                        "child_run_id": comment_result["run_id"],
                        **child_failure,
                    }
                )
        except Exception as exc:
            failure = {
                "stage": "comment_pipeline",
                "fr_document_number": document_number,
                "child_run_id": child_run_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            audit.record(
                "connected_comment_run_finished",
                status="failed",
                **failure,
            )
        parent_record.update(
            {
                "comment_runs": list(comment_runs),
                "failures": list(failures),
            }
        )
        atomic_write_json(parent_manifest, parent_record)
    parent_record.update(
        {
            "status": (
                "completed"
                if not failures
                else "completed_with_failures"
            ),
            "current_stage": "finished",
            "finished_at": utc_now(),
            "comment_runs": comment_runs,
            "failures": failures,
        }
    )
    atomic_write_json(parent_manifest, parent_record)
    audit.record(
        "connected_workflow_finished",
        status=parent_record["status"],
        notice_count=len(selected_numbers),
        comment_run_count=len(comment_runs),
        failure_count=len(failures),
    )
    return parent_record


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML configuration path (default: config.yaml).",
    )
    parser.add_argument(
        "--download-root",
        help="Artifact directory; defaults to application.artifact_root.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique run ID. A generated ID is used when omitted.",
    )
    parser.add_argument(
        "--log-dir",
        help="Structured log directory; defaults to application.log_dir.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console and JSONL logging level.",
    )


def _add_comparison_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-model",
        action="append",
        type=parseModelSpec,
        help=(
            "Run an independent extraction with PROVIDER:MODEL. "
            "Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--no-model-comparison",
        action="store_true",
        help="Disable comparison models configured in config.yaml.",
    )


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover Federal Register notices, analyze Regulations.gov "
            "comments, or run the local Flask control panel."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    notices = subparsers.add_parser(
        "notices",
        help="Discover and analyze relevant Federal Register notices.",
    )
    _add_runtime_options(notices)
    _add_comparison_options(notices)
    notices.add_argument(
        "--query",
        action="append",
        help="Override configured Federal Register query terms; repeat as needed.",
    )
    notices.add_argument(
        "--max-notices",
        type=int,
        help="Maximum relevant notices to process.",
    )
    notices.add_argument(
        "--workers",
        type=int,
        help="Concurrent notice workers.",
    )
    notices.add_argument(
        "--skip-ai",
        action="store_true",
        help="Collect notices and use deterministic excerpts without AI analysis.",
    )
    notices.add_argument("--sheet-url", help="Optional notice Google Sheet URL.")
    notices.add_argument(
        "--configured-sheet",
        action="store_true",
        help="Export using the environment variable named in notices.output.",
    )

    comments = subparsers.add_parser(
        "comments",
        help="Collect and analyze comments for one Federal Register document.",
    )
    _add_runtime_options(comments)
    _add_comparison_options(comments)
    comments.add_argument("fr_document_number")
    comments.add_argument(
        "--notice-title",
        help=(
            "Optional authoritative notice title for the Google Sheet tab; "
            "normally inferred automatically."
        ),
    )
    comments.add_argument("--sheet-url", help="Optional comment Google Sheet URL.")
    comments.add_argument(
        "--configured-sheet",
        action="store_true",
        help="Export using the environment variable named in comments.output.",
    )
    comments.add_argument("--workers", type=int, help="Concurrent comment workers.")
    comments.add_argument(
        "--review-threshold",
        type=float,
        help="Flag populated fields below this confidence (0-1).",
    )

    connected = subparsers.add_parser(
        "workflow",
        help="Discover notices, then analyze comments for the top results.",
    )
    _add_runtime_options(connected)
    _add_comparison_options(connected)
    connected.add_argument(
        "--analyze-top",
        type=int,
        required=True,
        help="Number of top-ranked notices to hand to comment analysis.",
    )
    connected.add_argument("--query", action="append")
    connected.add_argument("--max-notices", type=int)
    connected.add_argument("--workers", type=int)
    connected.add_argument("--review-threshold", type=float)
    connected.add_argument("--skip-notice-ai", action="store_true")
    connected.add_argument("--notice-sheet-url")
    connected.add_argument("--comment-sheet-url")
    connected.add_argument(
        "--configured-sheet",
        action="store_true",
        help=(
            "Export notices and per-notice comment tabs using configured "
            "Google Sheet environment variables."
        ),
    )

    web = subparsers.add_parser(
        "web",
        help="Run the local Flask control panel.",
    )
    web.add_argument("--config", default="config.yaml")
    web.add_argument("--host")
    web.add_argument("--port", type=int)
    web.add_argument("--debug", action="store_true")
    web.add_argument("--log-dir")
    web.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def _resolved_models(args: argparse.Namespace, settings: dict[str, Any]) -> list[dict[str, Any]]:
    if getattr(args, "no_model_comparison", False):
        return []
    explicit = getattr(args, "compare_model", None)
    return explicit if explicit is not None else settings["comparison_models"]


def _summary_for_comments(result: dict[str, Any]) -> dict[str, Any]:
    review_count = sum(
        1
        for record in result["records"]
        if record.get("analysis_record", {})
        .get("analysis", {})
        .get("review", {})
        .get("required")
    )
    failures = result.get("failures", [])
    return {
        "run_id": result["run_id"],
        "workflow": "comments",
        "status": result.get("status", "completed"),
        "fr_document_number": result["fr_document_number"],
        "notice_title": result.get("notice_title"),
        "collected_comment_count": result["collected_comment_count"],
        "analyzed_row_count": len(result["rows"]),
        "failure_count": len(failures),
        "degraded_comment_count": result.get(
            "degraded_comment_count",
            0,
        ),
        "failed_comment_count": result.get(
            "failed_comment_count",
            0,
        ),
        "failures": failures[:FAILURE_PREVIEW_LIMIT],
        "failures_truncated": max(
            0,
            len(failures) - FAILURE_PREVIEW_LIMIT,
        ),
        "review_required_count": review_count,
        "sheet_rows_appended": result["sheet_rows_appended"],
        "sheet_rows_updated": result.get("sheet_rows_updated", 0),
        "sheet_rows_skipped": result.get("sheet_rows_skipped", 0),
        "sheet_worksheet_title": result.get("sheet_worksheet_title"),
        "sheet_export": result.get("sheet_export"),
        "manifest_path": result["manifest_path"],
        "run_manifest_path": result["run_manifest_path"],
        "audit_path": result["audit_path"],
    }


def _summary_for_notices(result: dict[str, Any]) -> dict[str, Any]:
    review_count = sum(
        1
        for record in result["records"]
        if record.get("analysis", {}).get("review", {}).get("required")
    )
    failures = result.get("failures", [])
    return {
        "run_id": result["run_id"],
        "workflow": "notices",
        "status": result.get("status", "completed"),
        "candidate_count": result["candidate_count"],
        "selected_count": result["selected_count"],
        "processed_count": len(result["records"]),
        "failure_count": len(failures),
        "failures": failures[:FAILURE_PREVIEW_LIMIT],
        "failures_truncated": max(
            0,
            len(failures) - FAILURE_PREVIEW_LIMIT,
        ),
        "review_required_count": review_count,
        "sheet_rows_appended": result["sheet_rows_appended"],
        "manifest_path": result["manifest_path"],
        "run_manifest_path": result["run_manifest_path"],
        "audit_path": result["audit_path"],
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {"notices", "comments", "workflow", "web"}
    # Preserve the original invocation: ``python -m src.pipeline 2026-08281``.
    if arguments and arguments[0] not in commands and not arguments[0].startswith("-"):
        arguments.insert(0, "comments")
    parser = buildParser()
    args = parser.parse_args(arguments)

    if args.command == "web":
        try:
            web_settings = loadWebSettings(args.config)
            application = loadApplicationSettings(args.config)
            configure_logging(
                str(
                    Path(args.log_dir).resolve()
                    if args.log_dir
                    else resolveConfigPath(
                        args.config,
                        application["log_dir"],
                    )
                ),
                args.log_level or loadPipelineSettings(args.config)["log_level"],
            )
            from src.web import createApp

            app = createApp(args.config)
            app.run(
                host=args.host or web_settings["host"],
                port=args.port or web_settings["port"],
                debug=args.debug or web_settings["debug"],
                use_reloader=False,
            )
            return 0
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
            return 2

    run_id = args.run_id or create_run_id()
    download_root = args.download_root
    try:
        validate_run_id(run_id)
        application = loadApplicationSettings(args.config)
        settings = loadPipelineSettings(args.config)
        download_root = str(
            Path(download_root).resolve()
            if download_root
            else resolveConfigPath(
                args.config,
                application["artifact_root"],
            )
        )
        comparison_models = _resolved_models(args, settings)
        configure_logging(
            str(
                Path(args.log_dir).resolve()
                if args.log_dir
                else resolveConfigPath(
                    args.config,
                    application["log_dir"],
                )
            ),
            args.log_level or settings["log_level"],
            run_id,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2

    try:
        if args.command == "notices":
            result = runNoticePipeline(
                configPath=args.config,
                downloadRoot=download_root,
                spreadsheetUrl=args.sheet_url,
                useConfiguredSheet=args.configured_sheet,
                queryTerms=args.query,
                maxResults=args.max_notices,
                maxWorkers=args.workers,
                skipAI=args.skip_ai,
                comparisonModels=comparison_models,
                runId=run_id,
            )
            summary = _summary_for_notices(result)
        elif args.command == "comments":
            result = collectAndAnalyzeComments(
                args.fr_document_number,
                download_root,
                args.sheet_url,
                comparisonModels=comparison_models,
                reviewThreshold=(
                    args.review_threshold
                    if args.review_threshold is not None
                    else settings["review_threshold"]
                ),
                maxWorkers=(
                    args.workers
                    if args.workers is not None
                    else settings["max_workers"]
                ),
                runId=run_id,
                configPath=args.config,
                useConfiguredSheet=args.configured_sheet,
                noticeTitle=args.notice_title,
            )
            summary = _summary_for_comments(result)
        else:
            result = runConnectedWorkflow(
                analyzeTop=args.analyze_top,
                configPath=args.config,
                downloadRoot=download_root,
                noticeSheetUrl=args.notice_sheet_url,
                commentSheetUrl=args.comment_sheet_url,
                queryTerms=args.query,
                maxNotices=args.max_notices,
                maxWorkers=args.workers,
                reviewThreshold=args.review_threshold,
                comparisonModels=comparison_models,
                skipNoticeAI=args.skip_notice_ai,
                useConfiguredSheet=args.configured_sheet,
                runId=run_id,
            )
            summary = result
    except Exception as exc:
        logger.exception("%s workflow failed", args.command)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "workflow": args.command,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run_manifest_path": str(
                        Path(download_root) / "runs" / run_id / "run_manifest.json"
                    ),
                },
                indent=2,
            )
        )
        return 2

    print(json.dumps(summary, indent=2, default=str))
    failures = result.get("failures", [])
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
