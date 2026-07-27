"""Audited Federal Register notice discovery and analysis pipeline."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.ai_client import AIClient
from src.audit import (
    PROMPT_VERSION,
    AuditTrail,
    artifact_inventory,
    atomic_write_json,
    atomic_write_text,
    create_run_id,
    runtime_manifest,
    sha256_file,
    sha256_text,
    snapshot_artifacts,
    utc_now,
    validate_run_id,
)
from src.collect_notices.analysis import (
    NOTICE_ANALYSIS_SCHEMA_VERSION,
    NOTICE_AI_FIELDS,
    analyzeNotice,
    applyComparisonReviews,
    buildFieldMetadata,
    compareNoticeAnalyses,
    emptyNoticeAnalysis,
    saveCallEnvelopes,
    successfulModels,
    summarizeReview,
)
from src.collect_notices.client import (
    buildSession,
    fetchNoticeDetail,
    rankNotices,
    searchFederalRegister,
)
from src.config import loadConfig, resolveConfigPath
from src.export.sheets import exportMappingRows
from src.logging_config import log_context

logger = logging.getLogger(__name__)

NOTICE_ID_PATTERN = re.compile(r"^\d{4}-\d{4,6}$")
NOTICE_SHEET_COLUMNS = (
    "FR Document Number",
    "Title",
    "Agency",
    "Document Type",
    "Publication Date",
    "Comments Close On",
    "Open for Comment",
    "Relevance Score",
    "Matched Query Terms",
    "Matched Topics",
    "Matched Action Terms",
    "Brief Summary",
    "Relevant Topics",
    "Agency Objectives",
    "Questions for Commenters",
    "Affected Stakeholders",
    "Key Requirements or Proposals",
    "Docket IDs",
    "Federal Register URL",
    "PDF URL",
    "Regulations.gov URL",
    "Overall Confidence",
    "Review Required",
    "Review Fields",
    "Review Reasons",
    "Model Comparison",
    "Analysis Artifact",
    "Audit Trail",
    "Reproducibility Manifest",
    "Snapshot Manifest",
    "Run ID",
)


def _join(values: Any) -> str:
    if values is None:
        return ""
    if not isinstance(values, list):
        values = [values]
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("raw_name") or str(value)
        normalized = str(value or "").strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return " | ".join(output)


def _analysis_values(analysis: dict[str, Any], field: str) -> list[str]:
    raw = analysis.get(field)
    if field == "brief_summary":
        raw = [raw] if isinstance(raw, dict) else []
    if not isinstance(raw, list):
        return []
    return [
        str(item.get("value")).strip()
        for item in raw
        if isinstance(item, dict) and item.get("value")
    ]


def _agency_names(metadata: dict[str, Any]) -> list[str]:
    direct = metadata.get("agency_names")
    if isinstance(direct, list) and direct:
        return [str(value) for value in direct if value]
    names = []
    for agency in metadata.get("agencies") or []:
        if isinstance(agency, dict):
            name = agency.get("name") or agency.get("raw_name")
            if name:
                names.append(str(name))
    return names


def _fallback_analysis(
    metadata: dict[str, Any],
    body_text: str,
    reason: str,
) -> dict[str, Any]:
    analysis = emptyNoticeAnalysis()
    excerpt = str(metadata.get("abstract") or "").strip()
    if not excerpt:
        excerpt = body_text[:1_500].strip()
    if not excerpt:
        excerpt = str(metadata.get("title") or "").strip()
    if excerpt:
        analysis["brief_summary"] = {
            "value": excerpt,
            "evidence": excerpt,
            "confidence": 1.0,
            "inferred": False,
        }
    analysis["analysis_notes"] = [reason]
    return analysis


def _source_text(
    metadata: dict[str, Any],
    body_text: str,
    max_characters: int,
) -> tuple[str, bool]:
    sections = []
    for label, value in (
        ("TITLE", metadata.get("title")),
        ("ACTION", metadata.get("action")),
        ("ABSTRACT", metadata.get("abstract")),
        ("DATES", metadata.get("dates")),
    ):
        if value:
            sections.append(f"--- {label} ---\n{value}")
    if body_text:
        sections.append(f"--- FULL NOTICE TEXT ---\n{body_text}")
    combined = "\n\n".join(sections)
    if len(combined) <= max_characters:
        return combined, False
    head_size = max_characters * 3 // 4
    tail_size = max_characters - head_size
    return (
        combined[:head_size]
        + "\n\n--- TEXT TRUNCATED FOR MODEL CONTEXT ---\n\n"
        + combined[-tail_size:],
        True,
    )


def _comparison_summary(comparisons: list[dict[str, Any]]) -> str:
    summaries = []
    for comparison in comparisons:
        model = comparison.get("model") or {}
        label = f"{model.get('provider', '')}/{model.get('model', '')}".strip("/")
        score = comparison.get("agreement_score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        summaries.append(
            f"{label or 'comparison'}: {comparison.get('status', 'unknown')} "
            f"({score_text})"
        )
    return _join(summaries)


def buildNoticeSheetRow(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    discovery = record["discovery"]
    analysis = record["analysis"]
    review = analysis["review"]
    open_for_comment = discovery["relevance"].get("open_for_comment")
    row = {column: "" for column in NOTICE_SHEET_COLUMNS}
    row.update(
        {
            "FR Document Number": metadata.get("document_number") or "",
            "Title": metadata.get("title") or "",
            "Agency": _join(_agency_names(metadata)),
            "Document Type": metadata.get("type") or "",
            "Publication Date": metadata.get("publication_date") or "",
            "Comments Close On": metadata.get("comments_close_on") or "",
            "Open for Comment": (
                open_for_comment
                if isinstance(open_for_comment, bool)
                else ""
            ),
            "Relevance Score": discovery["relevance"].get("score", 0),
            "Matched Query Terms": _join(
                discovery.get("matched_query_terms", [])
            ),
            "Matched Topics": _join(
                discovery["relevance"].get("matched_topics", [])
            ),
            "Matched Action Terms": _join(
                discovery["relevance"].get("matched_action_terms", [])
            ),
            "Brief Summary": _join(
                _analysis_values(analysis, "brief_summary")
            ),
            "Relevant Topics": _join(
                _analysis_values(analysis, "relevant_topics")
            ),
            "Agency Objectives": _join(
                _analysis_values(analysis, "agency_objectives")
            ),
            "Questions for Commenters": _join(
                _analysis_values(analysis, "questions_for_commenters")
            ),
            "Affected Stakeholders": _join(
                _analysis_values(analysis, "affected_stakeholders")
            ),
            "Key Requirements or Proposals": _join(
                _analysis_values(
                    analysis,
                    "key_requirements_or_proposals",
                )
            ),
            "Docket IDs": _join(
                metadata.get("docket_ids") or metadata.get("docket_id")
            ),
            "Federal Register URL": metadata.get("html_url") or "",
            "PDF URL": metadata.get("pdf_url") or "",
            "Regulations.gov URL": (
                metadata.get("regulations_dot_gov_url")
                or metadata.get("comment_url")
                or ""
            ),
            "Overall Confidence": review.get("overall_confidence", ""),
            "Review Required": review.get("required", False),
            "Review Fields": _join(review.get("fields", [])),
            "Review Reasons": _join(review.get("reasons", [])),
            "Model Comparison": _comparison_summary(
                analysis.get("model_comparisons", [])
            ),
            "Analysis Artifact": record.get("analysis_path", ""),
            "Audit Trail": record.get("audit_path", ""),
            "Reproducibility Manifest": record.get(
                "reproducibility_path",
                "",
            ),
            "Snapshot Manifest": record.get("snapshot_manifest_path", ""),
            "Run ID": record.get("run_id", ""),
        }
    )
    return row


def _deduplicate_models(models: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for model in models or []:
        if not isinstance(model, dict):
            raise ValueError("Comparison model must be an object")
        provider = model.get("provider")
        name = model.get("model")
        if provider not in {"openrouter", "ollama"}:
            raise ValueError(f"Unsupported comparison provider: {provider}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Comparison model name cannot be empty")
        key = (provider, name.strip())
        if key in seen:
            continue
        seen.add(key)
        output.append({**model, "model": name.strip()})
    return output


def _validate_overrides(
    query_terms: list[str] | None,
    max_results: int | None,
    max_workers: int | None,
) -> None:
    if query_terms is not None and (
        not query_terms
        or any(not isinstance(term, str) or not term.strip() for term in query_terms)
    ):
        raise ValueError("queryTerms must contain non-empty strings")
    for value, name in (
        (max_results, "maxResults"),
        (max_workers, "maxWorkers"),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")


def runNoticePipeline(
    *,
    configPath: str = "config.yaml",
    downloadRoot: str | None = None,
    spreadsheetUrl: str | None = None,
    useConfiguredSheet: bool = False,
    queryTerms: list[str] | None = None,
    maxResults: int | None = None,
    maxWorkers: int | None = None,
    skipAI: bool = False,
    comparisonModels: list[dict[str, Any]] | None = None,
    runId: str | None = None,
) -> dict[str, Any]:
    """Discover, analyze, audit, and optionally export Federal Register notices."""
    _validate_overrides(queryTerms, maxResults, maxWorkers)
    config = loadConfig(configPath)
    settings = copy.deepcopy(config["notices"])
    if queryTerms is not None:
        settings["search"]["query_terms"] = [term.strip() for term in queryTerms]
    if maxResults is not None:
        settings["search"]["max_results"] = maxResults
    if maxWorkers is not None:
        settings["processing"]["max_workers"] = maxWorkers
    if skipAI:
        settings["processing"]["ai_analysis"] = False

    comparison_config = config["ai"]["comparison"]
    comparison_models = _deduplicate_models(
        comparisonModels
        if comparisonModels is not None
        else (
            comparison_config["models"]
            if comparison_config["enabled"]
            else []
        )
    )
    root = (
        Path(downloadRoot).resolve()
        if downloadRoot is not None
        else resolveConfigPath(
            configPath,
            config["application"]["artifact_root"],
        )
    )
    run_id = validate_run_id(runId or create_run_id())
    run_dir = root / "runs" / run_id
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        raise FileExistsError(
            f"Run ID already has a manifest; choose a unique run ID: {run_id}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(run_dir / "audit.jsonl", run_id, {"scope": "notices"})
    started_at = utc_now()
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "workflow": "notice_discovery",
        "notice_analysis_schema_version": NOTICE_ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "status": "searching",
        "started_at": started_at,
        "input": {
            "download_root": str(root),
            "settings": settings,
            "comparison_models": comparison_models,
            "sheet_export_requested": bool(spreadsheetUrl or useConfiguredSheet),
        },
        "runtime": runtime_manifest(
            Path(__file__).resolve().parents[2],
            configPath,
        ),
        "audit_path": str(audit.path),
    }
    atomic_write_json(run_manifest_path, manifest)
    audit.record(
        "notice_pipeline_started",
        query_terms=settings["search"]["query_terms"],
        max_results=settings["search"]["max_results"],
    )
    logger.info(
        "Notice discovery started with %d query term(s)",
        len(settings["search"]["query_terms"]),
        extra={
            "event": "notice_pipeline_started",
            "run_id": run_id,
            "workflow": "notice_discovery",
        },
    )

    try:
        search_result = searchFederalRegister(
            settings,
            run_dir,
            audit=audit,
        )
        ranked = rankNotices(search_result["candidates"], settings)
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "failure": {
                    "stage": "notice_search",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            }
        )
        atomic_write_json(run_manifest_path, manifest)
        audit.record(
            "notice_pipeline_finished",
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise

    candidate_manifest = run_dir / "notice_candidates.json"
    atomic_write_json(
        candidate_manifest,
        {
            "run_id": run_id,
            "candidate_count": len(search_result["candidates"]),
            "selected_count": len(ranked),
            "selected": ranked,
        },
    )
    audit.record(
        "notice_search_finished",
        candidate_count=len(search_result["candidates"]),
        selected_count=len(ranked),
        candidate_manifest_path=str(candidate_manifest),
    )

    ai_client = AIClient(configPath) if settings["processing"]["ai_analysis"] else None
    worker_count = settings["processing"]["max_workers"]
    review_threshold = settings["processing"]["review_threshold"]
    max_attempts = config["ai"]["validation_retries"] + 1

    def process_one(candidate: dict[str, Any]) -> dict[str, Any]:
        document_number = candidate.get("document_number")
        if not isinstance(document_number, str) or not NOTICE_ID_PATTERN.fullmatch(
            document_number
        ):
            return {
                "failure": {
                    "document_number": document_number,
                    "error_type": "ValueError",
                    "error": "Candidate has an invalid Federal Register document number",
                }
            }
        notice_root = (
            root
            / ".work"
            / sha256_text(run_id)[:12]
            / "notices"
        ).resolve()
        artifact_dir = (notice_root / document_number).resolve()
        try:
            artifact_dir.relative_to(notice_root)
        except ValueError:
            return {
                "failure": {
                    "document_number": document_number,
                    "error_type": "ValueError",
                    "error": "Notice artifact path escaped its root",
                }
            }
        notice_audit = AuditTrail(
            artifact_dir / "audit" / f"{run_id}.jsonl",
            run_id,
            {
                "scope": "notice",
                "document_number": document_number,
            },
        )
        with log_context(
            run_id=run_id,
            document_number=document_number,
            stage="notice_analysis",
        ):
            try:
                notice_audit.record("notice_processing_started")
                session = buildSession(settings["processing"]["request_retries"])
                fetched = fetchNoticeDetail(
                    document_number,
                    settings,
                    artifact_dir,
                    audit=notice_audit,
                    session=session,
                )
                metadata = fetched["metadata"]
                discovery = {
                    "matched_query_terms": candidate.get("_discovery", {}).get(
                        "matched_query_terms",
                        [],
                    ),
                    "search_artifacts": candidate.get("_discovery", {}).get(
                        "search_artifacts",
                        [],
                    ),
                    "relevance": candidate.get("_relevance", {}),
                }
                source_record = {
                    **metadata,
                    "_discovery": discovery,
                }
                source_record_path = artifact_dir / "source_record.json"
                atomic_write_json(source_record_path, source_record)
                source_text, text_truncated = _source_text(
                    metadata,
                    fetched["body_text"],
                    settings["processing"]["max_text_characters"],
                )
                model_input_path = artifact_dir / "model_input.txt"
                atomic_write_text(model_input_path, source_text)

                analysis_mode = "ai_extraction"
                additional_review_reasons: list[str] = []
                primary_envelopes: list[dict[str, Any]] = []
                primary_paths: list[str] = []
                if text_truncated:
                    additional_review_reasons.append("model_context_truncated")
                if fetched["body_error"]:
                    additional_review_reasons.append("full_text_unavailable")
                if (
                    discovery["relevance"].get("matched_action_terms")
                    and not metadata.get("comments_close_on")
                ):
                    additional_review_reasons.append(
                        "comment_deadline_unknown"
                    )

                if ai_client is not None and source_text.strip():
                    try:
                        analysis, _ = analyzeNotice(
                            ai_client,
                            source_record,
                            source_text,
                            maxAttempts=max_attempts,
                            callContext={
                                "workflow": "notice_discovery",
                                "run_id": run_id,
                                "document_number": document_number,
                                "stage": "notice_primary_analysis",
                            },
                            callEnvelopes=primary_envelopes,
                        )
                        primary_paths = saveCallEnvelopes(
                            artifact_dir / "ai",
                            "primary",
                            primary_envelopes,
                        )
                    except Exception as exc:
                        logger.exception(
                            "AI notice analysis failed for %s",
                            document_number,
                        )
                        if primary_envelopes:
                            primary_paths = saveCallEnvelopes(
                                artifact_dir / "ai",
                                "primary",
                                primary_envelopes,
                            )
                        analysis = _fallback_analysis(
                            metadata,
                            fetched["body_text"],
                            f"AI analysis failed: {type(exc).__name__}: {exc}",
                        )
                        analysis_mode = "deterministic_fallback"
                        additional_review_reasons.append("ai_analysis_failed")
                else:
                    analysis = _fallback_analysis(
                        metadata,
                        fetched["body_text"],
                        "AI analysis was disabled or no source text was available.",
                    )
                    analysis_mode = "deterministic_fallback"
                    if not settings["processing"]["ai_analysis"]:
                        additional_review_reasons.append("ai_analysis_disabled")
                    if not source_text.strip():
                        additional_review_reasons.append("source_text_empty")

                models = successfulModels(primary_envelopes)
                notice_audit.record(
                    "notice_ai_analysis_finished",
                    status=(
                        "ok"
                        if analysis_mode == "ai_extraction"
                        else "review_required"
                    ),
                    analysis_mode=analysis_mode,
                    models=models,
                    call_artifact_paths=primary_paths,
                )
                comparisons: list[dict[str, Any]] = []
                comparison_paths: list[str] = []
                if (
                    analysis_mode == "ai_extraction"
                    and ai_client is not None
                ):
                    for index, model in enumerate(comparison_models, start=1):
                        label = (
                            f"comparison_{index:02d}_"
                            f"{model['provider']}_{re.sub(r'[^A-Za-z0-9._-]+', '-', model['model'])[:50]}"
                        )
                        comparison_envelopes: list[dict[str, Any]] = []
                        try:
                            comparison_analysis, _ = analyzeNotice(
                                ai_client,
                                source_record,
                                source_text,
                                modelSettings=model,
                                maxAttempts=max_attempts,
                                callContext={
                                    "workflow": "notice_discovery",
                                    "run_id": run_id,
                                    "document_number": document_number,
                                    "stage": "notice_model_comparison",
                                },
                                callEnvelopes=comparison_envelopes,
                            )
                            comparison = compareNoticeAnalyses(
                                analysis,
                                comparison_analysis,
                                model,
                            )
                            comparison["analysis"] = comparison_analysis
                        except Exception as exc:
                            comparison = {
                                "model": {
                                    "provider": model.get("provider"),
                                    "model": model.get("model"),
                                },
                                "status": "failed",
                                "agreement_score": None,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        comparison_paths.extend(
                            saveCallEnvelopes(
                                artifact_dir / "ai" / "comparisons",
                                label,
                                comparison_envelopes,
                            )
                        )
                        comparison_path = (
                            artifact_dir
                            / "ai"
                            / "comparisons"
                            / f"{label}.json"
                        )
                        atomic_write_json(comparison_path, comparison)
                        comparison_paths.append(str(comparison_path))
                        comparisons.append(comparison)
                        notice_audit.record(
                            "notice_model_comparison_finished",
                            status=(
                                "failed"
                                if comparison.get("status") == "failed"
                                else "ok"
                            ),
                            model=comparison.get("model"),
                            agreement_score=comparison.get("agreement_score"),
                            comparison_status=comparison.get("status"),
                            artifact_path=str(comparison_path),
                        )

                deterministic_fields = {
                    "document_number": (
                        metadata.get("document_number"),
                        "/document_number",
                    ),
                    "title": (metadata.get("title"), "/title"),
                    "agency_names": (
                        _agency_names(metadata),
                        (
                            "/agency_names"
                            if metadata.get("agency_names")
                            else "/agencies"
                        ),
                    ),
                    "document_type": (metadata.get("type"), "/type"),
                    "publication_date": (
                        metadata.get("publication_date"),
                        "/publication_date",
                    ),
                    "comments_close_on": (
                        metadata.get("comments_close_on"),
                        "/comments_close_on",
                    ),
                    "docket_ids": (
                        metadata.get("docket_ids")
                        or metadata.get("docket_id"),
                        (
                            "/docket_ids"
                            if metadata.get("docket_ids")
                            else "/docket_id"
                        ),
                    ),
                    "federal_register_url": (
                        metadata.get("html_url"),
                        "/html_url",
                    ),
                    "pdf_url": (metadata.get("pdf_url"), "/pdf_url"),
                    "regulations_gov_url": (
                        metadata.get("regulations_dot_gov_url"),
                        "/regulations_dot_gov_url",
                    ),
                    "relevance_score": (
                        discovery["relevance"].get("score"),
                        "/_discovery/relevance/score",
                        "deterministic_relevance",
                    ),
                    "matched_query_terms": (
                        discovery.get("matched_query_terms"),
                        "/_discovery/matched_query_terms",
                        "deterministic_relevance",
                    ),
                    "matched_topics": (
                        discovery["relevance"].get("matched_topics"),
                        "/_discovery/relevance/matched_topics",
                        "deterministic_relevance",
                    ),
                    "matched_action_terms": (
                        discovery["relevance"].get("matched_action_terms"),
                        "/_discovery/relevance/matched_action_terms",
                        "deterministic_relevance",
                    ),
                    "open_for_comment": (
                        discovery["relevance"].get("open_for_comment"),
                        "/_discovery/relevance/open_for_comment",
                        "deterministic_date_evaluation",
                    ),
                }
                field_metadata = buildFieldMetadata(
                    source_record,
                    analysis,
                    fetched["body_text"],
                    metadataPath=source_record_path,
                    bodyPath=fetched["body_text_path"],
                    models=models,
                    reviewThreshold=review_threshold,
                    deterministicFields=deterministic_fields,
                )
                if analysis_mode != "ai_extraction":
                    for field in NOTICE_AI_FIELDS:
                        metadata_record = field_metadata[field]
                        for item in metadata_record["items"]:
                            item["extraction_method"] = analysis_mode
                            item["models"] = []
                applyComparisonReviews(field_metadata, comparisons)
                review = summarizeReview(
                    field_metadata,
                    additionalReasons=additional_review_reasons,
                )
                analysis.update(
                    {
                        "notice_analysis_schema_version": NOTICE_ANALYSIS_SCHEMA_VERSION,
                        "prompt_version": PROMPT_VERSION,
                        "analysis_mode": analysis_mode,
                        "field_metadata": field_metadata,
                        "review": review,
                        "model_comparisons": comparisons,
                    }
                )
                analysis_path = artifact_dir / "analysis.json"
                analysis_payload = {
                    "run_id": run_id,
                    "document_number": document_number,
                    "metadata": metadata,
                    "discovery": discovery,
                    "analysis": analysis,
                    "source": {
                        "metadata_path": fetched["metadata_path"],
                        "source_record_path": str(source_record_path),
                        "body_html_path": fetched["body_html_path"],
                        "body_text_path": fetched["body_text_path"],
                        "model_input_path": str(model_input_path),
                        "model_context_truncated": text_truncated,
                    },
                    "audit_path": str(notice_audit.path),
                }
                atomic_write_json(analysis_path, analysis_payload)

                artifact_paths = [
                    fetched["metadata_path"],
                    source_record_path,
                    model_input_path,
                    notice_audit.path,
                    analysis_path,
                    *primary_paths,
                    *comparison_paths,
                ]
                if fetched["body_html_path"]:
                    artifact_paths.append(fetched["body_html_path"])
                if fetched["body_text_path"]:
                    artifact_paths.append(fetched["body_text_path"])
                reproducibility_path = artifact_dir / "reproducibility.json"
                reproducibility = {
                    "notice_analysis_schema_version": NOTICE_ANALYSIS_SCHEMA_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "run_id": run_id,
                    "document_number": document_number,
                    "analysis_mode": analysis_mode,
                    "models": models,
                    "comparison_models": comparison_models,
                    "settings": {
                        "review_threshold": review_threshold,
                        "max_text_characters": settings["processing"][
                            "max_text_characters"
                        ],
                    },
                    "artifacts": artifact_inventory(
                        artifact_paths,
                        relative_to=artifact_dir,
                    ),
                    "created_at": utc_now(),
                }
                atomic_write_json(reproducibility_path, reproducibility)
                artifact_paths.append(reproducibility_path)
                snapshot_destination = run_dir / "notices" / document_number
                snapshot_manifest_path = (
                    snapshot_destination / "snapshot_manifest.json"
                )
                notice_audit.record(
                    "notice_processing_finished",
                    review_required=review["required"],
                    analysis_path=str(analysis_path),
                    snapshot_manifest_path=str(snapshot_manifest_path),
                )
                snapshot = snapshot_artifacts(
                    artifact_paths,
                    artifact_dir,
                    snapshot_destination,
                )
                snapshot_analysis_path = snapshot_destination / "analysis.json"
                snapshot_reproducibility_path = (
                    snapshot_destination / "reproducibility.json"
                )
                snapshot_audit_path = (
                    snapshot_destination
                    / notice_audit.path.resolve().relative_to(
                        artifact_dir.resolve()
                    )
                )
                record = {
                    **analysis_payload,
                    "artifact_dir": str(snapshot_destination),
                    "analysis_path": str(snapshot_analysis_path),
                    "audit_path": str(snapshot_audit_path),
                    "reproducibility_path": str(
                        snapshot_reproducibility_path
                    ),
                    "snapshot_manifest_path": str(snapshot_manifest_path),
                    "snapshot": snapshot,
                }
                return {"record": record}
            except Exception as exc:
                logger.exception(
                    "Unable to process notice %s",
                    document_number,
                )
                failure = {
                    "document_number": document_number,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                notice_audit.record(
                    "notice_processing_finished",
                    status="failed",
                    **failure,
                )
                return {"failure": failure}

    if len(ranked) <= 1 or worker_count == 1:
        outcomes = [process_one(candidate) for candidate in ranked]
    else:
        with ThreadPoolExecutor(
            max_workers=min(worker_count, len(ranked)),
            thread_name_prefix="grail-notice",
        ) as executor:
            outcomes = list(executor.map(process_one, ranked))

    records = [outcome["record"] for outcome in outcomes if "record" in outcome]
    failures = [outcome["failure"] for outcome in outcomes if "failure" in outcome]
    for record in records:
        audit.record(
            "notice_finished",
            document_number=record["document_number"],
            review_required=record["analysis"]["review"]["required"],
        )
    for failure in failures:
        audit.record("notice_finished", status="failed", **failure)
    rows = [buildNoticeSheetRow(record) for record in records]

    rows_appended = 0
    rows_skipped = 0
    sheet_worksheet_title = None
    export_requested = bool(
        spreadsheetUrl
        or useConfiguredSheet
        or settings["output"]["export_to_sheet"]
    )
    sheet_export: dict[str, Any] = {
        "requested": export_requested,
        "status": "not_requested",
    }
    if export_requested:
        output_env = settings["output"]["sheet_url_env"]
        shared_env = config["application"]["sheet_url_env"]
        export_url = (
            (spreadsheetUrl or "").strip()
            or os.getenv(output_env, "").strip()
            or os.getenv(shared_env, "").strip()
        )
        try:
            if not export_url:
                raise ValueError(
                    "Google Sheet export is enabled for notices, but neither "
                    f"{output_env} nor shared fallback {shared_env} is set"
                )
            credentials_path = Path(
                config["application"]["service_account_file"]
            )
            if not credentials_path.is_absolute():
                credentials_path = (
                    Path(configPath).resolve().parent / credentials_path
                )
            worksheet_name = (
                settings["output"]["worksheet_name"].strip() or None
            )
            export_details = exportMappingRows(
                export_url,
                rows,
                credentialsPath=str(credentials_path),
                worksheetTitle=worksheet_name,
                createWorksheet=worksheet_name is not None,
                worksheetRows=max(2, len(rows) + 1),
                worksheetCols=max(1, len(NOTICE_SHEET_COLUMNS)),
                canonicalHeaders=NOTICE_SHEET_COLUMNS,
                deduplicateBy=(
                    "FR Document Number"
                    if settings["output"]["deduplicate_by_fr_document"]
                    else None
                ),
                batchSize=settings["output"]["batch_size"],
            )
            rows_appended = export_details["appended"]
            rows_skipped = export_details["skipped"]
            sheet_worksheet_title = export_details["worksheet_title"]
            sheet_export = {
                "requested": True,
                "status": "completed",
                **export_details,
            }
            audit.record(
                "notice_sheet_export_finished",
                row_count=rows_appended,
                skipped_count=rows_skipped,
                worksheet_title=sheet_worksheet_title,
                spreadsheet_url=export_url,
            )
        except Exception as exc:
            sheet_failure = {
                "stage": "notice_sheet_export",
                "document_number": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(sheet_failure)
            sheet_export = {
                "requested": True,
                "status": "failed",
                **sheet_failure,
            }
            audit.record(
                "notice_sheet_export_finished",
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
            logger.exception(
                "Google Sheets export failed; notice artifacts were retained"
            )

    batch_payload = {
        "run_id": run_id,
        "workflow": "notice_discovery",
        "notice_analysis_schema_version": NOTICE_ANALYSIS_SCHEMA_VERSION,
        "candidate_count": len(search_result["candidates"]),
        "selected_count": len(ranked),
        "processed_count": len(records),
        "failure_count": len(failures),
        "sheet_rows_appended": rows_appended,
        "sheet_rows_skipped": rows_skipped,
        "sheet_worksheet_title": sheet_worksheet_title,
        "sheet_export": sheet_export,
        "document_numbers": [
            record["document_number"] for record in records
        ],
        "failures": failures,
        "rows": rows,
        "notice_artifacts": [
            {
                "document_number": record["document_number"],
                "analysis_path": record["analysis_path"],
                "audit_path": record["audit_path"],
                "reproducibility_path": record["reproducibility_path"],
                "snapshot_manifest_path": record["snapshot_manifest_path"],
            }
            for record in records
        ],
    }
    run_batch_path = run_dir / "notice_batch.json"
    latest_batch_path = root / "notice_batch.json"
    atomic_write_json(run_batch_path, batch_payload)
    atomic_write_json(latest_batch_path, batch_payload)
    finished_at = utc_now()
    status = "completed" if not failures else "completed_with_failures"
    manifest.update(
        {
            "status": status,
            "finished_at": finished_at,
            "results": {
                **{
                    key: batch_payload[key]
                    for key in (
                        "candidate_count",
                        "selected_count",
                        "processed_count",
                        "failure_count",
                        "sheet_rows_appended",
                        "sheet_rows_skipped",
                        "sheet_worksheet_title",
                        "document_numbers",
                    )
                },
                "batch_manifest_path": str(run_batch_path),
                "latest_batch_manifest_path": str(latest_batch_path),
                "candidate_manifest_path": str(candidate_manifest),
            },
        }
    )
    atomic_write_json(run_manifest_path, manifest)
    audit.record(
        "notice_pipeline_finished",
        status=status,
        processed_count=len(records),
        failure_count=len(failures),
        run_manifest_path=str(run_manifest_path),
    )
    logger.info(
        "Notice discovery finished: %d processed, %d failed",
        len(records),
        len(failures),
        extra={
            "event": "notice_pipeline_finished",
            "run_id": run_id,
            "success_count": len(records),
            "failure_count": len(failures),
        },
    )
    return {
        "run_id": run_id,
        "status": status,
        "records": records,
        "rows": rows,
        "failures": failures,
        "candidate_count": len(search_result["candidates"]),
        "selected_count": len(ranked),
        "sheet_rows_appended": rows_appended,
        "sheet_rows_skipped": rows_skipped,
        "sheet_worksheet_title": sheet_worksheet_title,
        "sheet_export": sheet_export,
        "manifest_path": str(run_batch_path),
        "latest_manifest_path": str(latest_batch_path),
        "run_manifest_path": str(run_manifest_path),
        "audit_path": str(audit.path),
    }
