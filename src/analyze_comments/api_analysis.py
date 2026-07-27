import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

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
from src.logging_config import log_context

from .ai_analysis import configureAIClient, emptyAnalysis
from .attachments import getAttachmentURLs, saveJSON
from .comment_pipeline import analyzeComment
from .provenance import compact_confidence_map

load_dotenv()

logger = logging.getLogger(__name__)

SHEET_COLUMNS = {
    "Comment ID": "",
    "FR Document Number": "",
    "Docket ID": "",
    "Comment On Document ID": "",
    "Organization Name": "",
    "Date Submitted": "",
    "Date Posted": "",
    "Submitter Name": "",
    "Submitter Role": "",
    "Organization Type": "",
    "Contact Information": "",
    "Website URL": "",
    "Link to Comment Text": "",
    "Brief Summary of Comment": "",
    "Relevant Issues Addressed": "",
    "Position or Stance": "",
    "Recommendations": "",
    "Policy Requests": "",
    "AI Topics": "",
    "Affected Stakeholders": "",
    "Evidence or Sources Cited": "",
    "Attachment URLs": "",
    "Duplicate Comments": "",
    "Withdrawn": "",
    "Fields Inferred": "",
    "Analysis Notes": "",
    "Overall Confidence": "",
    "Review Required": "",
    "Review Fields": "",
    "Field Confidence Scores": "",
    "Model Comparison": "",
    "Field Provenance": "",
    "Audit Trail": "",
    "Reproducibility Manifest": "",
    "Snapshot Manifest": "",
    "Run ID": "",
}

BASE_URL = "https://api.regulations.gov/v4"
SHEET_CELL_CHARACTER_LIMIT = 45_000


def loadAPIKey():
    """Load the Regulations.gov API key from the environment."""
    api_key = os.getenv("REGULATION_API_KEY", "").strip()
    if not api_key:
        raise ValueError("REGULATION_API_KEY not set")
    return api_key


def joinList(values, maxCharacters=SHEET_CELL_CHARACTER_LIMIT):
    joined_values = []
    seen = set()
    for value in values:
        normalized = str(value).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            joined_values.append(normalized)
    joined = " | ".join(joined_values)
    if len(joined) <= maxCharacters:
        return joined
    suffix = " ... [truncated; see analysis artifact]"
    return joined[: maxCharacters - len(suffix)].rstrip() + suffix


def identityValue(analysis, field):
    identity = analysis.get(field, {})
    if not isinstance(identity, dict):
        return ""
    return identity.get("value") or ""


def getDirectSubmitterName(attributes):
    return " ".join(
        part
        for part in [attributes.get("firstName") or "", attributes.get("lastName") or ""]
        if part
    )


def getDirectAddress(attributes):
    locality = ", ".join(
        value
        for value in [
            attributes.get("city") or "",
            attributes.get("stateProvinceRegion") or "",
            attributes.get("zip") or "",
        ]
        if value
    )
    return ", ".join(
        value
        for value in [
            attributes.get("address1") or "",
            attributes.get("address2") or "",
            locality,
            attributes.get("country") or "",
        ]
        if value
    )


def formatAPIResponse(metadata):
    """Build an API-only sheet row without attachment text analysis."""
    row = buildSheetRow(metadata, emptyAnalysis())
    row["Brief Summary of Comment"] = (
        metadata.get("data", {}).get("attributes", {}).get("comment") or ""
    )
    return row


def _model_comparison_summary(comparisons):
    values = []
    for comparison in comparisons or []:
        model = comparison.get("model", {})
        label = f"{model.get('provider', '')}/{model.get('model', '')}".strip("/")
        status = comparison.get("status", "unknown")
        score = comparison.get("agreement_score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        values.append(f"{label or 'comparison model'}: {status} ({score_text})")
    return joinList(values)


def buildSheetRow(metadata, analysis, analysisRecord=None):
    """Combine direct API data and validated AI analysis into one sheet row."""
    row = SHEET_COLUMNS.copy()
    comment = metadata.get("data", {})
    attributes = comment.get("attributes", {})
    collection_source = metadata.get("collection_source", {})
    direct_contact = [
        attributes.get("email"),
        attributes.get("phone"),
        attributes.get("fax"),
        getDirectAddress(attributes),
    ]
    contact_information = direct_contact + analysis.get("contact_information", [])
    organization_name = attributes.get("organization") or identityValue(
        analysis, "organization_name"
    )
    submitter_name = getDirectSubmitterName(attributes) or identityValue(
        analysis, "submitter_name"
    )
    organization_type = attributes.get("category") or identityValue(
        analysis, "organization_type"
    )
    brief_summary = analysis.get("brief_summary") or ""
    inferred_fields = list(analysis.get("fields_inferred", []))
    direct_identity_fields = {
        "Organization Name": bool(attributes.get("organization")),
        "Submitter Name": bool(getDirectSubmitterName(attributes)),
        "Submitter Role": bool(attributes.get("submitterRep")),
        "Organization Type": bool(attributes.get("category")),
    }
    inferred_fields = [
        field
        for field in inferred_fields
        if not direct_identity_fields.get(field, False)
    ]

    for analysis_field, sheet_field in {
        "organization_name": "Organization Name",
        "submitter_name": "Submitter Name",
        "submitter_role": "Submitter Role",
        "organization_type": "Organization Type",
    }.items():
        identity = analysis.get(analysis_field, {})
        if (
            not direct_identity_fields[sheet_field]
            and identity.get("inferred")
            and sheet_field not in inferred_fields
        ):
            inferred_fields.append(sheet_field)

    analysis_record = analysisRecord or {}
    artifact_dir = analysis_record.get("artifact_dir")
    analysis_path = analysis_record.get("analysis_path") or (
        str(Path(artifact_dir) / "analysis.json") if artifact_dir else ""
    )
    review = analysis.get("review") or {}
    field_metadata = analysis.get("field_metadata") or {}
    row.update(
        {
            "Comment ID": comment.get("id", ""),
            "FR Document Number": attributes.get("frDocNum")
            or collection_source.get("fr_doc_id")
            or "",
            "Docket ID": attributes.get("docketId") or "",
            "Comment On Document ID": attributes.get("commentOnDocumentId") or "",
            "Organization Name": organization_name,
            "Date Submitted": attributes.get("receiveDate") or "",
            "Date Posted": attributes.get("postedDate") or "",
            "Submitter Name": submitter_name,
            "Submitter Role": attributes.get("submitterRep")
            or identityValue(analysis, "submitter_role"),
            "Organization Type": organization_type,
            "Contact Information": joinList(contact_information),
            "Website URL": attributes.get("website")
            or analysis.get("website_url")
            or "",
            "Link to Comment Text": comment.get("links", {}).get("self", ""),
            "Brief Summary of Comment": brief_summary,
            "Relevant Issues Addressed": joinList(
                analysis.get("relevant_issues_addressed", [])
            ),
            "Position or Stance": analysis.get("position_or_stance") or "",
            "Recommendations": joinList(analysis.get("recommendations", [])),
            "Policy Requests": joinList(analysis.get("policy_requests", [])),
            "AI Topics": joinList(analysis.get("ai_topics", [])),
            "Affected Stakeholders": joinList(analysis.get("affected_stakeholders", [])),
            "Evidence or Sources Cited": joinList(
                analysis.get("evidence_or_sources_cited", [])
            ),
            "Attachment URLs": joinList(getAttachmentURLs(metadata)),
            "Duplicate Comments": attributes.get("duplicateComments", ""),
            "Withdrawn": attributes.get("withdrawn", ""),
            "Fields Inferred": joinList(inferred_fields),
            "Analysis Notes": joinList(analysis.get("analysis_notes", [])),
            "Overall Confidence": (
                review.get("overall_confidence")
                if review.get("overall_confidence") is not None
                else ""
            ),
            "Review Required": review.get("required", False),
            "Review Fields": joinList(review.get("fields", [])),
            "Field Confidence Scores": compact_confidence_map(field_metadata),
            "Model Comparison": _model_comparison_summary(
                analysis.get("model_comparisons", [])
            ),
            "Field Provenance": (
                f"{analysis_path}#analysis.field_metadata"
                if analysis_path
                else ""
            ),
            "Audit Trail": analysis_record.get("audit_path", ""),
            "Reproducibility Manifest": analysis_record.get(
                "reproducibility_path",
                "",
            ),
            "Snapshot Manifest": analysis_record.get(
                "snapshot_manifest_path",
                "",
            ),
            "Run ID": analysis_record.get("run_id", ""),
        }
    )
    return row


def fetchCommentMetadata(commentID):
    """Fetch a full comment record with attachment metadata."""
    if not isinstance(commentID, str) or not commentID.strip():
        raise ValueError("Comment ID cannot be empty")
    commentID = commentID.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", commentID):
        raise ValueError("Comment ID contains unsupported characters")

    response = requests.get(
        f"{BASE_URL}/comments/{commentID}",
        headers={
            "X-Api-Key": loadAPIKey(),
            "Accept": "application/json",
        },
        params={"include": "attachments"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    returned_id = payload.get("data", {}).get("id")
    if returned_id != commentID:
        raise ValueError(
            f"Regulations.gov returned comment ID {returned_id!r}; expected {commentID!r}"
        )
    return payload


def analyzeCommentID(
    commentID,
    downloadRoot="downloads",
    collectionSource=None,
    comparisonModels=None,
    reviewThreshold=0.75,
    runId=None,
):
    """Fetch and analyze one comment while retaining its raw artifact manifest."""
    metadata = fetchCommentMetadata(commentID)
    if collectionSource:
        metadata["collection_source"] = collectionSource
    analysis_options = {}
    if comparisonModels:
        analysis_options["comparisonModels"] = comparisonModels
    if reviewThreshold != 0.75:
        analysis_options["reviewThreshold"] = reviewThreshold
    if runId is not None:
        analysis_options["runId"] = runId
    analysis_record = analyzeComment(
        metadata,
        downloadRoot,
        **analysis_options,
    )
    return {
        "comment_id": commentID,
        "metadata": metadata,
        "analysis_record": analysis_record,
        "row": buildSheetRow(
            metadata,
            analysis_record["analysis"],
            analysis_record,
        ),
    }


def fetchAPI(commentID, downloadRoot="downloads"):
    """Compatibility entry point returning one sheet-ready row."""
    return analyzeCommentID(commentID, downloadRoot)["row"]


def analyzeCollectedComments(
    comments,
    downloadRoot="downloads",
    comparisonModels=None,
    reviewThreshold=0.75,
    maxWorkers=4,
    runId=None,
    configPath="config.yaml",
):
    """Analyze collected comment summaries without losing the rest of a batch.

    Each comment gets its own source/AI artifact directory.  Per-comment failures
    are returned in a batch manifest rather than aborting successful records.
    """
    if isinstance(maxWorkers, bool) or not isinstance(maxWorkers, int) or maxWorkers <= 0:
        raise ValueError("maxWorkers must be a positive integer")
    if (
        isinstance(reviewThreshold, bool)
        or not isinstance(reviewThreshold, (int, float))
        or not 0 <= reviewThreshold <= 1
    ):
        raise ValueError("reviewThreshold must be between 0 and 1")

    configureAIClient(configPath)
    comparison_models = []
    seen_comparison_models = set()
    for model in comparisonModels or []:
        key = json.dumps(model, sort_keys=True, default=str)
        if key in seen_comparison_models:
            continue
        seen_comparison_models.add(key)
        comparison_models.append(model)
    run_id = validate_run_id(runId or create_run_id())
    download_root = Path(downloadRoot)
    run_dir = download_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(
        run_dir / "audit.jsonl",
        run_id,
        {"scope": "batch"},
    )
    started_at = utc_now()
    repo_root = Path(__file__).resolve().parents[2]
    run_manifest_path = run_dir / "run_manifest.json"
    existing_manifest: dict[str, Any] = {}
    if run_manifest_path.is_file():
        try:
            loaded_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
            if isinstance(loaded_manifest, dict):
                existing_manifest = loaded_manifest
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Existing run manifest is unreadable: {run_manifest_path}"
            ) from exc
    if existing_manifest and existing_manifest.get("status") != "collecting":
        raise FileExistsError(
            f"Run ID already has a manifest; choose a unique run ID: {run_id}"
        )
    run_manifest: dict[str, Any] = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "status": "running",
        "started_at": existing_manifest.get("started_at", started_at),
        "input": {
            **existing_manifest.get("input", {}),
            "comment_count": len(comments),
            "download_root": str(download_root),
            "comparison_models": comparison_models,
            "review_threshold": reviewThreshold,
            "max_workers": maxWorkers,
        },
        "runtime": existing_manifest.get("runtime")
        or runtime_manifest(repo_root, configPath),
        "audit_path": str(audit.path),
    }
    atomic_write_json(run_manifest_path, run_manifest)
    audit.record(
        "batch_analysis_started",
        comment_count=len(comments),
        max_workers=maxWorkers,
        comparison_models=comparison_models,
    )
    logger.info(
        "Batch analysis started for %d comment(s) with %d worker(s)",
        len(comments),
        maxWorkers,
        extra={
            "event": "batch_analysis_started",
            "run_id": run_id,
            "comment_count": len(comments),
            "max_workers": maxWorkers,
        },
    )

    def analyze_one(comment):
        comment_id = comment if isinstance(comment, str) else comment.get("id")
        if not comment_id:
            return {
                "failure": {
                    "comment_id": None,
                    "error_type": "ValueError",
                    "error": "Collected comment did not include an ID.",
                }
            }

        collection_source = (
            comment.get("source", {}) if isinstance(comment, dict) else {}
        )
        with log_context(run_id=run_id, comment_id=comment_id):
            try:
                record = analyzeCommentID(
                    comment_id,
                    downloadRoot,
                    collection_source,
                    comparison_models,
                    reviewThreshold,
                    run_id,
                )
                processing_issues = (
                    record.get("analysis_record", {}).get(
                        "processing_issues",
                        [],
                    )
                )
                partial_failure = None
                if processing_issues:
                    partial_failure = {
                        "comment_id": comment_id,
                        "stage": "comment_analysis",
                        "error_type": "PartialAnalysis",
                        "error": (
                            "Analysis retained a row but completed with "
                            f"{len(processing_issues)} processing issue(s)."
                        ),
                        "record_retained": True,
                        "issues": processing_issues,
                    }
                audit.record(
                    "comment_finished",
                    status=(
                        "completed_with_failures"
                        if partial_failure
                        else "completed"
                    ),
                    comment_id=comment_id,
                    review_required=record.get("analysis_record", {})
                    .get("analysis", {})
                    .get("review", {})
                    .get("required"),
                    processing_issue_count=len(processing_issues),
                    processing_issue_codes=[
                        issue.get("code")
                        for issue in processing_issues
                        if isinstance(issue, dict)
                    ],
                )
                outcome = {"record": record}
                if partial_failure:
                    outcome["failure"] = partial_failure
                return outcome
            except Exception as exc:
                logger.exception("Unable to analyze comment %s", comment_id)
                failure = {
                    "comment_id": comment_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                audit.record(
                    "comment_finished",
                    status="failed",
                    **failure,
                )
                return {"failure": failure}

    if len(comments) <= 1 or maxWorkers == 1:
        outcomes = [analyze_one(comment) for comment in comments]
    else:
        worker_count = min(maxWorkers, len(comments))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="grail-comment",
        ) as executor:
            outcomes = list(executor.map(analyze_one, comments))

    records = [
        outcome["record"]
        for outcome in outcomes
        if "record" in outcome
    ]
    rows = [record["row"] for record in records]
    failures = [
        outcome["failure"]
        for outcome in outcomes
        if "failure" in outcome
    ]
    degraded_comment_count = sum(
        1
        for failure in failures
        if failure.get("record_retained") is True
    )
    failed_comment_count = len(failures) - degraded_comment_count
    status = "completed" if not failures else "completed_with_failures"

    batch_result = {
        "status": status,
        "rows": rows,
        "records": records,
        "failures": failures,
        "degraded_comment_count": degraded_comment_count,
        "failed_comment_count": failed_comment_count,
    }
    batch_manifest = Path(downloadRoot) / "analysis_batch.json"
    batch_manifest.parent.mkdir(parents=True, exist_ok=True)
    batch_payload = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": run_id,
        "run_manifest_path": str(run_manifest_path),
        "analyzed_comment_ids": [record["comment_id"] for record in records],
        "failures": failures,
        "degraded_comment_count": degraded_comment_count,
        "failed_comment_count": failed_comment_count,
        "row_count": len(rows),
        "rows": rows,
        "comment_artifacts": [
            {
                "comment_id": record["comment_id"],
                "analysis_path": (
                    record.get("analysis_record", {}).get("analysis_path")
                    or (
                        str(
                            Path(record["analysis_record"]["artifact_dir"])
                            / "analysis.json"
                        )
                        if record.get("analysis_record", {}).get("artifact_dir")
                        else None
                    )
                ),
                "audit_path": record.get("analysis_record", {}).get(
                    "audit_path"
                ),
                "reproducibility_path": record.get("analysis_record", {}).get(
                    "reproducibility_path"
                ),
                "snapshot_manifest_path": record.get(
                    "analysis_record",
                    {},
                ).get("snapshot_manifest_path"),
            }
            for record in records
        ],
    }
    saveJSON(batch_manifest, batch_payload)
    run_batch_manifest = run_dir / "analysis_batch.json"
    saveJSON(run_batch_manifest, batch_payload)

    finished_at = utc_now()
    run_manifest.update(
        {
            "status": status,
            "finished_at": finished_at,
            "results": {
                "analyzed_comment_count": len(records),
                "failure_count": len(failures),
                "degraded_comment_count": degraded_comment_count,
                "failed_comment_count": failed_comment_count,
                "row_count": len(rows),
                "batch_manifest_path": str(run_batch_manifest),
                "latest_batch_manifest_path": str(batch_manifest),
                "comment_artifacts": batch_payload["comment_artifacts"],
            },
        }
    )
    atomic_write_json(run_manifest_path, run_manifest)
    audit.record(
        "batch_analysis_finished",
        status=run_manifest["status"],
        analyzed_comment_count=len(records),
        failure_count=len(failures),
        degraded_comment_count=degraded_comment_count,
        failed_comment_count=failed_comment_count,
        run_manifest_path=str(run_manifest_path),
    )
    logger.info(
        "Batch analysis finished: %d analyzed, %d degraded, %d failed",
        len(records),
        degraded_comment_count,
        failed_comment_count,
        extra={
            "event": "batch_analysis_finished",
            "run_id": run_id,
            "success_count": len(records) - degraded_comment_count,
            "degraded_count": degraded_comment_count,
            "failure_count": len(failures),
            "failed_comment_count": failed_comment_count,
        },
    )
    batch_result["manifest_path"] = str(batch_manifest)
    batch_result["run_id"] = run_id
    batch_result["run_manifest_path"] = str(run_manifest_path)
    batch_result["audit_path"] = str(audit.path)
    return batch_result
