"""End-to-end attachment and AI analysis for one public comment."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.audit import (
    ANALYSIS_SCHEMA_VERSION,
    PROMPT_VERSION,
    AuditTrail,
    artifact_inventory,
    atomic_write_json,
    create_run_id,
    sha256_json,
    sha256_text,
    snapshot_artifacts,
    utc_now,
    validate_run_id,
)
from src.logging_config import log_context

from .ai_analysis import (
    IDENTITY_FIELDS,
    TRACEABLE_FIELDS,
    analyzeIdentity,
    analyzePolicyChunk,
    buildIdentityContext,
    consolidatePolicy,
    describeAIClient,
    emptyIdentityExtraction,
    emptyPolicyExtraction,
    flattenAnalysis,
    identityFromMetadata,
    mergePolicyResults,
    mergeIdentityResults,
    validateIdentityResult,
    validatePolicyResult,
    validateResult,
)
from .attachments import chunkText, prepareCommentText, saveJSON
from .provenance import (
    apply_comparison_reviews,
    build_field_metadata,
    compare_analyses,
    summarize_review,
)

logger = logging.getLogger(__name__)
ATTACHMENT_PLACEHOLDER = re.compile(
    r"\b(?:please\s+)?see\s+(?:the\s+)?attach(?:ed|ment)|"
    r"\bcomments?\s+(?:are\s+)?attached\b",
    re.IGNORECASE,
)


def _error_artifact(error: Exception) -> dict[str, str]:
    return {
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _has_substantive_policy_text(
    metadata: dict[str, Any],
    prepared: dict[str, Any],
) -> bool:
    document_text = prepared["cleaned_text"]
    if not document_text.strip():
        return False

    attachment_has_text = any(
        attachment.get("text_path")
        and Path(attachment["text_path"]).exists()
        and Path(attachment["text_path"]).stat().st_size > 0
        for attachment in prepared["attachments"]
    )
    comment_body = (
        metadata.get("data", {}).get("attributes", {}).get("comment") or ""
    ).strip()
    if (
        not attachment_has_text
        and len(comment_body) <= 300
        and ATTACHMENT_PLACEHOLDER.search(comment_body)
    ):
        return False
    return True


def _model_slug(model: dict[str, Any]) -> str:
    label = f"{model.get('provider', 'model')}-{model.get('model', 'unknown')}"
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "model"
    return f"{readable[:40]}-{sha256_text(label)[:10]}"


def _validate_comparison_model(model: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise TypeError("Each comparison model must be an object")
    provider = model.get("provider")
    model_name = model.get("model")
    if provider not in {"openrouter", "ollama"}:
        raise ValueError(
            f"Unsupported comparison model provider: {provider}"
        )
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Comparison model name cannot be empty")
    result = {
        "provider": provider,
        "model": model_name.strip(),
    }
    if "timeout_seconds" in model:
        timeout = model["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("Comparison timeout_seconds must be a positive integer")
        result["timeout_seconds"] = timeout
    if "temperature" in model:
        temperature = model["temperature"]
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("Comparison temperature must be between 0 and 2")
        result["temperature"] = temperature
    return result


def _persist_call_audit(
    destination: Path,
    prefix: str,
    attempts: list[dict[str, Any]],
) -> list[str]:
    paths = []
    for attempt_number, attempt in enumerate(attempts, start=1):
        path = destination / f"{prefix}_call_{attempt_number:02d}.json"
        saveJSON(path, attempt)
        paths.append(str(path))
    return paths


def _run_comparison_variant(
    metadata: dict[str, Any],
    document_text: str,
    identity_context: str,
    document_chunks: list[str],
    has_substantive_policy_text: bool,
    ai_dir: Path,
    raw_model: dict[str, Any],
) -> dict[str, Any]:
    """Run an independent extraction with one explicitly selected comparison model."""
    model = _validate_comparison_model(raw_model)
    variant = f"{model['provider']}/{model['model']}"
    comparison_dir = ai_dir / "comparisons" / _model_slug(model)
    call_audit: list[dict[str, Any]] = []
    artifact_paths: list[str] = []
    errors: list[dict[str, str]] = []

    identity_result = identityFromMetadata(metadata)
    identity_calls: list[dict[str, Any]] = []
    identity_attempts: list[dict[str, Any]] = []
    try:
        ai_identity = analyzeIdentity(
            metadata,
            identity_context,
            identity_attempts,
            modelSettings=model,
            callAudit=identity_calls,
            callContext={
                "stage": "identity",
                "variant": variant,
                "comparison": True,
            },
        )
        identity_result = mergeIdentityResults(identity_result, ai_identity)
        validateIdentityResult(identity_result, identity_context, metadata)
    except Exception as exc:
        errors.append(
            {
                "stage": "identity",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        call_audit.extend(identity_calls)
        for attempt_number, attempt in enumerate(identity_attempts, start=1):
            path = comparison_dir / f"identity_attempt_{attempt_number:02d}.json"
            saveJSON(path, attempt)
            artifact_paths.append(str(path))
        artifact_paths.extend(
            _persist_call_audit(comparison_dir, "identity", identity_calls)
        )
        identity_path = comparison_dir / "identity.json"
        saveJSON(identity_path, identity_result)
        artifact_paths.append(str(identity_path))

    policy_result = emptyPolicyExtraction()
    successful_policies = []
    if has_substantive_policy_text:
        for chunk_number, chunk in enumerate(document_chunks, start=1):
            chunk_calls: list[dict[str, Any]] = []
            chunk_attempts: list[dict[str, Any]] = []
            try:
                chunk_policy = analyzePolicyChunk(
                    metadata,
                    chunk,
                    chunk_number,
                    chunk_attempts,
                    modelSettings=model,
                    callAudit=chunk_calls,
                    callContext={
                        "stage": "policy",
                        "substage": "chunk",
                        "chunk_number": chunk_number,
                        "variant": variant,
                        "comparison": True,
                    },
                )
                successful_policies.append(chunk_policy)
                path = (
                    comparison_dir
                    / "chunks"
                    / f"chunk_{chunk_number:04d}.json"
                )
                saveJSON(path, chunk_policy)
                artifact_paths.append(str(path))
            except Exception as exc:
                errors.append(
                    {
                        "stage": f"policy_chunk_{chunk_number}",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            finally:
                call_audit.extend(chunk_calls)
                for attempt_number, attempt in enumerate(
                    chunk_attempts,
                    start=1,
                ):
                    path = (
                        comparison_dir
                        / "chunks"
                        / f"chunk_{chunk_number:04d}_attempt_{attempt_number:02d}.json"
                    )
                    saveJSON(path, attempt)
                    artifact_paths.append(str(path))
                artifact_paths.extend(
                    _persist_call_audit(
                        comparison_dir / "chunks",
                        f"chunk_{chunk_number:04d}",
                        chunk_calls,
                    )
                )

    if successful_policies:
        consolidation_calls: list[dict[str, Any]] = []
        consolidation_attempts: list[dict[str, Any]] = []
        try:
            policy_result = consolidatePolicy(
                metadata,
                successful_policies,
                document_text,
                consolidation_attempts,
                modelSettings=model,
                callAudit=consolidation_calls,
                callContext={
                    "stage": "policy",
                    "substage": "consolidation",
                    "variant": variant,
                    "comparison": True,
                },
            )
        except Exception as exc:
            errors.append(
                {
                    "stage": "policy_consolidation",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            policy_result = mergePolicyResults(successful_policies)
            validatePolicyResult(
                policy_result,
                document_text,
                metadata,
                hasSubstantialText=False,
            )
        finally:
            call_audit.extend(consolidation_calls)
            for attempt_number, attempt in enumerate(
                consolidation_attempts,
                start=1,
            ):
                path = (
                    comparison_dir
                    / f"consolidated_policy_attempt_{attempt_number:02d}.json"
                )
                saveJSON(path, attempt)
                artifact_paths.append(str(path))
            artifact_paths.extend(
                _persist_call_audit(
                    comparison_dir,
                    "consolidated_policy",
                    consolidation_calls,
                )
            )

    policy_path = comparison_dir / "consolidated_policy.json"
    saveJSON(policy_path, policy_result)
    artifact_paths.append(str(policy_path))
    analysis = flattenAnalysis(
        identity_result,
        policy_result,
        [
            f"Comparison extraction {variant} had {len(errors)} failed stage(s)."
        ]
        if errors
        else None,
    )
    usable = any(
        str(attempt.get("status", "")).startswith("validated")
        for attempt in call_audit
    )
    result = {
        "model": model,
        "status": (
            "success"
            if usable and not errors
            else ("partial" if usable else "failed")
        ),
        "analysis": analysis,
        "errors": errors,
        "call_audit": call_audit,
        "artifact_paths": artifact_paths,
    }
    comparison_analysis_path = comparison_dir / "analysis.json"
    saveJSON(
        comparison_analysis_path,
        {
            key: value
            for key, value in result.items()
            if key not in {"call_audit", "artifact_paths"}
        },
    )
    result["artifact_paths"].append(str(comparison_analysis_path))
    result["analysis_path"] = str(comparison_analysis_path)
    return result


def _apply_source_integrity_reviews(
    field_metadata: dict[str, dict[str, Any]],
    prepared: dict[str, Any],
    has_substantive_policy_text: bool,
) -> None:
    integrity_reasons = []
    if any(item.get("download_error") for item in prepared["attachments"]):
        integrity_reasons.append("attachment_download_failed")
    if any(item.get("extraction_error") for item in prepared["attachments"]):
        integrity_reasons.append("attachment_extraction_failed")
    if any(item.get("ocr_errors") for item in prepared["attachments"]):
        integrity_reasons.append("attachment_ocr_incomplete")
    if any(item.get("unsupported_format") for item in prepared["attachments"]):
        integrity_reasons.append("attachment_format_unsupported")
    if not has_substantive_policy_text:
        integrity_reasons.append("substantive_source_unavailable")

    if not integrity_reasons:
        return
    identity_field_set = {
        *IDENTITY_FIELDS,
        "contact_information",
        "website_url",
    }
    for field in TRACEABLE_FIELDS:
        if field in identity_field_set:
            continue
        field_record = field_metadata[field]
        for reason in integrity_reasons:
            if reason not in field_record["review_reasons"]:
                field_record["review_reasons"].append(reason)
        field_record["review_required"] = True


def analyzeComment(
    metadata: dict[str, Any],
    downloadRoot: str = "downloads",
    comparisonModels: list[dict[str, Any]] | None = None,
    reviewThreshold: float = 0.75,
    runId: str | None = None,
):
    """Run one comment in an isolated logging and audit context."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be an object")
    data = metadata.get("data")
    if not isinstance(data, dict):
        raise ValueError("metadata.data must be an object")
    comment_id = data.get("id")
    resolved_run_id = validate_run_id(runId or create_run_id())
    if (
        isinstance(reviewThreshold, bool)
        or not isinstance(reviewThreshold, (int, float))
        or not 0 <= reviewThreshold <= 1
    ):
        raise ValueError("reviewThreshold must be between 0 and 1")
    if isinstance(comment_id, str) and Path(comment_id).name == comment_id:
        existing_snapshot = (
            Path(downloadRoot).resolve()
            / "runs"
            / resolved_run_id
            / "comments"
            / comment_id
            / "snapshot_manifest.json"
        )
        if existing_snapshot.exists():
            raise FileExistsError(
                "Run/comment snapshot already exists; use a unique run ID: "
                f"{resolved_run_id}/{comment_id}"
            )
    comparison_models = []
    seen_models = set()
    for model in comparisonModels or []:
        key = sha256_json(model)
        if key in seen_models:
            continue
        seen_models.add(key)
        comparison_models.append(model)
    with log_context(
        run_id=resolved_run_id,
        comment_id=comment_id,
        stage="comment_analysis",
    ):
        return _analyze_comment(
            metadata,
            downloadRoot,
            comparison_models,
            reviewThreshold,
            resolved_run_id,
        )


def _analyze_comment(
    metadata: dict[str, Any],
    downloadRoot: str,
    comparisonModels: list[dict[str, Any]],
    reviewThreshold: float,
    runId: str,
):
    """Run staged, evidence-backed analysis for one Regulations.gov comment.

    Raw source text, identity output, each policy chunk, and consolidated policy
    output are persisted separately.  A failure in one AI stage is recorded and
    does not discard successful output from the other stages.
    """
    work_root = (
        Path(downloadRoot).resolve()
        / ".work"
        / sha256_text(runId)[:12]
    )
    prepared = prepareCommentText(metadata, str(work_root))
    artifact_dir = Path(prepared["artifact_dir"])
    ai_dir = artifact_dir / "ai"
    document_text = prepared["cleaned_text"]
    operational_notes: list[str] = []
    comment_id = metadata.get("data", {}).get("id")
    audit = AuditTrail(
        artifact_dir / "audit" / f"{runId}.jsonl",
        runId,
        {"comment_id": comment_id},
    )
    audit.record(
        "comment_analysis_started",
        source_text_sha256=sha256_json(
            {
                "metadata": metadata,
                "cleaned_text": document_text,
            }
        ),
        comparison_models=comparisonModels,
        review_threshold=reviewThreshold,
    )
    logger.info(
        "Comment analysis started",
        extra={"event": "comment_analysis_started"},
    )
    primary_call_audit: list[dict[str, Any]] = []

    metadata_identity = identityFromMetadata(metadata)
    identity_result = emptyIdentityExtraction()
    identity_path = ai_dir / "identity.json"
    identity_attempt_paths: list[str] = []
    identity_call_paths: list[str] = []
    identity_attempts: list[dict[str, Any]] = []
    identity_call_audit: list[dict[str, Any]] = []
    identity_analysis_failed = False
    identity_context, identity_context_truncated = buildIdentityContext(document_text)
    if identity_context_truncated:
        operational_notes.append(
            "Identity extraction used bounded excerpts from the beginning and end "
            "of the document."
        )

    try:
        ai_identity_result = analyzeIdentity(
            metadata,
            identity_context,
            identity_attempts,
            callAudit=identity_call_audit,
            callContext={"stage": "identity", "variant": "primary"},
        )
        identity_result = mergeIdentityResults(
            metadata_identity,
            ai_identity_result,
        )
        validateIdentityResult(identity_result, identity_context, metadata)
        saveJSON(identity_path, identity_result)
        recovered_attempts = [
            attempt
            for attempt in identity_call_audit
            if attempt.get("status") == "validated_with_recovery"
        ]
        if recovered_attempts:
            rejected_count = sum(
                int(
                    attempt.get("recovery", {}).get(
                        "rejection_count",
                        0,
                    )
                )
                for attempt in recovered_attempts
            )
            replacement_count = sum(
                int(
                    attempt.get("recovery", {}).get(
                        "replacement_count",
                        0,
                    )
                )
                for attempt in recovered_attempts
            )
            operational_notes.append(
                "Unsupported AI identity findings were safely quarantined "
                f"({rejected_count} rejected, {replacement_count} replaced "
                "with direct API metadata)."
            )
    except Exception as exc:
        identity_analysis_failed = True
        logger.exception(
            "Identity analysis failed for comment %s",
            metadata.get("data", {}).get("id"),
        )
        saveJSON(ai_dir / "identity.error.json", _error_artifact(exc))
        identity_result = metadata_identity
        saveJSON(identity_path, identity_result)
        operational_notes.append(
            "AI identity extraction could not be completed; direct API identity "
            "evidence was retained."
        )
    finally:
        for attempt_number, attempt in enumerate(identity_attempts, start=1):
            attempt_path = ai_dir / f"identity_attempt_{attempt_number:02d}.json"
            saveJSON(attempt_path, attempt)
            identity_attempt_paths.append(str(attempt_path))
        for attempt_number, call_record in enumerate(identity_call_audit, start=1):
            call_path = ai_dir / f"identity_call_{attempt_number:02d}.json"
            saveJSON(call_path, call_record)
            identity_call_paths.append(str(call_path))
        primary_call_audit.extend(identity_call_audit)
        audit.record(
            "identity_analysis_finished",
            status=(
                "ok"
                if any(
                    attempt.get("status") == "validated"
                    for attempt in identity_call_audit
                )
                else "degraded"
            ),
            call_count=len(identity_call_audit),
            output_path=str(identity_path),
        )

    chunk_results: list[dict[str, Any]] = []
    chunk_artifacts: list[str] = []
    policy_call_paths: list[str] = []
    policy_result = emptyPolicyExtraction()
    policy_path = ai_dir / "consolidated_policy.json"
    policy_artifact_created = False
    failed_policy_chunks = 0
    policy_consolidation_failed = False
    consolidation_attempt_paths: list[str] = []
    has_substantive_policy_text = _has_substantive_policy_text(metadata, prepared)
    document_chunks = (
        chunkText(document_text) if has_substantive_policy_text else []
    )

    if has_substantive_policy_text:
        for chunk_number, chunk in enumerate(document_chunks, start=1):
            chunk_path = ai_dir / "chunks" / f"chunk_{chunk_number:04d}.json"
            chunk_attempts: list[dict[str, Any]] = []
            chunk_call_audit: list[dict[str, Any]] = []
            try:
                chunk_policy = analyzePolicyChunk(
                    metadata,
                    chunk,
                    chunk_number,
                    chunk_attempts,
                    callAudit=chunk_call_audit,
                    callContext={
                        "stage": "policy",
                        "substage": "chunk",
                        "chunk_number": chunk_number,
                        "variant": "primary",
                    },
                )
                saveJSON(chunk_path, chunk_policy)
                chunk_artifacts.append(str(chunk_path))
                chunk_results.append(
                    {
                        "chunk_number": chunk_number,
                        "policy": chunk_policy,
                    }
                )
            except Exception as exc:
                failed_policy_chunks += 1
                logger.exception(
                    "Policy analysis failed for comment %s chunk %d",
                    metadata.get("data", {}).get("id"),
                    chunk_number,
                )
                error_path = ai_dir / "chunks" / f"chunk_{chunk_number:04d}.error.json"
                saveJSON(error_path, _error_artifact(exc))
                chunk_artifacts.append(str(error_path))
                operational_notes.append(
                    f"AI policy extraction failed for chunk {chunk_number}."
                )
            finally:
                for attempt_number, attempt in enumerate(chunk_attempts, start=1):
                    attempt_path = (
                        ai_dir
                        / "chunks"
                        / f"chunk_{chunk_number:04d}_attempt_{attempt_number:02d}.json"
                    )
                    saveJSON(attempt_path, attempt)
                    chunk_artifacts.append(str(attempt_path))
                for attempt_number, call_record in enumerate(
                    chunk_call_audit,
                    start=1,
                ):
                    call_path = (
                        ai_dir
                        / "chunks"
                        / f"chunk_{chunk_number:04d}_call_{attempt_number:02d}.json"
                    )
                    saveJSON(call_path, call_record)
                    policy_call_paths.append(str(call_path))
                    chunk_artifacts.append(str(call_path))
                primary_call_audit.extend(chunk_call_audit)

        successful_policies = [result["policy"] for result in chunk_results]
        if successful_policies:
            consolidation_attempts: list[dict[str, Any]] = []
            consolidation_call_audit: list[dict[str, Any]] = []
            try:
                policy_result = consolidatePolicy(
                    metadata,
                    successful_policies,
                    document_text,
                    consolidation_attempts,
                    callAudit=consolidation_call_audit,
                    callContext={
                        "stage": "policy",
                        "substage": "consolidation",
                        "variant": "primary",
                    },
                )
            except Exception as exc:
                policy_consolidation_failed = True
                logger.exception(
                    "Policy consolidation failed for comment %s",
                    metadata.get("data", {}).get("id"),
                )
                saveJSON(ai_dir / "consolidated_policy.error.json", _error_artifact(exc))
                policy_result = mergePolicyResults(successful_policies)
                validatePolicyResult(
                    policy_result,
                    document_text,
                    metadata,
                    hasSubstantialText=False,
                )
                operational_notes.append(
                    "AI policy consolidation failed; validated chunk findings were "
                    "merged deterministically."
                )
            finally:
                for attempt_number, attempt in enumerate(
                    consolidation_attempts,
                    start=1,
                ):
                    attempt_path = (
                        ai_dir
                        / f"consolidated_policy_attempt_{attempt_number:02d}.json"
                    )
                    saveJSON(attempt_path, attempt)
                    consolidation_attempt_paths.append(str(attempt_path))
                for attempt_number, call_record in enumerate(
                    consolidation_call_audit,
                    start=1,
                ):
                    call_path = (
                        ai_dir
                        / f"consolidated_policy_call_{attempt_number:02d}.json"
                    )
                    saveJSON(call_path, call_record)
                    consolidation_attempt_paths.append(str(call_path))
                    policy_call_paths.append(str(call_path))
                primary_call_audit.extend(consolidation_call_audit)
            saveJSON(policy_path, policy_result)
            policy_artifact_created = True
        else:
            operational_notes.append("AI policy analysis could not be completed.")
    elif document_text.strip():
        operational_notes.append(
            "Only an attachment placeholder was available; no substantive text "
            "was sent for policy analysis."
        )
    else:
        operational_notes.append(
            "No extractable comment or attachment text was available for policy analysis."
        )

    policy_recovery_attempts = [
        attempt
        for attempt in primary_call_audit
        if attempt.get("context", {}).get("stage") == "policy"
        and attempt.get("status") == "validated_with_recovery"
    ]
    policy_recovery_reports = [
        attempt.get("recovery", {})
        for attempt in policy_recovery_attempts
        if isinstance(attempt.get("recovery"), dict)
    ]
    policy_repair_count = sum(
        int(report.get("repair_count", 0))
        for report in policy_recovery_reports
    )
    policy_rejection_count = sum(
        int(report.get("rejection_count", 0))
        for report in policy_recovery_reports
    )
    policy_lossless_repair_count = sum(
        int(
            report.get(
                "lossless_repair_count",
                sum(
                    1
                    for item in report.get("repairs", [])
                    if item.get("disposition") == "lossless"
                ),
            )
        )
        for report in policy_recovery_reports
    )
    policy_review_repair_count = sum(
        int(
            report.get(
                "review_repair_count",
                sum(
                    1
                    for item in report.get("repairs", [])
                    if item.get("disposition") != "lossless"
                ),
            )
        )
        for report in policy_recovery_reports
    )
    policy_degraded_repair_count = sum(
        int(
            report.get(
                "degraded_repair_count",
                sum(
                    1
                    for item in report.get("repairs", [])
                    if item.get("disposition") == "degraded"
                ),
            )
        )
        for report in policy_recovery_reports
    )
    policy_dropped_item_count = sum(
        int(
            report.get(
                "dropped_item_count",
                sum(
                    max(1, int(item.get("rejected_count", 1)))
                    for item in report.get("rejected", [])
                    if item.get("final_disposition", "dropped") == "dropped"
                ),
            )
        )
        for report in policy_recovery_reports
    )
    policy_substituted_field_count = sum(
        int(
            report.get(
                "substituted_field_count",
                sum(
                    1
                    for item in report.get("rejected", [])
                    if item.get("final_disposition") == "substituted"
                ),
            )
        )
        for report in policy_recovery_reports
    )
    policy_review_recovery_attempts = [
        attempt
        for attempt in policy_recovery_attempts
        if bool(
            attempt.get("recovery", {}).get(
                "review_required",
                attempt.get("recovery", {}).get("repair_count", 0)
                or attempt.get("recovery", {}).get("rejection_count", 0),
            )
        )
    ]
    policy_degraded_recovery_attempts = [
        attempt
        for attempt in policy_recovery_attempts
        if bool(
            attempt.get("recovery", {}).get(
                "degraded",
                attempt.get("recovery", {}).get("rejection_count", 0),
            )
        )
    ]
    if policy_degraded_recovery_attempts:
        operational_notes.append(
            "Policy extraction recovery discarded or substituted unsupported "
            f"content ({policy_dropped_item_count} dropped, "
            f"{policy_substituted_field_count} substituted)."
        )
    elif policy_review_recovery_attempts:
        operational_notes.append(
            "Policy extraction reconstructed source provenance for "
            f"{policy_review_repair_count} field(s); review is recommended."
        )

    audit.record(
        "policy_analysis_finished",
        status=(
            (
                "degraded"
                if policy_degraded_recovery_attempts
                else (
                    "review_required"
                    if policy_review_recovery_attempts
                    else "ok"
                )
            )
            if policy_artifact_created
            else ("skipped" if not has_substantive_policy_text else "degraded")
        ),
        chunk_count=len(document_chunks),
        successful_chunk_count=len(chunk_results),
        recovered_call_count=len(policy_recovery_attempts),
        recovery_repair_count=policy_repair_count,
        recovery_rejection_count=policy_rejection_count,
        recovery_lossless_repair_count=policy_lossless_repair_count,
        recovery_review_repair_count=policy_review_repair_count,
        recovery_degraded_repair_count=policy_degraded_repair_count,
        recovery_dropped_item_count=policy_dropped_item_count,
        recovery_substituted_field_count=policy_substituted_field_count,
        recovery_review_call_count=len(policy_review_recovery_attempts),
        recovery_degraded_call_count=len(policy_degraded_recovery_attempts),
        output_path=str(policy_path) if policy_artifact_created else None,
    )

    failed_downloads = [
        attachment
        for attachment in prepared["attachments"]
        if attachment.get("download_error")
    ]
    if failed_downloads:
        operational_notes.append(
            f"{len(failed_downloads)} attachment(s) could not be downloaded for analysis."
        )

    failed_extractions = [
        attachment
        for attachment in prepared["attachments"]
        if attachment.get("extraction_error")
    ]
    if failed_extractions:
        operational_notes.append(
            f"{len(failed_extractions)} attachment(s) could not be text-extracted."
        )

    unsupported_attachments = [
        attachment
        for attachment in prepared["attachments"]
        if attachment.get("unsupported_format")
    ]
    if unsupported_attachments:
        operational_notes.append(
            f"{len(unsupported_attachments)} attachment(s) used an unsupported file format."
        )

    ocr_error_pages = sum(
        len(attachment.get("ocr_errors", []))
        for attachment in prepared["attachments"]
    )
    if ocr_error_pages:
        operational_notes.append(
            f"OCR failed on {ocr_error_pages} attachment page(s); native text was retained."
        )

    processing_issues: list[dict[str, Any]] = []
    if identity_analysis_failed:
        processing_issues.append(
            {
                "stage": "identity",
                "code": "identity_analysis_failed",
                "message": (
                    "AI identity extraction failed; only direct API identity "
                    "metadata was retained."
                ),
            }
        )
    if has_substantive_policy_text and not policy_artifact_created:
        processing_issues.append(
            {
                "stage": "policy",
                "code": "policy_analysis_failed",
                "message": (
                    "AI policy extraction failed for every usable text chunk."
                ),
                "failed_chunk_count": failed_policy_chunks,
                "chunk_count": len(document_chunks),
            }
        )
    elif failed_policy_chunks:
        processing_issues.append(
            {
                "stage": "policy",
                "code": "policy_chunk_analysis_failed",
                "message": (
                    f"AI policy extraction failed for {failed_policy_chunks} "
                    f"of {len(document_chunks)} text chunk(s); successful chunks "
                    "were retained."
                ),
                "failed_chunk_count": failed_policy_chunks,
                "chunk_count": len(document_chunks),
            }
        )
    if policy_consolidation_failed:
        processing_issues.append(
            {
                "stage": "policy",
                "code": "policy_consolidation_failed",
                "message": (
                    "AI policy consolidation failed; validated chunk findings "
                    "were merged deterministically."
                ),
            }
        )
    if policy_degraded_recovery_attempts:
        processing_issues.append(
            {
                "stage": "policy",
                "code": "policy_field_recovery",
                "message": (
                    "Policy recovery discarded unsupported findings or substituted "
                    "an extractive summary; affected field paths are retained in "
                    "AI call audits."
                ),
                "recovered_call_count": len(
                    policy_degraded_recovery_attempts
                ),
                "repair_count": policy_repair_count,
                "rejection_count": policy_rejection_count,
                "dropped_item_count": policy_dropped_item_count,
                "substituted_field_count": policy_substituted_field_count,
            }
        )
    for stage, code, items, message in (
        (
            "attachment_download",
            "attachment_download_failed",
            failed_downloads,
            "attachment(s) could not be downloaded",
        ),
        (
            "attachment_extraction",
            "attachment_extraction_failed",
            failed_extractions,
            "attachment(s) could not be text-extracted",
        ),
        (
            "attachment_extraction",
            "attachment_format_unsupported",
            unsupported_attachments,
            "attachment(s) used an unsupported file format",
        ),
    ):
        if items:
            processing_issues.append(
                {
                    "stage": stage,
                    "code": code,
                    "message": f"{len(items)} {message}.",
                    "attachment_count": len(items),
                }
            )
    if ocr_error_pages:
        processing_issues.append(
            {
                "stage": "attachment_ocr",
                "code": "attachment_ocr_incomplete",
                "message": (
                    f"OCR failed on {ocr_error_pages} attachment page(s); "
                    "available native text was retained."
                ),
                "page_count": ocr_error_pages,
            }
        )

    analysis = flattenAnalysis(
        identity_result,
        policy_result,
        operational_notes,
    )
    comparison_reports: list[dict[str, Any]] = []
    comparison_artifacts: list[str] = []
    for raw_model in comparisonModels:
        model_label = (
            f"{raw_model.get('provider', '')}/{raw_model.get('model', '')}"
            if isinstance(raw_model, dict)
            else str(raw_model)
        )
        audit.record(
            "model_comparison_started",
            model=model_label,
        )
        try:
            variant = _run_comparison_variant(
                metadata,
                document_text,
                identity_context,
                document_chunks,
                has_substantive_policy_text,
                ai_dir,
                raw_model,
            )
            comparison_artifacts.extend(variant["artifact_paths"])
            if variant["status"] == "failed":
                report = {
                    "model": variant["model"],
                    "status": "failed",
                    "agreement_score": None,
                    "review_required": True,
                    "review_fields": [],
                    "errors": variant["errors"],
                    "analysis_path": variant["analysis_path"],
                }
            else:
                report = compare_analyses(
                    analysis,
                    variant["analysis"],
                    variant["model"],
                )
                report.update(
                    {
                        "status": variant["status"],
                        "errors": variant["errors"],
                        "analysis_path": variant["analysis_path"],
                    }
                )
            report_path = (
                Path(variant["analysis_path"]).parent / "comparison_report.json"
            )
            saveJSON(report_path, report)
            comparison_artifacts.append(str(report_path))
            report["report_path"] = str(report_path)
            comparison_reports.append(report)
            audit.record(
                "model_comparison_finished",
                status=variant["status"],
                model=model_label,
                agreement_score=report.get("agreement_score"),
                review_fields=report.get("review_fields", []),
                report_path=str(report_path),
            )
        except Exception as exc:
            logger.exception("Comparison model failed: %s", model_label)
            failed_report = {
                "model": raw_model,
                "status": "failed",
                "agreement_score": None,
                "review_required": True,
                "review_fields": [],
                "errors": [
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                ],
            }
            comparison_reports.append(failed_report)
            operational_notes.append(
                f"Model comparison {model_label} could not be completed."
            )
            audit.record(
                "model_comparison_finished",
                status="failed",
                model=model_label,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    for report in comparison_reports:
        if report.get("status") not in {"failed", "partial"}:
            continue
        model = report.get("model", {})
        model_label = (
            f"{model.get('provider', '')}/{model.get('model', '')}".strip("/")
            if isinstance(model, dict)
            else str(model)
        )
        processing_issues.append(
            {
                "stage": "model_comparison",
                "code": f"model_comparison_{report.get('status')}",
                "message": (
                    f"Model comparison {model_label or 'unknown model'} "
                    f"{report.get('status')}."
                ),
                "model": model,
            }
        )

    field_metadata = build_field_metadata(
        identity_result,
        policy_result,
        metadata,
        prepared,
        primary_call_audit,
        reviewThreshold,
    )
    apply_comparison_reviews(
        field_metadata,
        [
            report
            for report in comparison_reports
            if "fields" in report
        ],
    )
    for report in comparison_reports:
        if report.get("status") != "failed":
            continue
        model = report.get("model", {})
        label = (
            f"{model.get('provider', '')}/{model.get('model', '')}".strip("/")
            if isinstance(model, dict)
            else str(model)
        )
        reason = f"comparison_model_failed:{label or 'unknown_model'}"
        for field_record in field_metadata.values():
            if not field_record["items"]:
                continue
            if reason not in field_record["review_reasons"]:
                field_record["review_reasons"].append(reason)
            field_record["review_required"] = True
    _apply_source_integrity_reviews(
        field_metadata,
        prepared,
        has_substantive_policy_text,
    )

    def mark_review(
        fields: list[str],
        reason: str,
        *,
        empty_confidence: float | None = None,
        confidence_cap: float | None = None,
    ) -> None:
        for field in fields:
            field_record = field_metadata.get(field)
            if field_record is None:
                continue
            if reason not in field_record["review_reasons"]:
                field_record["review_reasons"].append(reason)
            field_record["review_required"] = True
            if not field_record["items"] and empty_confidence is not None:
                field_record["confidence"] = empty_confidence
            elif (
                confidence_cap is not None
                and field_record["confidence"] is not None
            ):
                field_record["confidence"] = min(
                    field_record["confidence"],
                    confidence_cap,
                )

    identity_traceable_fields = [
        *IDENTITY_FIELDS,
        "contact_information",
        "website_url",
    ]
    policy_traceable_fields = [
        field
        for field in TRACEABLE_FIELDS
        if field not in identity_traceable_fields
    ]
    if identity_analysis_failed:
        mark_review(
            [
                field
                for field in identity_traceable_fields
                if not field_metadata[field]["items"]
            ],
            "identity_analysis_failed",
            empty_confidence=0.0,
        )

    quarantined_fields = {
        str(item.get("field", "")).split("[", 1)[0]
        for attempt in identity_call_audit
        if attempt.get("status") == "validated_with_recovery"
        for item in attempt.get("recovery", {}).get("rejected", [])
        if item.get("field")
    }
    if quarantined_fields:
        mark_review(
            sorted(quarantined_fields),
            "unsupported_ai_identity_quarantined",
            empty_confidence=0.0,
        )

    recovered_policy_fields = {
        str(item.get("field", "")).split("[", 1)[0]
        for attempt in policy_recovery_attempts
        for item in attempt.get("recovery", {}).get("repairs", [])
        if item.get("field")
        and bool(
            item.get(
                "review_required",
                item.get("disposition") != "lossless",
            )
        )
        and item.get("disposition") != "degraded"
    }
    if recovered_policy_fields:
        mark_review(
            sorted(recovered_policy_fields),
            "ai_policy_exact_source_recovery",
            empty_confidence=0.0,
            confidence_cap=0.75,
        )

    quarantined_policy_fields = {
        str(item.get("field", "")).split("[", 1)[0]
        for attempt in policy_recovery_attempts
        for item in attempt.get("recovery", {}).get("rejected", [])
        if item.get("field")
        and item.get("final_disposition", "dropped") == "dropped"
    }
    if quarantined_policy_fields:
        mark_review(
            sorted(quarantined_policy_fields),
            "unsupported_ai_policy_finding_quarantined",
            empty_confidence=0.0,
        )

    substituted_policy_fields = {
        str(item.get("field", "")).split("[", 1)[0]
        for attempt in policy_recovery_attempts
        for item in attempt.get("recovery", {}).get("repairs", [])
        if item.get("field")
        and item.get("disposition") == "degraded"
    }
    if substituted_policy_fields:
        mark_review(
            sorted(substituted_policy_fields),
            "ai_policy_extractive_summary_fallback",
            empty_confidence=0.0,
            confidence_cap=0.5,
        )

    if has_substantive_policy_text and not policy_artifact_created:
        mark_review(
            policy_traceable_fields,
            "policy_analysis_failed",
            empty_confidence=0.0,
            confidence_cap=0.0,
        )
    elif failed_policy_chunks:
        mark_review(
            policy_traceable_fields,
            "policy_chunk_analysis_failed",
            empty_confidence=0.0,
            confidence_cap=0.5,
        )
    if policy_consolidation_failed:
        mark_review(
            policy_traceable_fields,
            "policy_consolidation_failed",
            empty_confidence=0.0,
            confidence_cap=0.5,
        )

    analysis["field_metadata"] = field_metadata
    analysis["model_comparisons"] = comparison_reports
    analysis["review"] = summarize_review(field_metadata)
    # Comparison failures are operational notes; disagreements are already attached
    # to the exact affected fields above.
    analysis["analysis_notes"] = list(
        dict.fromkeys([*analysis["analysis_notes"], *operational_notes])
    )
    validateResult(
        analysis,
        hasSubstantialText=(
            has_substantive_policy_text and policy_artifact_created
        ),
    )

    reproducibility_path = artifact_dir / "reproducibility.json"
    input_fingerprint = {
        "metadata_sha256": sha256_json(metadata),
        "cleaned_text_sha256": sha256_text(document_text),
        "prompt_version": PROMPT_VERSION,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "primary_model": describeAIClient(),
        "comparison_models": comparisonModels,
        "review_threshold": reviewThreshold,
    }
    source_and_ai_paths: list[str | Path] = [
        artifact_dir / "metadata.json",
        artifact_dir / "attachments.json",
        artifact_dir / "raw_text.txt",
        artifact_dir / "cleaned_text.txt",
        *identity_attempt_paths,
        *identity_call_paths,
        *chunk_artifacts,
        *policy_call_paths,
        *consolidation_attempt_paths,
        *comparison_artifacts,
    ]
    if identity_path.exists():
        source_and_ai_paths.append(identity_path)
    if policy_artifact_created:
        source_and_ai_paths.append(policy_path)
    for attachment in prepared["attachments"]:
        source_and_ai_paths.extend(
            path
            for path in [
                attachment.get("path"),
                attachment.get("raw_text_path"),
                attachment.get("text_path"),
            ]
            if path
        )
    reproducibility = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": runId,
        "comment_id": comment_id,
        "generated_at": utc_now(),
        "input_fingerprint": input_fingerprint,
        "cache_key": sha256_json(input_fingerprint),
        "models": {
            "primary_and_fallback": describeAIClient(),
            "comparisons": comparisonModels,
        },
        "artifacts": artifact_inventory(
            source_and_ai_paths,
            relative_to=artifact_dir,
        ),
    }

    analysis_path = artifact_dir / "analysis.json"
    artifact_manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_id": runId,
        "comment_id": comment_id,
        "artifact_dir": str(artifact_dir),
        "metadata_path": str(artifact_dir / "metadata.json"),
        "attachments_path": str(artifact_dir / "attachments.json"),
        "raw_text_path": str(artifact_dir / "raw_text.txt"),
        "cleaned_text_path": str(artifact_dir / "cleaned_text.txt"),
        "audit_path": str(audit.path),
        "reproducibility_path": str(reproducibility_path),
        "ai_artifacts": {
            "identity": str(identity_path) if identity_path.exists() else None,
            "identity_attempts": identity_attempt_paths,
            "identity_calls": identity_call_paths,
            "policy_chunks": chunk_artifacts,
            "policy_calls": policy_call_paths,
            "consolidation_attempts": consolidation_attempt_paths,
            "consolidated_policy": (
                str(policy_path) if policy_artifact_created else None
            ),
            "model_comparisons": comparison_artifacts,
        },
        "processing_issues": processing_issues,
        "analysis": analysis,
    }
    saveJSON(analysis_path, artifact_manifest)
    reproducibility["artifacts"] = artifact_inventory(
        [*source_and_ai_paths, analysis_path],
        relative_to=artifact_dir,
    )
    atomic_write_json(reproducibility_path, reproducibility)
    audit.record(
        "comment_analysis_finished",
        status="review_required" if analysis["review"]["required"] else "ok",
        analysis_path=str(analysis_path),
        reproducibility_path=str(reproducibility_path),
        overall_confidence=analysis["review"]["overall_confidence"],
        review_fields=analysis["review"]["fields"],
        processing_issue_count=len(processing_issues),
        processing_issue_codes=[
            issue["code"]
            for issue in processing_issues
        ],
    )
    snapshot_dir = (
        Path(downloadRoot).resolve()
        / "runs"
        / runId
        / "comments"
        / str(comment_id)
    )
    snapshot = snapshot_artifacts(
        [
            *source_and_ai_paths,
            analysis_path,
            reproducibility_path,
            audit.path,
        ],
        source_root=artifact_dir,
        destination_root=snapshot_dir,
    )
    snapshot_analysis_path = snapshot_dir / "analysis.json"
    snapshot_audit_path = snapshot_dir / audit.path.resolve().relative_to(
        artifact_dir.resolve()
    )
    snapshot_reproducibility_path = snapshot_dir / "reproducibility.json"
    logger.info(
        (
            "Comment analysis finished with processing issues"
            if processing_issues
            else "Comment analysis finished"
        ),
        extra={
            "event": "comment_analysis_finished",
            "review_required": analysis["review"]["required"],
            "overall_confidence": analysis["review"]["overall_confidence"],
            "processing_issue_count": len(processing_issues),
            "processing_issue_codes": [
                issue["code"]
                for issue in processing_issues
            ],
        },
    )

    return {
        **artifact_manifest,
        "analysis_path": str(snapshot_analysis_path),
        "audit_path": str(snapshot_audit_path),
        "reproducibility_path": str(snapshot_reproducibility_path),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_manifest_path": str(snapshot_dir / "snapshot_manifest.json"),
        "snapshot": snapshot,
        "attachments": prepared["attachments"],
        "identity_result": identity_result,
        "chunk_results": chunk_results,
        "policy_result": policy_result,
    }
