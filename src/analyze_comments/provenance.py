"""Field-level provenance, confidence, review flags, and model comparison."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from src.audit import sha256_file, sha256_text

from .ai_analysis import (
    IDENTITY_FIELDS,
    TRACEABLE_FIELDS,
    identityFromMetadata,
)
from .attachments import cleanText, normalizeCommentHTML

EXTRACTED_FIELDS = TRACEABLE_FIELDS
SCALAR_FIELDS = {
    *IDENTITY_FIELDS,
    "website_url",
    "brief_summary",
    "position_or_stance",
}
PAGE_MARKER = re.compile(r"^--- PAGE (\d+) ---$", re.MULTILINE)
COMMENT_BODY_MARKER = re.compile(
    r"^--- COMMENT BODY ---\s*$",
    re.MULTILINE,
)
ATTACHMENT_MARKER = re.compile(
    r"^--- ATTACHMENT .+ ---\s*$",
    re.MULTILINE,
)
COMMENT_JSON_POINTER = "/data/attributes/comment"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _json_pointer_segment(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _find_metadata_paths(
    value: Any,
    evidence: str,
    path: str = "",
) -> list[str]:
    matches: list[str] = []
    if isinstance(value, str):
        if _normalized_text(evidence) in _normalized_text(value):
            matches.append(path or "/")
    elif isinstance(value, dict):
        for key, nested in value.items():
            matches.extend(
                _find_metadata_paths(
                    nested,
                    evidence,
                    f"{path}/{_json_pointer_segment(str(key))}",
                )
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            matches.extend(
                _find_metadata_paths(nested, evidence, f"{path}/{index}")
            )
    return matches


def _metadata_scalar_entries(
    value: Any,
    path: str = "",
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    if isinstance(value, str) and value.strip():
        entries.append((path or "/", value.strip()))
    elif isinstance(value, dict):
        for key, nested in value.items():
            entries.extend(
                _metadata_scalar_entries(
                    nested,
                    f"{path}/{_json_pointer_segment(str(key))}",
                )
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            entries.extend(
                _metadata_scalar_entries(nested, f"{path}/{index}")
            )
    return entries


def _page_for_evidence(
    text: str,
    evidence: str,
    claimed_page: int | None,
) -> int | None:
    normalized_evidence = _normalized_text(evidence)
    markers = list(PAGE_MARKER.finditer(text))
    if not markers:
        return None if normalized_evidence in _normalized_text(text) else None

    for index, marker in enumerate(markers):
        page = int(marker.group(1))
        if claimed_page is not None and page != claimed_page:
            continue
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        if normalized_evidence in _normalized_text(text[marker.end() : end]):
            return page
    return None


def _source_record(
    *,
    source_type: str,
    source_id: str,
    artifact_path: str,
    evidence: str,
    page: int | None = None,
    json_pointer: str | None = None,
    attachment_id: str | None = None,
    content_sha256: str | None = None,
    source_transformation: str | None = None,
    transformed_content_sha256: str | None = None,
    transformed_artifact_path: str | None = None,
    transformed_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_type": source_type,
        "source_id": source_id,
        "artifact_path": artifact_path,
        "page": page,
        "evidence": evidence,
        "content_sha256": content_sha256
        if content_sha256 is not None
        else (
            sha256_file(artifact_path)
            if artifact_path and Path(artifact_path).is_file()
            else None
        ),
    }
    if json_pointer is not None:
        record["json_pointer"] = json_pointer
    if attachment_id is not None:
        record["attachment_id"] = attachment_id
    if source_transformation is not None:
        record["source_transformation"] = source_transformation
    if transformed_content_sha256 is not None:
        record["transformed_content_sha256"] = transformed_content_sha256
    if transformed_artifact_path is not None:
        record["transformed_artifact_path"] = transformed_artifact_path
    if transformed_artifact_sha256 is not None:
        record["transformed_artifact_sha256"] = transformed_artifact_sha256
    return record


def _metadata_comment(metadata: dict[str, Any]) -> str | None:
    comment = (
        metadata.get("data", {})
        .get("attributes", {})
        .get("comment")
    )
    return comment if isinstance(comment, str) and comment.strip() else None


def _normalized_metadata_comment(metadata: dict[str, Any]) -> str | None:
    """Apply the same HTML-to-text cleanup used for the AI document text."""
    comment = _metadata_comment(metadata)
    if comment is None:
        return None
    return cleanText(normalizeCommentHTML(comment))


def _cleaned_comment_artifact(
    prepared: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Return only the saved comment section, never attachment text."""
    artifact_dir = Path(prepared["artifact_dir"])
    cleaned_path = artifact_dir / "cleaned_text.txt"
    if not cleaned_path.is_file():
        return None
    cleaned_text = cleaned_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    marker = COMMENT_BODY_MARKER.search(cleaned_text)
    if marker is None:
        return None
    attachment = ATTACHMENT_MARKER.search(cleaned_text, marker.end())
    end = attachment.start() if attachment is not None else len(cleaned_text)
    comment_text = cleaned_text[marker.end() : end].strip()
    if not comment_text:
        return None
    return (
        comment_text,
        str(cleaned_path),
        sha256_file(cleaned_path),
    )


def _build_source_cache(prepared: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path(prepared["artifact_dir"])
    metadata_path = artifact_dir / "metadata.json"
    attachments = []
    for attachment in prepared.get("attachments", []):
        text_path = attachment.get("text_path")
        if not text_path or not Path(text_path).is_file():
            continue
        path = Path(text_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        attachments.append(
            {
                "record": attachment,
                "path": str(path),
                "text": text,
                "normalized_text": _normalized_text(text),
                "content_sha256": sha256_file(path),
            }
        )
    return {
        "metadata_path": str(metadata_path),
        "metadata_sha256": (
            sha256_file(metadata_path) if metadata_path.is_file() else None
        ),
        "attachments": attachments,
    }


def locate_evidence_sources(
    evidence_item: dict[str, Any],
    metadata: dict[str, Any],
    prepared: dict[str, Any],
    source_cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve an evidence excerpt to exact API, comment, or attachment artifacts."""
    evidence = evidence_item.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        return []

    evidence = evidence.strip()
    claimed_page = evidence_item.get("page")
    cache = source_cache or _build_source_cache(prepared)
    metadata_path = cache["metadata_path"]
    sources: list[dict[str, Any]] = []
    metadata_paths = _find_metadata_paths(metadata, evidence)

    for pointer in metadata_paths:
        source_type = (
            "comment_body"
            if pointer.endswith("/data/attributes/comment")
            else "api_metadata"
        )
        sources.append(
            _source_record(
                source_type=source_type,
                source_id=(
                    "regulations.gov:data.attributes.comment"
                    if source_type == "comment_body"
                    else f"regulations.gov:{pointer}"
                ),
                artifact_path=metadata_path,
                json_pointer=pointer,
                evidence=evidence,
                page=None,
                content_sha256=cache["metadata_sha256"],
            )
        )

    # The AI sees an HTML-decoded and cleaned comment body. Resolve evidence
    # against that exact transformation, but keep the authoritative citation
    # anchored to the raw metadata artifact and JSON pointer.
    comment_pointer_resolved = any(
        source.get("json_pointer") == COMMENT_JSON_POINTER
        for source in sources
    )
    normalized_comment = _normalized_metadata_comment(metadata)
    if (
        not comment_pointer_resolved
        and normalized_comment
        and _normalized_text(evidence) in _normalized_text(normalized_comment)
    ):
        sources.append(
            _source_record(
                source_type="comment_body",
                source_id="regulations.gov:data.attributes.comment",
                artifact_path=metadata_path,
                json_pointer=COMMENT_JSON_POINTER,
                evidence=evidence,
                page=None,
                content_sha256=cache["metadata_sha256"],
                source_transformation="html_to_text_and_clean_text",
                transformed_content_sha256=sha256_text(normalized_comment),
            )
        )
        comment_pointer_resolved = True

    # Defensive fallback for previously prepared artifacts. This remains
    # comment-section-only so attachment evidence cannot be mislabeled.
    if not comment_pointer_resolved and _metadata_comment(metadata):
        cleaned_comment = _cleaned_comment_artifact(prepared)
        if (
            cleaned_comment is not None
            and _normalized_text(evidence)
            in _normalized_text(cleaned_comment[0])
        ):
            sources.append(
                _source_record(
                    source_type="comment_body",
                    source_id="regulations.gov:data.attributes.comment",
                    artifact_path=metadata_path,
                    json_pointer=COMMENT_JSON_POINTER,
                    evidence=evidence,
                    page=None,
                    content_sha256=cache["metadata_sha256"],
                    source_transformation="saved_cleaned_comment_text",
                    transformed_content_sha256=sha256_text(
                        cleaned_comment[0]
                    ),
                    transformed_artifact_path=cleaned_comment[1],
                    transformed_artifact_sha256=cleaned_comment[2],
                )
            )

    # A claimed page unambiguously points to extracted document content, so place
    # attachment matches ahead of coincidental API metadata matches.
    attachment_sources: list[dict[str, Any]] = []
    for cached_attachment in cache["attachments"]:
        attachment = cached_attachment["record"]
        text_path = cached_attachment["path"]
        text = cached_attachment["text"]
        page = _page_for_evidence(text, evidence, claimed_page)
        if claimed_page is not None and page is None:
            continue
        if _normalized_text(evidence) not in cached_attachment["normalized_text"]:
            continue
        attachment_sources.append(
            _source_record(
                source_type="attachment",
                source_id=f"attachment:{attachment.get('attachment_id', Path(text_path).name)}",
                artifact_path=str(text_path),
                attachment_id=attachment.get("attachment_id"),
                evidence=evidence,
                page=page,
                content_sha256=cached_attachment["content_sha256"],
            )
        )

    if claimed_page is not None:
        sources = [*attachment_sources, *sources]
    else:
        sources.extend(attachment_sources)

    # Preserve all genuinely distinct matches while eliminating duplicate pointers.
    deduplicated: list[dict[str, Any]] = []
    seen = set()
    for source in sources:
        key = (
            source["source_type"],
            source["artifact_path"],
            source.get("json_pointer"),
            source.get("page"),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(source)
    return deduplicated


def _locate_direct_value_components(
    value: str,
    metadata: dict[str, Any],
    source_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map a deterministically combined API value back to its scalar JSON fields."""
    normalized_value = _normalized_text(value)
    records = []
    for pointer, component in _metadata_scalar_entries(metadata):
        normalized_component = _normalized_text(component)
        if not normalized_component or normalized_component not in normalized_value:
            continue
        records.append(
            _source_record(
                source_type=(
                    "comment_body"
                    if pointer.endswith("/data/attributes/comment")
                    else "api_metadata"
                ),
                source_id=f"regulations.gov:{pointer}",
                artifact_path=source_cache["metadata_path"],
                json_pointer=pointer,
                evidence=component,
                page=None,
                content_sha256=source_cache["metadata_sha256"],
            )
        )
    return records


def _successful_models(
    call_audit: Iterable[dict[str, Any]],
    stage: str,
) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    seen = set()
    for attempt in call_audit:
        context = attempt.get("context", {})
        if context.get("stage") != stage:
            continue
        for call in attempt.get("provider_calls", []):
            if call.get("status") != "success":
                continue
            key = (call.get("provider"), call.get("model"))
            if key in seen:
                continue
            seen.add(key)
            models.append(
                {
                    "provider": call.get("provider", ""),
                    "model": call.get("model", ""),
                }
            )
    return models


def _field_evidence_items(
    identity_result: dict[str, Any],
    policy_result: dict[str, Any],
    field: str,
) -> list[dict[str, Any]]:
    if field in IDENTITY_FIELDS:
        item = identity_result[field]
        return [item] if item.get("value") is not None else []
    if field == "contact_information":
        return [
            item
            for item in identity_result[field]
            if item.get("value") is not None
        ]
    if field == "website_url":
        item = identity_result[field]
        return [item] if item.get("value") is not None else []
    if field in {"brief_summary", "position_or_stance"}:
        item = policy_result[field]
        return [item] if item.get("value") is not None else []
    return [
        item
        for item in policy_result.get(field, [])
        if item.get("value") is not None
    ]


def build_field_metadata(
    identity_result: dict[str, Any],
    policy_result: dict[str, Any],
    metadata: dict[str, Any],
    prepared: dict[str, Any],
    call_audit: list[dict[str, Any]] | None = None,
    review_threshold: float = 0.75,
) -> dict[str, dict[str, Any]]:
    """Create one traceable confidence/review record for every extracted field."""
    if isinstance(review_threshold, bool) or not isinstance(
        review_threshold,
        (int, float),
    ):
        raise TypeError("review_threshold must be numeric")
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be between 0 and 1")

    audit = call_audit or []
    direct_identity = identityFromMetadata(metadata)
    source_cache = _build_source_cache(prepared)

    def is_direct_api_item(field: str, item: dict[str, Any]) -> bool:
        if field in IDENTITY_FIELDS or field == "website_url":
            candidate = direct_identity[field]
            return (
                candidate.get("value") is not None
                and candidate.get("value") == item.get("value")
                and candidate.get("evidence") == item.get("evidence")
            )
        if field == "contact_information":
            return any(
                candidate.get("value") == item.get("value")
                and candidate.get("evidence") == item.get("evidence")
                for candidate in direct_identity["contact_information"]
            )
        return False

    field_metadata: dict[str, dict[str, Any]] = {}
    for field in EXTRACTED_FIELDS:
        stage = "identity" if field in {
            *IDENTITY_FIELDS,
            "contact_information",
            "website_url",
        } else "policy"
        models = _successful_models(audit, stage)
        item_records = []
        for item in _field_evidence_items(
            identity_result,
            policy_result,
            field,
        ):
            confidence = float(item.get("confidence", 0))
            sources = locate_evidence_sources(
                item,
                metadata,
                prepared,
                source_cache,
            )
            direct_api_item = is_direct_api_item(field, item)
            if direct_api_item and isinstance(item.get("value"), str):
                sources.extend(
                    _locate_direct_value_components(
                        item["value"],
                        metadata,
                        source_cache,
                    )
                )
                deduplicated_sources = []
                seen_sources = set()
                for source in sources:
                    key = (
                        source["source_type"],
                        source["artifact_path"],
                        source.get("json_pointer"),
                        source.get("page"),
                    )
                    if key not in seen_sources:
                        seen_sources.add(key)
                        deduplicated_sources.append(source)
                sources = deduplicated_sources
            review_reasons = []
            if confidence < review_threshold:
                review_reasons.append(
                    f"confidence_below_{review_threshold:g}"
                )
            if item.get("inferred"):
                review_reasons.append("inferred_value")
            if item.get("evidence") and not sources:
                review_reasons.append("source_not_resolved")
            item_records.append(
                {
                    "value": item.get("value"),
                    "confidence": confidence,
                    "inferred": bool(item.get("inferred")),
                    "review_required": bool(review_reasons),
                    "review_reasons": review_reasons,
                    "sources": sources,
                    "extraction_method": (
                        "deterministic_api"
                        if direct_api_item
                        else "ai_extraction"
                    ),
                    "models": [] if direct_api_item else models,
                }
            )

        confidences = [item["confidence"] for item in item_records]
        review_reasons = sorted(
            {
                reason
                for item in item_records
                for reason in item["review_reasons"]
            }
        )
        field_metadata[field] = {
            "confidence": min(confidences) if confidences else None,
            "review_required": bool(review_reasons),
            "review_reasons": review_reasons,
            "items": item_records,
        }
    return field_metadata


def _analysis_values(analysis: dict[str, Any], field: str) -> list[str]:
    value = analysis.get(field)
    if field in IDENTITY_FIELDS:
        value = value.get("value") if isinstance(value, dict) else value
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1
    }


def _value_similarity(left: str, right: str) -> float:
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if left_normalized == right_normalized:
        return 1.0
    sequence_score = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    token_score = (
        len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if left_tokens or right_tokens
        else 0.0
    )
    return round(max(sequence_score, token_score), 4)


def _compare_value_lists(
    primary_values: list[str],
    comparison_values: list[str],
    match_threshold: float = 0.72,
) -> dict[str, Any]:
    if not primary_values and not comparison_values:
        return {
            "status": "both_empty",
            "agreement_score": 1.0,
            "matched": [],
            "primary_only": [],
            "comparison_only": [],
        }
    if not primary_values or not comparison_values:
        return {
            "status": "mismatch",
            "agreement_score": 0.0,
            "matched": [],
            "primary_only": primary_values,
            "comparison_only": comparison_values,
        }

    candidates = []
    for primary_index, primary in enumerate(primary_values):
        for comparison_index, comparison in enumerate(comparison_values):
            candidates.append(
                (
                    _value_similarity(primary, comparison),
                    primary_index,
                    comparison_index,
                )
            )
    candidates.sort(reverse=True)
    matched_primary = set()
    matched_comparison = set()
    matches = []
    for score, primary_index, comparison_index in candidates:
        if score < match_threshold:
            continue
        if primary_index in matched_primary or comparison_index in matched_comparison:
            continue
        matched_primary.add(primary_index)
        matched_comparison.add(comparison_index)
        matches.append(
            {
                "primary": primary_values[primary_index],
                "comparison": comparison_values[comparison_index],
                "similarity": score,
            }
        )

    denominator = max(len(primary_values), len(comparison_values))
    agreement_score = round(
        sum(match["similarity"] for match in matches) / denominator,
        4,
    )
    primary_only = [
        value
        for index, value in enumerate(primary_values)
        if index not in matched_primary
    ]
    comparison_only = [
        value
        for index, value in enumerate(comparison_values)
        if index not in matched_comparison
    ]
    if not primary_only and not comparison_only and agreement_score >= 0.85:
        status = "match"
    elif matches:
        status = "partial_match"
    else:
        status = "mismatch"
    return {
        "status": status,
        "agreement_score": agreement_score,
        "matched": matches,
        "primary_only": primary_only,
        "comparison_only": comparison_only,
    }


def compare_analyses(
    primary_analysis: dict[str, Any],
    comparison_analysis: dict[str, Any],
    comparison_model: dict[str, Any],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in EXTRACTED_FIELDS:
        fields[field] = _compare_value_lists(
            _analysis_values(primary_analysis, field),
            _analysis_values(comparison_analysis, field),
        )

    compared = [
        result
        for result in fields.values()
        if result["status"] != "both_empty"
    ]
    agreement_score = (
        round(
            sum(result["agreement_score"] for result in compared) / len(compared),
            4,
        )
        if compared
        else 1.0
    )
    review_fields = [
        field
        for field, result in fields.items()
        if result["status"] in {"partial_match", "mismatch"}
    ]
    return {
        "model": comparison_model,
        "agreement_score": agreement_score,
        "review_required": bool(review_fields),
        "review_fields": review_fields,
        "fields": fields,
    }


def apply_comparison_reviews(
    field_metadata: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    for comparison in comparisons:
        model = comparison.get("model", {})
        model_label = f"{model.get('provider', '')}/{model.get('model', '')}".strip("/")
        for field in comparison.get("review_fields", []):
            if field not in field_metadata:
                continue
            reason = f"model_disagreement:{model_label or 'comparison_model'}"
            field_record = field_metadata[field]
            if reason not in field_record["review_reasons"]:
                field_record["review_reasons"].append(reason)
            field_record["review_required"] = True


def summarize_review(
    field_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    populated_confidences = [
        record["confidence"]
        for record in field_metadata.values()
        if record["confidence"] is not None
    ]
    fields = [
        field
        for field, record in field_metadata.items()
        if record["review_required"]
    ]
    reasons = sorted(
        {
            reason
            for record in field_metadata.values()
            for reason in record["review_reasons"]
        }
    )
    return {
        "required": bool(fields),
        "fields": fields,
        "reasons": reasons,
        "overall_confidence": (
            round(sum(populated_confidences) / len(populated_confidences), 4)
            if populated_confidences
            else None
        ),
    }


def compact_confidence_map(
    field_metadata: dict[str, dict[str, Any]],
) -> str:
    return json.dumps(
        {
            field: record["confidence"]
            for field, record in field_metadata.items()
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
