from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.services.ai_client import generate_json, get_client, research_comment_fields

logger = logging.getLogger(__name__)


COMMENT_SCHEMA: dict[str, Any] = {
    "Comment ID": None,
    "Filename": None,
    "Date Submitted": None,
    "Submitter Name": None,
    "Organization Name": None,
    "Organization Type": None,
    "501c Status": None,
    "NTEE": None,
    "Organization Role": None,
    "Contact Info": {
        "professional_email": None,
        "professional_phone": None,
        "organization_website": None,
        "professional_profile": None,
    },
    "Keywords": [],
    "Brief Summary": None,
    "Issues Addressed": [],
    "Full Text": None,
    "Evidence": {},
    "Sources": [],
    "Confidence": {},
    "Research Notes": None,
}


TEXT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "Submitter Name": None,
    "Organization Name": None,
    "Organization Role": None,
    "Keywords": [],
    "Brief Summary": None,
    "Issues Addressed": [],
}


RESEARCH_SCHEMA: dict[str, Any] = {
    "Submitter Name": None,
    "Organization Name": None,
    "Organization Type": None,
    "501c Status": None,
    "NTEE": None,
    "Organization Role": None,
    "Contact Info": {
        "professional_email": None,
        "professional_phone": None,
        "organization_website": None,
        "professional_profile": None,
    },
    "Evidence": {},
    "Sources": [],
    "Confidence": {},
    "Research Notes": None,
}


_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "Comment ID": ("Comment ID", "comment_id", "commentId", "id"),
    "Filename": ("Filename", "filename", "file_name"),
    "Date Submitted": (
        "Date Submitted",
        "date_submitted",
        "submittedDate",
        "postedDate",
    ),
    "Submitter Name": (
        "Submitter Name",
        "Submitted By",
        "submittedBy",
        "submitterName",
    ),
    "Organization Name": (
        "Organization Name",
        "organization_name",
        "organizationName",
        "organization",
    ),
}


def processCommentText(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    provider: str | None = None,
    allow_fallback: bool = True,
    research: bool = True,
) -> dict[str, Any]:
    """Return one completed comment record.

    Args:
        text: Extracted text from the comment or its attachment.
        metadata: Existing Regulations.gov metadata. Existing non-empty values
            take priority over AI-generated values.
        provider: ``"openrouter"``, ``"ollama"``, or ``None`` to use the
            configured primary provider and fallback.
        allow_fallback: Allow the configured fallback provider.
        research: Disable this to perform text analysis without web research.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Comment text is empty")

    normalized_metadata = _normalize_metadata(metadata or {})
    comment_id = normalized_metadata.get("Comment ID", "unknown")
    logger.info("Analyzing comment text for %s", comment_id)

    analysis_prompt = f"""Analyze this public regulatory comment using only the
supplied text. Do not use outside knowledge and do not guess missing identity
or organization information.

Rules:
- Submitter Name: use only a clearly identified author or signatory.
- Organization Name: use only an organization explicitly represented by the
  comment, not an organization merely discussed in the text.
- Organization Role: briefly describe the organization's role in relation to
  the rulemaking or issue.
- Keywords: return 5 to 12 specific topics as short strings.
- Brief Summary: write a factual 2 to 4 sentence summary.
- Issues Addressed: return a list of the main policy or technical issues.
- Use null or an empty list when the text does not support a field.

Known metadata:
{json.dumps(normalized_metadata, ensure_ascii=False, default=str, indent=2)}

Comment text:
{cleaned_text}
"""

    enrichment_notes: list[str] = []
    blocked_reason = get_client().provider_block_reason(provider)

    if blocked_reason is not None:
        analyzed = {}
        enrichment_notes.append("AI text analysis was unavailable.")
        logger.info("Skipping AI text analysis for %s: %s", comment_id, blocked_reason)
    else:
        try:
            analyzed = generate_json(
                analysis_prompt,
                schema=TEXT_ANALYSIS_SCHEMA,
                provider=provider,
                allow_fallback=allow_fallback,
                temperature=0,
            )
            if not isinstance(analyzed, Mapping):
                raise TypeError("AI text analysis must return an object")
            analyzed = dict(analyzed)
        except Exception as error:
            logger.warning(
                "AI text analysis failed; preserving source data for this comment: %s",
                error,
            )
            analyzed = {}
            enrichment_notes.append("AI text analysis was unavailable.")

    blocked_reason = get_client().provider_block_reason(provider)
    research_block_reason = get_client().research_block_reason(provider)

    researched: dict[str, Any] = {}
    if research and blocked_reason is not None:
        enrichment_notes.append("AI research enrichment was unavailable.")
        logger.info("Skipping AI research for %s: %s", comment_id, blocked_reason)
    elif research and research_block_reason is not None:
        enrichment_notes.append("AI research enrichment was unavailable.")
        logger.info(
            "Skipping AI research for %s: %s",
            comment_id,
            research_block_reason,
        )
    elif research:
        research_record = {
            **normalized_metadata,
            **analyzed,
            # Enough text for identity and affiliation clues without making the
            # web-research prompt absurdly large.
            "Comment Text Excerpt": cleaned_text[:12_000],
        }
        try:
            researched = research_comment_fields(
                research_record,
                schema=RESEARCH_SCHEMA,
                provider=provider,
                allow_fallback=allow_fallback,
            )
            if not isinstance(researched, Mapping):
                raise TypeError("AI research must return an object")
            researched = dict(researched)
        except Exception as error:
            logger.warning(
                "AI research failed; preserving source data for this comment: %s",
                error,
            )
            researched = {}
            enrichment_notes.append("AI research enrichment was unavailable.")

    result = deepcopy(COMMENT_SCHEMA)

    # Research is the weakest source, direct text is stronger, and supplied
    # API metadata is strongest. Apply them in that order.
    _fill_missing(result, researched)
    _fill_missing(result, analyzed)
    _overwrite_present(result, normalized_metadata)

    if enrichment_notes and _is_missing(result["Research Notes"]):
        result["Research Notes"] = " ".join(enrichment_notes)

    # Never let a model rewrite or truncate the source text.
    result["Full Text"] = cleaned_text
    logger.info("Completed comment text analysis for %s", comment_id)
    return result


def process_comment_file(
    text_file: str | Path,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Read a UTF-8 text file and process it into the comment schema."""
    path = Path(text_file)
    text = path.read_text(encoding="utf-8")
    combined_metadata = dict(metadata or {})
    combined_metadata.setdefault("Filename", path.name)
    return processCommentText(text, combined_metadata, **kwargs)


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for output_key, aliases in _METADATA_ALIASES.items():
        for alias in aliases:
            value = metadata.get(alias)
            if not _is_missing(value):
                normalized[output_key] = value
                break

    first_name = metadata.get("firstName") or metadata.get("first_name")
    last_name = metadata.get("lastName") or metadata.get("last_name")
    if "Submitter Name" not in normalized and (first_name or last_name):
        normalized["Submitter Name"] = " ".join(
            str(part).strip() for part in (first_name, last_name) if part
        )

    # Keep useful filing context for web identity matching.
    for key in (
        "docketId",
        "documentId",
        "commentOnDocumentId",
        "agencyId",
        "title",
        "category",
    ):
        value = metadata.get(key)
        if not _is_missing(value):
            normalized[key] = value

    return normalized


def _fill_missing(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key not in target:
            continue
        if isinstance(target[key], dict) and isinstance(value, Mapping):
            _fill_missing(target[key], value)
        elif _is_missing(target[key]) and not _is_missing(value):
            target[key] = deepcopy(value)


def _overwrite_present(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if key in target and not _is_missing(value):
            target[key] = deepcopy(value)


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
