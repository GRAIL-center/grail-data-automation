from __future__ import annotations

import ast
import html
import json
import logging
import os
import re
from datetime import datetime
from typing import Any

import requests

from src.collect_comments.analyze_comments import (
    analyzeMetadata,
    fillMetadata,
    initComment,
)
from src.collect_comments.process_comment_text import processCommentText
from src.collect_comments.retrieve_comment_body import getCommentText
from src.services.config import loadRegKey
from src.services.sheets import (
    addRows,
    ensureHeaders,
    getExistingFRIDs,
    getTab,
    setupGoogleSheets,
)

logger = logging.getLogger(__name__)

FR_API_URL = "https://www.federalregister.gov/api/v1"
REGULATIONS_API_URL = "https://api.regulations.gov/v4"

PAGE_SIZE = 250
MAX_PAGES = 20
TIMEOUT = 30

JsonDict = dict[str, Any]

DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
    }
)


def normalizeFRNumber(value: Any) -> str:
    return str(value or "").translate(DASH_TRANSLATION).strip().upper()


def requestJSON(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> JsonDict:
    response = session.get(
        url,
        params=params,
        timeout=TIMEOUT,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object from {url}.")

    return payload


def getDocuments(
    session: requests.Session,
    filters: dict[str, Any],
) -> list[JsonDict]:
    documents: list[JsonDict] = []

    for page_number in range(1, MAX_PAGES + 1):
        payload = requestJSON(
            session,
            f"{REGULATIONS_API_URL}/documents",
            {
                **filters,
                "page[size]": PAGE_SIZE,
                "page[number]": page_number,
            },
        )

        page = payload.get("data", [])

        if not isinstance(page, list):
            raise TypeError("Invalid documents response.")

        documents.extend(page)

        total = int(
            payload.get("meta", {}).get(
                "totalElements",
                len(documents),
            )
        )

        if len(documents) >= total or len(page) < PAGE_SIZE:
            break
    else:
        raise RuntimeError(
            "Document search exceeded the Regulations.gov "
            "5,000-result pagination limit."
        )

    return documents


def matchesFRNumber(
    document: JsonDict,
    fr_number: str,
) -> bool:
    document_fr_number = document.get("attributes", {}).get("frDocNum")

    return normalizeFRNumber(document_fr_number) == fr_number


def dedupeDocuments(
    documents: list[JsonDict],
) -> list[JsonDict]:
    unique_documents: dict[str, JsonDict] = {}

    for document in documents:
        document_id = document.get("id")

        if document_id:
            unique_documents[str(document_id)] = document

    return list(unique_documents.values())


def getRegulationsDocuments(
    session: requests.Session,
    fr_number: str,
) -> list[JsonDict]:
    documents = getDocuments(
        session,
        {"filter[frDocNum]": fr_number},
    )

    exact_matches = [
        document for document in documents if matchesFRNumber(document, fr_number)
    ]

    if exact_matches:
        return dedupeDocuments(exact_matches)

    search_results = getDocuments(
        session,
        {"filter[searchTerm]": fr_number},
    )

    exact_search_matches = [
        document for document in search_results if matchesFRNumber(document, fr_number)
    ]

    if exact_search_matches:
        return dedupeDocuments(exact_search_matches)

    return dedupeDocuments(search_results)


def getFederalRegisterDocuments(
    fr_session: requests.Session,
    regulations_session: requests.Session,
    fr_number: str,
) -> list[JsonDict]:
    response = fr_session.get(
        f"{FR_API_URL}/documents/{fr_number}.json",
        timeout=TIMEOUT,
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()
    fr_document = response.json()

    raw_docket_ids = fr_document.get("docket_ids") or []

    if isinstance(raw_docket_ids, str):
        docket_ids = {raw_docket_ids}
    else:
        docket_ids = {
            str(docket_id).strip()
            for docket_id in raw_docket_ids
            if str(docket_id).strip()
        }

    single_docket_id = fr_document.get("docket_id")

    if single_docket_id:
        docket_ids.add(str(single_docket_id).strip())

    documents: list[JsonDict] = []

    for docket_id in docket_ids:
        docket_documents = getDocuments(
            regulations_session,
            {"filter[docketId]": docket_id},
        )

        exact_matches = [
            document
            for document in docket_documents
            if matchesFRNumber(document, fr_number)
        ]

        if exact_matches:
            documents.extend(exact_matches)
            continue

        searchable_documents = getDocuments(
            regulations_session,
            {
                "filter[docketId]": docket_id,
                "filter[searchTerm]": fr_number,
            },
        )

        if searchable_documents:
            documents.extend(searchable_documents)
            continue

        # Some Regulations.gov document records omit frDocNum.
        # In that case, keep the primary docket documents that
        # comments can be attached to.
        documents.extend(
            document
            for document in docket_documents
            if (
                document.get("attributes", {}).get("objectId")
                and document.get("attributes", {}).get("documentType")
                in {
                    "Notice",
                    "Proposed Rule",
                    "Rule",
                    "Supporting & Related Material",
                }
            )
        )

    return dedupeDocuments(documents)


def getObjectIDs(
    documents: list[JsonDict],
) -> set[str]:
    object_ids: set[str] = set()

    for document in documents:
        object_id = document.get("attributes", {}).get("objectId")

        if object_id:
            object_ids.add(str(object_id))

    return object_ids


def countComments(
    session: requests.Session,
    documents: list[JsonDict],
) -> int:
    total = 0

    for object_id in getObjectIDs(documents):
        payload = requestJSON(
            session,
            f"{REGULATIONS_API_URL}/comments",
            {
                "filter[commentOnId]": object_id,
                "page[size]": 5,
                "page[number]": 1,
            },
        )

        total += int(
            payload.get("meta", {}).get(
                "totalElements",
                len(payload.get("data", [])),
            )
        )

    return total


def getCommentsForDocument(
    session: requests.Session,
    object_id: str,
) -> list[JsonDict]:
    comments: dict[str, JsonDict] = {}

    for page_number in range(1, MAX_PAGES + 1):
        payload = requestJSON(
            session,
            f"{REGULATIONS_API_URL}/comments",
            {
                "filter[commentOnId]": object_id,
                "page[size]": PAGE_SIZE,
                "page[number]": page_number,
            },
        )

        page = payload.get("data", [])

        if not isinstance(page, list):
            raise TypeError("Invalid comments response.")

        for comment in page:
            comment_id = comment.get("id")

            if comment_id:
                comments[str(comment_id)] = comment

        total = int(
            payload.get("meta", {}).get(
                "totalElements",
                len(comments),
            )
        )

        if len(comments) >= total or len(page) < PAGE_SIZE:
            return list(comments.values())

    raise RuntimeError(
        f"Comment search for {object_id} exceeded the "
        "Regulations.gov 5,000-result pagination limit."
    )


def getComments(fr_doc_number: str) -> list[JsonDict]:
    fr_number = normalizeFRNumber(fr_doc_number)

    if not fr_number:
        raise ValueError("Federal Register number cannot be empty.")

    with requests.Session() as fr_session:
        with requests.Session() as regulations_session:
            regulations_session.headers.update(
                {
                    "X-Api-Key": loadRegKey(),
                    "Accept": "application/json",
                }
            )

            regulations_documents = getRegulationsDocuments(
                regulations_session,
                fr_number,
            )

            federal_register_documents = getFederalRegisterDocuments(
                fr_session,
                regulations_session,
                fr_number,
            )

            regulations_count = countComments(
                regulations_session,
                regulations_documents,
            )

            federal_register_count = countComments(
                regulations_session,
                federal_register_documents,
            )

            if federal_register_count > regulations_count:
                source = "Federal Register docket route"
                selected_documents = federal_register_documents
            else:
                source = "Regulations.gov FR-number route"
                selected_documents = regulations_documents

            if not selected_documents:
                raise LookupError(
                    f"No Regulations.gov documents found for {fr_number}."
                )

            print(
                f"Federal Register route: "
                f"{federal_register_count} comments\n"
                f"Regulations.gov route: "
                f"{regulations_count} comments\n"
                f"Using: {source}"
            )

            comments: dict[str, JsonDict] = {}

            for object_id in getObjectIDs(selected_documents):
                document_comments = getCommentsForDocument(
                    regulations_session,
                    object_id,
                )

                for comment in document_comments:
                    comment_id = comment.get("id")

                    if comment_id:
                        comments[str(comment_id)] = comment

            return list(comments.values())


def getMetadata(comment_id: str) -> JsonDict:
    comment_id = comment_id.strip()

    if not comment_id:
        raise ValueError("Comment ID cannot be empty.")

    with requests.Session() as session:
        session.headers.update(
            {
                "X-Api-Key": loadRegKey(),
                "Accept": "application/json",
            }
        )

        payload = requestJSON(
            session,
            f"{REGULATIONS_API_URL}/comments/{comment_id}",
        )

    data = payload.get("data")

    if not data:
        raise LookupError(f"Comment not found: {comment_id}")

    return {
        "id": data["id"],
        **data.get("attributes", {}),
    }

# main
def processComments(
    frNum: str,
    spreadsheet_url: str | None = None,
) -> int:
    def getValue(data: dict, key: str):
        if data.get(key) not in (None, ""):
            return data[key]

        nested_data = data.get("data", {})
        if not isinstance(nested_data, dict):
            return None

        if key == "id" and nested_data.get("id"):
            return nested_data["id"]

        attributes = nested_data.get("attributes", {})
        if isinstance(attributes, dict):
            return attributes.get(key)

        return None

    def isEmpty(value) -> bool:
        if value is None:
            return True

        if isinstance(value, str):
            return not value.strip() or value.strip().lower() in {
                "null",
                "none",
                "n/a",
                "not available",
                "unknown",
            }

        return value in ([], {}, ())

    def humanizeKey(key) -> str:
        text = re.sub(r"(?<!^)(?=[A-Z])", " ", str(key))
        text = text.replace("_", " ").replace("-", " ")
        text = " ".join(text.split()).strip()

        replacements = {
            "ntee": "NTEE",
            "url": "URL",
            "urls": "URLs",
            "email": "Email",
            "phone": "Phone",
            "501c status": "501(c) Status",
            "501 c status": "501(c) Status",
            "professional profile": "Professional Profile",
            "professional email": "Professional Email",
            "professional phone": "Professional Phone",
            "organization website": "Organization Website",
        }

        return replacements.get(text.lower(), text.title())

    def parseStructuredValue(value):
        """Turn JSON/Python-looking strings back into normal data safely."""
        if not isinstance(value, str):
            return value

        text = value.strip()

        if not text:
            return ""

        looks_structured = (
            (text.startswith("{") and text.endswith("}"))
            or (text.startswith("[") and text.endswith("]"))
            or (text.startswith("(") and text.endswith(")"))
        )

        if not looks_structured:
            return value

        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return value

    def cleanText(value, *, preserve_paragraphs: bool = True) -> str:
        if isEmpty(value):
            return ""

        text = html.unescape(str(value))
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove model/document wrappers that should never appear in a cell.
        text = re.sub(
            r"```(?:json|python|text)?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = text.replace("```", "")
        text = re.sub(
            r"(?im)^\s*=+\s*(?:inline comment|attachment text|comment text)\s*=+\s*$",
            "",
            text,
        )

        # Clean whitespace without destroying intentional paragraphs.
        text = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        if preserve_paragraphs:
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def protectFormula(text: str) -> str:
        # addRows uses USER_ENTERED, so protect arbitrary extracted text from
        # becoming a spreadsheet formula.
        if text and text.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + text

        return text

    def indentText(text: str, spaces: int = 2) -> str:
        prefix = " " * spaces

        return "\n".join(prefix + line if line else line for line in text.splitlines())

    def uniqueItems(items: list[str]) -> list[str]:
        seen = set()
        output = []

        for item in items:
            cleaned = cleanText(
                item,
                preserve_paragraphs=False,
            )
            key = cleaned.casefold()

            if cleaned and key not in seen:
                seen.add(key)
                output.append(cleaned)

        return output

    def renderPlainValue(value) -> str:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return ""

        if isinstance(value, bool):
            return "Yes" if value else "No"

        if isinstance(value, dict):
            lines = []

            for key, nested_value in value.items():
                if isEmpty(nested_value):
                    continue

                rendered = renderPlainValue(nested_value)

                if not rendered:
                    continue

                label = humanizeKey(key)

                if "\n" in rendered:
                    lines.append(f"{label}:\n{indentText(rendered)}")
                else:
                    lines.append(f"{label}: {rendered}")

            return "\n".join(lines)

        if isinstance(value, (list, tuple, set)):
            items = uniqueItems(flattenTextItems(value))
            return "\n".join(f"• {item}" for item in items)

        return cleanText(value)

    def flattenTextItems(value) -> list[str]:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return []

        if isinstance(value, dict):
            items = []

            for key, nested_value in value.items():
                if isEmpty(nested_value):
                    continue

                rendered = renderPlainValue(nested_value)

                if rendered:
                    items.append(f"{humanizeKey(key)}: {rendered}")

            return items

        if isinstance(value, (list, tuple, set)):
            items = []

            for item in value:
                items.extend(flattenTextItems(item))

            return items

        text = cleanText(
            value,
            preserve_paragraphs=False,
        )
        text = re.sub(r"^[•\-*]\s*", "", text)

        return [text] if text else []

    def renderKeywords(value) -> str:
        return ", ".join(uniqueItems(flattenTextItems(value)))

    def renderBulletList(value) -> str:
        items = uniqueItems(flattenTextItems(value))

        return "\n".join(f"• {item.rstrip('.')}" for item in items)

    def renderContactInfo(value) -> str:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return ""

        if not isinstance(value, dict):
            return renderPlainValue(value)

        preferred_order = [
            "professional_email",
            "email",
            "professional_phone",
            "phone",
            "organization_website",
            "website",
            "professional_profile",
            "profile",
            "address",
        ]

        ordered_keys = []

        for key in preferred_order:
            if key in value and key not in ordered_keys:
                ordered_keys.append(key)

        ordered_keys.extend(key for key in value if key not in ordered_keys)

        lines = []

        for key in ordered_keys:
            nested_value = value.get(key)

            if isEmpty(nested_value):
                continue

            rendered = renderPlainValue(nested_value)

            if rendered:
                lines.append(f"{humanizeKey(key)}: {rendered}")

        return "\n".join(lines)

    def renderEvidence(value) -> str:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return ""

        if not isinstance(value, dict):
            return renderPlainValue(value)

        sections = []

        for field, evidence in value.items():
            if isEmpty(evidence):
                continue

            heading = humanizeKey(field)

            if isinstance(evidence, dict):
                details = []
                preferred_keys = (
                    "finding",
                    "evidence",
                    "quote",
                    "source",
                    "url",
                    "reason",
                    "notes",
                )

                ordered_keys = [key for key in preferred_keys if key in evidence]
                ordered_keys.extend(key for key in evidence if key not in ordered_keys)

                for key in ordered_keys:
                    nested_value = evidence.get(key)

                    if isEmpty(nested_value):
                        continue

                    rendered = renderPlainValue(nested_value)

                    if rendered:
                        details.append(f"{humanizeKey(key)}: {rendered}")

                if details:
                    sections.append(f"{heading}\n{indentText(chr(10).join(details))}")

            else:
                rendered = renderPlainValue(evidence)

                if rendered:
                    sections.append(f"{heading}: {rendered}")

        return "\n\n".join(sections)

    def renderSources(value) -> str:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return ""

        sources = value if isinstance(value, (list, tuple, set)) else [value]

        rendered_sources = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            if isinstance(source, dict):
                title = cleanText(
                    source.get("title")
                    or source.get("name")
                    or source.get("label")
                    or source.get("publisher")
                    or "",
                    preserve_paragraphs=False,
                )
                url = cleanText(
                    source.get("url")
                    or source.get("link")
                    or source.get("source")
                    or "",
                    preserve_paragraphs=False,
                )
                supports = cleanText(
                    source.get("supports")
                    or source.get("evidence")
                    or source.get("description")
                    or "",
                    preserve_paragraphs=False,
                )

                lines = [f"{index}. {title or url or 'Source'}"]

                if title and url:
                    lines.append(f"   {url}")

                if supports:
                    lines.append(f"   Supports: {supports}")

                rendered_sources.append("\n".join(lines))

            else:
                rendered = cleanText(source)

                if rendered:
                    rendered_sources.append(f"{index}. {rendered}")

        return "\n\n".join(rendered_sources)

    def confidenceLabel(score: float) -> str:
        if score >= 0.85:
            return "High"

        if score >= 0.60:
            return "Moderate"

        return "Low"

    def renderConfidence(value) -> str:
        value = parseStructuredValue(value)

        if isEmpty(value):
            return ""

        if not isinstance(value, dict):
            return renderPlainValue(value)

        lines = []

        for field, confidence in value.items():
            if isEmpty(confidence):
                continue

            label = humanizeKey(field)
            reason = ""

            if isinstance(confidence, dict):
                raw_score = (
                    confidence.get("score")
                    if confidence.get("score") is not None
                    else confidence.get("confidence")
                )
                raw_level = confidence.get("level")
                reason = cleanText(
                    confidence.get("reason") or confidence.get("notes") or "",
                    preserve_paragraphs=False,
                )
            else:
                raw_score = confidence
                raw_level = None

            display = ""

            if isinstance(raw_score, (int, float)):
                numeric = float(raw_score)
                normalized = numeric if 0 <= numeric <= 1 else numeric / 100

                if 0 <= normalized <= 1:
                    display = f"{confidenceLabel(normalized)} ({normalized:.0%})"
                else:
                    display = f"{numeric:g}"

            elif raw_score is not None:
                display = cleanText(
                    raw_score,
                    preserve_paragraphs=False,
                ).title()

            if not display and raw_level:
                display = cleanText(
                    raw_level,
                    preserve_paragraphs=False,
                ).title()

            if not display:
                display = renderPlainValue(confidence)

            line = f"{label}: {display}"

            if reason:
                line += f" | {reason}"

            lines.append(line)

        return "\n".join(lines)

    def renderDate(value) -> str:
        text = cleanText(
            value,
            preserve_paragraphs=False,
        )

        if not text:
            return ""

        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text

        date_text = f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}"
        time_text = parsed.strftime("%I:%M %p").lstrip("0")
        timezone_name = parsed.tzname()

        if timezone_name:
            return f"{date_text} at {time_text} {timezone_name}"

        return f"{date_text} at {time_text}"

    def renderFullText(value) -> str:
        text = cleanText(
            value,
            preserve_paragraphs=True,
        )

        text = re.sub(
            r"(?m)^\s*-{5,}\s*$",
            "",
            text,
        )
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def sheetValue(header: str, value) -> str:
        if isEmpty(value):
            return ""

        renderers = {
            "Contact Info": renderContactInfo,
            "Keywords": renderKeywords,
            "Issues Addressed": renderBulletList,
            "Evidence": renderEvidence,
            "Sources": renderSources,
            "Confidence": renderConfidence,
            "Date Submitted": renderDate,
            "Full Text": renderFullText,
        }

        renderer = renderers.get(
            header,
            renderPlainValue,
        )

        return protectFormula(renderer(value))

    frNum = frNum.strip()

    if not frNum:
        raise ValueError("frNum is required")

    spreadsheet_url = spreadsheet_url or os.getenv("COMMENT_SHEET_URL", "").strip()

    if not spreadsheet_url:
        raise RuntimeError("COMMENT_SHEET_URL is missing from .env")

    comments = getComments(frNum)

    if not comments:
        logger.warning(
            "No comments found for FR number %s",
            frNum,
        )
        return 0

    first_comment_id = str(comments[0].get("id") or "").strip()

    if not first_comment_id:
        raise ValueError("The first comment has no ID")

    first_metadata = getMetadata(first_comment_id)

    notice_name = "Notice"
    notice_document_id = getValue(
        first_metadata,
        "commentOnDocumentId",
    )

    if notice_document_id:
        try:
            notice_metadata = getMetadata(str(notice_document_id))
            notice_name = str(getValue(notice_metadata, "title") or "Notice")
        except Exception:
            logger.exception(
                "Could not retrieve notice title for document %s",
                notice_document_id,
            )

    tab_name = f"{frNum} - {notice_name}"
    tab_name = re.sub(
        r"[:\\/?*\[\]]",
        "-",
        tab_name,
    )
    tab_name = " ".join(tab_name.split())[:100]

    spreadsheet, _ = setupGoogleSheets(spreadsheet_url)
    worksheet = getTab(
        spreadsheet,
        tab_name,
    )

    headers = [
        "Comment ID",
        "Filename",
        "Date Submitted",
        "Submitter Name",
        "Organization Name",
        "Organization Type",
        "501c Status",
        "NTEE",
        "Organization Role",
        "Contact Info",
        "Keywords",
        "Brief Summary",
        "Issues Addressed",
        "Full Text",
        "Evidence",
        "Sources",
        "Confidence",
        "Research Notes",
    ]

    header_notes = [
        "Unique Regulations.gov comment identifier.",
        "Original attachment filename, when available.",
        "Date and time the comment was submitted.",
        "Verified name of the person who submitted the comment.",
        "Organization represented by the submitter.",
        "Classification of the organization.",
        "Verified nonprofit or 501(c) classification.",
        "National Taxonomy of Exempt Entities classification.",
        "The organization's role or relationship to the filing.",
        "Public professional contact information only.",
        "Major concepts and searchable terms from the comment.",
        "Concise summary of the comment's central argument.",
        "Specific policy or technical issues raised.",
        "Complete extracted comment or attachment text. Click the cell to inspect all content.",
        "Field-level supporting evidence used by the analysis.",
        "Public sources used during research.",
        "Field-level confidence values from the analysis.",
        "Uncertainty, conflicts, missing evidence, and other research notes.",
    ]

    ensureHeaders(
        worksheet,
        headers,
    )

    existing_comment_ids = getExistingFRIDs(worksheet)

    rows = []

    for comment_summary in comments:
        comment_id = str(comment_summary.get("id") or "").strip()

        if not comment_id:
            logger.warning("Skipping comment with no ID")
            continue

        if comment_id in existing_comment_ids:
            logger.info(
                "Skipping existing comment %s",
                comment_id,
            )
            continue

        try:
            metadata = (
                first_metadata
                if comment_id == first_comment_id
                else getMetadata(comment_id)
            )

            comment = initComment()

            fillMetadata(
                metadata,
                comment,
            )
            analyzeMetadata(
                metadata,
                comment,
            )

            bodyText = getCommentText(comment_id) or ""
            comment.comment_text = bodyText

            result = processCommentText(
                text=bodyText,
                metadata=metadata,
            )

            if not isinstance(result, dict):
                raise TypeError("processCommentText must return a dictionary")

            if not result.get("Comment ID"):
                result["Comment ID"] = comment_id

            if not result.get("Date Submitted"):
                result["Date Submitted"] = getValue(
                    metadata,
                    "postedDate",
                )

            if not result.get("Full Text"):
                result["Full Text"] = bodyText.replace("===== INLINE COMMENT =====", "")

            row = [
                sheetValue(
                    header,
                    result.get(header),
                )
                for header in headers
            ]

            rows.append(row)
            existing_comment_ids.add(comment_id)

        except Exception:
            logger.exception(
                "Failed to process comment %s",
                comment_id,
            )

    if rows:
        addRows(
            worksheet,
            rows,
        )

    column_count = len(headers)
    last_row = max(
        2,
        len(existing_comment_ids) + 1,
    )
    target_row_count = max(
        worksheet.row_count,
        last_row + 50,
    )
    sheet_id = worksheet.id

    column_widths = [
        190,  # Comment ID
        190,  # Filename
        145,  # Date Submitted
        175,  # Submitter Name
        230,  # Organization Name
        165,  # Organization Type
        110,  # 501c Status
        100,  # NTEE
        190,  # Organization Role
        280,  # Contact Info
        240,  # Keywords
        430,  # Brief Summary
        340,  # Issues Addressed
        600,  # Full Text
        440,  # Evidence
        360,  # Sources
        230,  # Confidence
        360,  # Research Notes
    ]

    requests = []

    # Remove old banded ranges from this generated worksheet before
    # replacing them, preventing duplicate formatting after reruns.
    try:
        sheet_metadata = spreadsheet.fetch_sheet_metadata(
            params={
                "fields": ("sheets(properties(sheetId),bandedRanges(bandedRangeId))")
            }
        )

        for sheet in sheet_metadata.get("sheets", []):
            properties = sheet.get("properties", {})

            if properties.get("sheetId") != sheet_id:
                continue

            for banded_range in sheet.get(
                "bandedRanges",
                [],
            ):
                banding_id = banded_range.get("bandedRangeId")

                if banding_id is not None:
                    requests.append({"deleteBanding": {"bandedRangeId": banding_id}})

    except Exception:
        logger.warning(
            "Could not inspect existing worksheet banding",
            exc_info=True,
        )

    requests.extend(
        [
            {
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": sheet_id,
                        "tabColor": {
                            "red": 0.11,
                            "green": 0.29,
                            "blue": 0.47,
                        },
                        "gridProperties": {
                            "rowCount": target_row_count,
                            "columnCount": column_count,
                            "frozenRowCount": 1,
                            "frozenColumnCount": 3,
                            "hideGridlines": True,
                        },
                    },
                    "fields": (
                        "tabColor,"
                        "gridProperties.rowCount,"
                        "gridProperties.columnCount,"
                        "gridProperties.frozenRowCount,"
                        "gridProperties.frozenColumnCount,"
                        "gridProperties.hideGridlines"
                    ),
                }
            },
            {
                "addBanding": {
                    "bandedRange": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 0,
                            "endRowIndex": last_row,
                            "startColumnIndex": 0,
                            "endColumnIndex": column_count,
                        },
                        "rowProperties": {
                            "headerColor": {
                                "red": 0.11,
                                "green": 0.29,
                                "blue": 0.47,
                            },
                            "firstBandColor": {
                                "red": 1.0,
                                "green": 1.0,
                                "blue": 1.0,
                            },
                            "secondBandColor": {
                                "red": 0.95,
                                "green": 0.97,
                                "blue": 0.99,
                            },
                        },
                    }
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {
                                "red": 0.11,
                                "green": 0.29,
                                "blue": 0.47,
                            },
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE",
                            "wrapStrategy": "WRAP",
                            "textFormat": {
                                "foregroundColor": {
                                    "red": 1.0,
                                    "green": 1.0,
                                    "blue": 1.0,
                                },
                                "fontFamily": "Arial",
                                "fontSize": 10,
                                "bold": True,
                            },
                            "padding": {
                                "top": 8,
                                "bottom": 8,
                                "left": 6,
                                "right": 6,
                            },
                        }
                    },
                    "fields": (
                        "userEnteredFormat.backgroundColor,"
                        "userEnteredFormat.horizontalAlignment,"
                        "userEnteredFormat.verticalAlignment,"
                        "userEnteredFormat.wrapStrategy,"
                        "userEnteredFormat.textFormat,"
                        "userEnteredFormat.padding"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "LEFT",
                            "verticalAlignment": "TOP",
                            "wrapStrategy": "WRAP",
                            "textFormat": {
                                "foregroundColor": {
                                    "red": 0.12,
                                    "green": 0.14,
                                    "blue": 0.18,
                                },
                                "fontFamily": "Arial",
                                "fontSize": 9,
                            },
                            "padding": {
                                "top": 6,
                                "bottom": 6,
                                "left": 6,
                                "right": 6,
                            },
                        }
                    },
                    "fields": (
                        "userEnteredFormat.horizontalAlignment,"
                        "userEnteredFormat.verticalAlignment,"
                        "userEnteredFormat.wrapStrategy,"
                        "userEnteredFormat.textFormat,"
                        "userEnteredFormat.padding"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 3,
                        "endColumnIndex": 5,
                    },
                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                    "fields": ("userEnteredFormat.textFormat.bold"),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 15,
                        "endColumnIndex": 16,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {
                                "foregroundColor": {
                                    "red": 0.05,
                                    "green": 0.32,
                                    "blue": 0.62,
                                }
                            }
                        }
                    },
                    "fields": ("userEnteredFormat.textFormat.foregroundColor"),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {
                                "foregroundColor": {
                                    "red": 0.05,
                                    "green": 0.28,
                                    "blue": 0.52,
                                },
                                "bold": True,
                            }
                        }
                    },
                    "fields": (
                        "userEnteredFormat."
                        "textFormat.foregroundColor,"
                        "userEnteredFormat."
                        "textFormat.bold"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE",
                        }
                    },
                    "fields": (
                        "userEnteredFormat.horizontalAlignment,"
                        "userEnteredFormat.verticalAlignment"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 2,
                        "endColumnIndex": 3,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE",
                        }
                    },
                    "fields": (
                        "userEnteredFormat.horizontalAlignment,"
                        "userEnteredFormat.verticalAlignment"
                    ),
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": last_row,
                        "startColumnIndex": 5,
                        "endColumnIndex": 8,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE",
                        }
                    },
                    "fields": (
                        "userEnteredFormat.horizontalAlignment,"
                        "userEnteredFormat.verticalAlignment"
                    ),
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "startIndex": 0,
                        "endIndex": 1,
                    },
                    "properties": {"pixelSize": 42},
                    "fields": "pixelSize",
                }
            },
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "ROWS",
                        "startIndex": 1,
                        "endIndex": last_row,
                    },
                    "properties": {"pixelSize": 105},
                    "fields": "pixelSize",
                }
            },
            {
                "updateBorders": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": last_row,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    },
                    "bottom": {
                        "style": "SOLID",
                        "color": {
                            "red": 0.78,
                            "green": 0.82,
                            "blue": 0.87,
                        },
                    },
                    "innerHorizontal": {
                        "style": "SOLID",
                        "color": {
                            "red": 0.88,
                            "green": 0.90,
                            "blue": 0.93,
                        },
                    },
                }
            },
            {
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": column_count,
                    },
                    "rows": [{"values": [{"note": note} for note in header_notes]}],
                    "fields": "note",
                }
            },
            {"clearBasicFilter": {"sheetId": sheet_id}},
            {
                "setBasicFilter": {
                    "filter": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 0,
                            "endRowIndex": last_row,
                            "startColumnIndex": 0,
                            "endColumnIndex": column_count,
                        }
                    }
                }
            },
        ]
    )

    for column_index, width in enumerate(column_widths):
        requests.append(
            {
                "updateDimensionProperties": {
                    "range": {
                        "sheetId": sheet_id,
                        "dimension": "COLUMNS",
                        "startIndex": column_index,
                        "endIndex": column_index + 1,
                    },
                    "properties": {"pixelSize": width},
                    "fields": "pixelSize",
                }
            }
        )

    spreadsheet.batch_update({"requests": requests})

    logger.info(
        "Added %d comments to tab %s",
        len(rows),
        tab_name,
    )

    return len(rows)


if __name__ == "__main__":
    processComments("2024-09852")
