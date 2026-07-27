from __future__ import annotations

import logging
import math
import re
from typing import Any

import requests as req
from bs4 import BeautifulSoup as bs
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.services.config import loadCommentSheetUrl, loadRegKey
from src.services.sheets import getOrCreateTab, setupGoogleSheets

logger = logging.getLogger(__name__)

REGULATIONS_API_URL = "https://api.regulations.gov/v4"
FEDERAL_REGISTER_API_URL = "https://www.federalregister.gov/api/v1"

PAGE_SIZE = 250
MAX_API_PAGES = 20
MAX_RESULTS_PER_WINDOW = PAGE_SIZE * MAX_API_PAGES
REQUEST_TIMEOUT = 30
SHEET_BATCH_SIZE = 250
MAX_SHEET_CELL_LENGTH = 49_000

COMMENT_HEADERS = [
    "Comment ID",
    "Title",
    "Comment",
    "Posted Date",
    "Submitted By",
    "Organization",
]


def _createSession(api_key: str | None = None) -> Session:
    """Create one requests session with retries for temporary API failures."""
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session = req.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})

    if api_key:
        session.headers.update({"X-Api-Key": api_key})

    return session


def _getJson(
    session: Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a GET request and return a JSON object with useful errors."""
    try:
        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except req.RequestException as error:
        body = ""

        if getattr(error, "response", None) is not None:
            body = error.response.text[:1000]

        raise RuntimeError(
            f"Request failed for {url}: {error}. Response: {body}"
        ) from error

    try:
        payload = response.json()
    except ValueError as error:
        raise RuntimeError(
            f"Invalid JSON returned by {response.url}: "
            f"{response.text[:500]}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected a JSON object from {response.url}."
        )

    return payload


def _getTotalElements(payload: dict[str, Any], fallback: int) -> int:
    total = payload.get("meta", {}).get("totalElements", fallback)

    try:
        return max(0, int(total))
    except (TypeError, ValueError):
        return fallback


def _fetchPages(
    session: Session,
    endpoint: str,
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch every page available in one Regulations.gov search window.

    Regulations.gov exposes at most 20 pages per search window.
    """
    first_params = {
        **params,
        "page[number]": 1,
        "page[size]": PAGE_SIZE,
    }

    first_payload = _getJson(session, endpoint, first_params)
    first_page = first_payload.get("data", [])

    if not isinstance(first_page, list):
        raise RuntimeError("Regulations.gov returned an invalid data field.")

    total_elements = _getTotalElements(first_payload, len(first_page))
    total_pages = min(
        MAX_API_PAGES,
        max(1, math.ceil(total_elements / PAGE_SIZE)),
    )

    records = list(first_page)

    for page_number in range(2, total_pages + 1):
        page_payload = _getJson(
            session,
            endpoint,
            {
                **params,
                "page[number]": page_number,
                "page[size]": PAGE_SIZE,
            },
        )

        page_records = page_payload.get("data", [])

        if not isinstance(page_records, list):
            raise RuntimeError(
                f"Regulations.gov returned invalid data on page {page_number}."
            )

        records.extend(page_records)

        if len(page_records) < PAGE_SIZE:
            break

    return records, total_elements


def _formatCursor(last_modified_date: str) -> str:
    """
    Convert an API timestamp into the cursor format accepted by the API.

    No ZoneInfo dependency is needed. That dependency was the source of the
    Windows crash and added no useful value here.
    """
    cursor = last_modified_date.strip()

    if not cursor:
        raise ValueError("lastModifiedDate cannot be empty.")

    cursor = cursor.replace("T", " ").replace("Z", "")

    if "." in cursor:
        cursor = cursor.split(".", 1)[0]

    return cursor


def _deduplicateRecords(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        record_id = record.get("id")

        if record_id:
            records_by_id[str(record_id)] = record

    return list(records_by_id.values())


def _fetchAllComments(
    session: Session,
    object_id: str,
    document_id: str,
) -> list[dict[str, Any]]:
    """
    Fetch all comments for a document, including collections over 5,000.

    The API only exposes 5,000 results in one search window. For larger
    collections, lastModifiedDate is used as an inclusive cursor and IDs are
    deduplicated between windows.
    """
    comments_by_id: dict[str, dict[str, Any]] = {}
    cursor: str | None = None

    while True:
        params: dict[str, Any] = {
            "filter[commentOnId]": object_id,
            "sort": "lastModifiedDate",
        }

        if cursor:
            params["filter[lastModifiedDate][ge]"] = cursor

        window_comments, total_elements = _fetchPages(
            session,
            f"{REGULATIONS_API_URL}/comments",
            params,
        )

        previous_count = len(comments_by_id)

        for comment in window_comments:
            comment_id = comment.get("id")

            if comment_id:
                comments_by_id[str(comment_id)] = comment

        added_count = len(comments_by_id) - previous_count

        logger.info(
            "Fetched %d comment(s) in this window for %s; "
            "%d unique total.",
            len(window_comments),
            document_id,
            len(comments_by_id),
        )

        if total_elements <= MAX_RESULTS_PER_WINDOW:
            break

        if not window_comments:
            raise RuntimeError(
                f"Pagination stopped before all comments for {document_id} "
                "were collected."
            )

        last_modified_date = (
            window_comments[-1]
            .get("attributes", {})
            .get("lastModifiedDate")
        )

        if not last_modified_date:
            raise RuntimeError(
                f"Cannot continue pagination for {document_id} because the "
                "last comment has no lastModifiedDate."
            )

        next_cursor = _formatCursor(str(last_modified_date))

        if next_cursor == cursor and added_count == 0:
            raise RuntimeError(
                f"Pagination for {document_id} stopped making progress at "
                f"{next_cursor}."
            )

        cursor = next_cursor

    return list(comments_by_id.values())


def _normalizeDocument(document: dict[str, Any]) -> dict[str, Any]:
    attributes = document.get("attributes", {})

    return {
        "document_id": document.get("id"),
        "docket_id": attributes.get("docketId"),
        "object_id": attributes.get("objectId"),
        "document_type": attributes.get("documentType"),
        "subtype": attributes.get("subtype"),
        "title": attributes.get("title"),
        "fr_doc_num": attributes.get("frDocNum"),
        "posted_date": attributes.get("postedDate"),
        "comment_start_date": attributes.get("commentStartDate"),
        "comment_end_date": attributes.get("commentEndDate"),
        "open_for_comment": attributes.get("openForComment"),
        "withdrawn": attributes.get("withdrawn"),
    }


def getDocuments(
    fr_doc_id: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Resolve a Federal Register document number to its docket or dockets, then
    fetch every Regulations.gov document in those dockets.
    """
    fr_doc_id = fr_doc_id.strip()

    if not fr_doc_id:
        raise ValueError("Federal Register document ID cannot be empty.")

    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    session = _createSession(api_key)

    try:
        matching_documents, matching_total = _fetchPages(
            session,
            f"{REGULATIONS_API_URL}/documents",
            {
                "filter[frDocNum]": fr_doc_id,
                "sort": "documentId",
            },
        )

        if matching_total > MAX_RESULTS_PER_WINDOW:
            raise RuntimeError(
                f"The FR document lookup returned more than "
                f"{MAX_RESULTS_PER_WINDOW} results, which is not expected."
            )

        if not matching_documents:
            logger.warning(
                "No Regulations.gov documents found for %s.",
                fr_doc_id,
            )
            return []

        docket_ids = {
            document.get("attributes", {}).get("docketId")
            for document in matching_documents
            if document.get("attributes", {}).get("docketId")
        }

        if not docket_ids:
            logger.warning("No docket IDs found for %s.", fr_doc_id)
            return []

        documents_by_id: dict[str, dict[str, Any]] = {}

        for docket_id in sorted(docket_ids):
            docket_documents, docket_total = _fetchPages(
                session,
                f"{REGULATIONS_API_URL}/documents",
                {
                    "filter[docketId]": docket_id,
                    "sort": "documentId",
                },
            )

            if docket_total > MAX_RESULTS_PER_WINDOW:
                raise RuntimeError(
                    f"Docket {docket_id} contains more than "
                    f"{MAX_RESULTS_PER_WINDOW} documents."
                )

            for document in docket_documents:
                normalized = _normalizeDocument(document)
                document_id = normalized.get("document_id")

                if document_id:
                    documents_by_id[str(document_id)] = normalized

        documents = list(documents_by_id.values())

        logger.info(
            "Found %d document(s) across %d docket(s) for %s.",
            len(documents),
            len(docket_ids),
            fr_doc_id,
        )

        return documents
    finally:
        session.close()


def fetchComments(
    document: dict[str, Any],
    api_key: str,
) -> list[dict[str, Any]]:
    """Fetch all comments attached to one Regulations.gov document."""
    object_id = document.get("object_id")
    document_id = str(document.get("document_id") or "unknown document")

    if not object_id:
        logger.info(
            "Skipping %s because it has no object_id.",
            document_id,
        )
        return []

    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    session = _createSession(api_key)

    try:
        comments = _fetchAllComments(
            session,
            str(object_id),
            document_id,
        )

        logger.info(
            "Found %d comment(s) for %s.",
            len(comments),
            document_id,
        )

        return comments
    finally:
        session.close()


def collectComments(fr_doc_id: str) -> list[dict[str, Any]]:
    """Collect and deduplicate every comment associated with an FR document."""
    api_key = loadRegKey()

    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    fr_doc_id = fr_doc_id.strip()

    if not fr_doc_id:
        raise ValueError("Federal Register document ID cannot be empty.")

    documents = getDocuments(fr_doc_id, api_key)
    comments_by_id: dict[str, dict[str, Any]] = {}

    for document in documents:
        source = {
            "fr_doc_id": fr_doc_id,
            "document_id": document.get("document_id"),
            "docket_id": document.get("docket_id"),
            "object_id": document.get("object_id"),
            "document_type": document.get("document_type"),
            "document_title": document.get("title"),
        }

        for comment in fetchComments(document, api_key):
            comment_id = comment.get("id")

            if not comment_id:
                continue

            comment_id = str(comment_id)
            existing = comments_by_id.get(comment_id)

            if existing is None:
                comments_by_id[comment_id] = {
                    **comment,
                    "source": source,
                    "additional_sources": [],
                }
                continue

            all_sources = [
                existing.get("source"),
                *existing.get("additional_sources", []),
            ]

            if source not in all_sources:
                existing["additional_sources"].append(source)

    comments = sorted(
        comments_by_id.values(),
        key=lambda comment: (
            str(comment.get("attributes", {}).get("postedDate") or ""),
            str(comment.get("id") or ""),
        ),
    )

    logger.info(
        "Collected %d unique comment(s) for %s.",
        len(comments),
        fr_doc_id,
    )

    return comments


def _stripHtml(value: Any) -> str:
    if value is None:
        return ""

    text = bs(str(value), "html.parser").get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_SHEET_CELL_LENGTH:
        suffix = " ... [TRUNCATED FOR GOOGLE SHEETS]"
        text = text[: MAX_SHEET_CELL_LENGTH - len(suffix)] + suffix

    return text


def getNoticeTitle(fr_doc_id: str) -> str | None:
    url = f"{FEDERAL_REGISTER_API_URL}/documents/{fr_doc_id}.json"
    session = _createSession()

    try:
        payload = _getJson(
            session,
            url,
            {"fields[]": "title"},
        )
        title = payload.get("title")
        return str(title).strip() if title else None
    except RuntimeError as error:
        logger.warning(
            "Failed to fetch notice title for %s: %s",
            fr_doc_id,
            error,
        )
        return None
    finally:
        session.close()


def processComment(comment: dict[str, Any]) -> list[str]:
    attributes = comment.get("attributes", {})

    submitted_by = attributes.get("submittedBy")

    if not submitted_by:
        submitted_by = " ".join(
            part
            for part in [
                str(attributes.get("firstName") or "").strip(),
                str(attributes.get("lastName") or "").strip(),
            ]
            if part
        )

    organization = (
        attributes.get("organization")
        or attributes.get("organizationName")
        or ""
    )

    return [
        _stripHtml(comment.get("id")),
        _stripHtml(attributes.get("title")),
        _stripHtml(attributes.get("comment")),
        _stripHtml(attributes.get("postedDate")),
        _stripHtml(submitted_by),
        _stripHtml(organization),
    ]


def _sanitizeTabName(name: str) -> str:
    name = re.sub(r"[\\/?*\[\]:]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:100] or "Comments"


def _getSpreadsheet(sheet_url: str) -> Any:
    """Support either return style used by setupGoogleSheets()."""
    result = setupGoogleSheets(sheet_url)

    if isinstance(result, tuple):
        return result[0]

    return result


def _appendRows(worksheet: Any, rows: list[list[str]]) -> None:
    for start in range(0, len(rows), SHEET_BATCH_SIZE):
        worksheet.append_rows(
            rows[start : start + SHEET_BATCH_SIZE],
            value_input_option="RAW",
        )


def collectAndStoreComments(fr_doc_id: str) -> list[dict[str, Any]]:
    comments = collectComments(fr_doc_id)

    if not comments:
        return []

    title = getNoticeTitle(fr_doc_id)
    tab_label = f"{title} ({fr_doc_id})" if title else fr_doc_id
    tab_name = _sanitizeTabName(tab_label)

    spreadsheet = _getSpreadsheet(loadCommentSheetUrl())
    worksheet = getOrCreateTab(spreadsheet, tab_name)

    current_header = worksheet.row_values(1)

    if current_header and current_header[: len(COMMENT_HEADERS)] != COMMENT_HEADERS:
        raise RuntimeError(
            f"Tab {tab_name!r} has the wrong headers. Expected "
            f"{COMMENT_HEADERS}, found "
            f"{current_header[:len(COMMENT_HEADERS)]}."
        )

    existing_comment_ids = {
        str(comment_id).strip()
        for comment_id in worksheet.col_values(1)[1:]
        if str(comment_id).strip()
    }

    rows: list[list[str]] = []

    if not current_header:
        rows.append(COMMENT_HEADERS)

    for comment in comments:
        comment_id = str(comment.get("id") or "").strip()

        if not comment_id or comment_id in existing_comment_ids:
            continue

        existing_comment_ids.add(comment_id)
        rows.append(processComment(comment))

    if rows:
        _appendRows(worksheet, rows)

    new_comment_count = len(rows) - (1 if not current_header else 0)

    logger.info(
        "Wrote %d new comment(s) to tab %r.",
        new_comment_count,
        tab_name,
    )

    return comments


if __name__ == "__main__":
    fr_doc_id = "2026-08281"
    collectAndStoreComments(fr_doc_id)
