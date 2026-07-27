import logging
import math
import re
from typing import Any

import requests as req
from bs4 import BeautifulSoup as bs

from src.services.config import loadCommentSheetUrl, loadRegKey
from src.services.sheets import addRow, getOrCreateTab, setupGoogleSheets

logger = logging.getLogger(__name__)

REGULATIONS_API_URL = "https://api.regulations.gov/v4"
PAGE_SIZE = 250
MAX_STANDARD_PAGES = 20


def getDocuments(
    fr_doc_id: str,
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Resolve a Federal Register document number to its docket or dockets,
    then fetch every Regulations.gov document in those dockets.

    Returns document records containing:
    - document_id
    - docket_id
    - object_id
    - document_type
    - subtype
    - title
    - fr_doc_num
    - posted_date
    - comment_start_date
    - comment_end_date
    - open_for_comment
    - withdrawn
    """

    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    fr_doc_id = fr_doc_id.strip()

    if not fr_doc_id:
        raise ValueError("Federal Register document ID cannot be empty.")

    headers = {
        "X-Api-Key": api_key,
        "Accept": "application/json",
    }

    # Step 1: Find Regulations.gov records matching the FR document number.
    fr_response = req.get(
        f"{REGULATIONS_API_URL}/documents",
        headers=headers,
        params={
            "filter[frDocNum]": fr_doc_id,
            "page[size]": PAGE_SIZE,
        },
        timeout=30,
    )

    if not fr_response.ok:
        logger.error(
            "FR document lookup failed for %s. Status: %s. Body: %s",
            fr_doc_id,
            fr_response.status_code,
            fr_response.text[:1000],
        )

    fr_response.raise_for_status()

    fr_results = fr_response.json().get("data", [])

    if not fr_results:
        logger.warning(
            "No Regulations.gov documents found for Federal Register ID %s",
            fr_doc_id,
        )
        return []

    # One FR publication can theoretically map to multiple dockets.
    docket_ids = {
        document.get("attributes", {}).get("docketId")
        for document in fr_results
        if document.get("attributes", {}).get("docketId")
    }

    if not docket_ids:
        logger.warning(
            "No docket IDs found for Federal Register ID %s",
            fr_doc_id,
        )
        return []

    # Step 2: Fetch every document in every resolved docket.
    documents_by_id: dict[str, dict[str, Any]] = {}

    for docket_id in sorted(docket_ids):
        docket_response = req.get(
            f"{REGULATIONS_API_URL}/documents",
            headers=headers,
            params={
                "filter[docketId]": docket_id,
                "page[size]": PAGE_SIZE,
            },
            timeout=30,
        )

        if not docket_response.ok:
            logger.error(
                "Docket lookup failed for %s. Status: %s. Body: %s",
                docket_id,
                docket_response.status_code,
                docket_response.text[:1000],
            )

        docket_response.raise_for_status()

        for document in docket_response.json().get("data", []):
            attributes = document.get("attributes", {})
            document_id = document.get("id")

            if not document_id:
                continue

            documents_by_id[document_id] = {
                "document_id": document_id,
                "docket_id": attributes.get("docketId"),
                "object_id": attributes.get("objectId"),
                "fr_doc_num": attributes.get("frDocNum"),
                "document_type": attributes.get("documentType"),
                "subtype": attributes.get("subtype"),
                "title": attributes.get("title"),
                "posted_date": attributes.get("postedDate"),
                "comment_start_date": attributes.get("commentStartDate"),
                "comment_end_date": attributes.get("commentEndDate"),
                "open_for_comment": attributes.get("openForComment"),
                "withdrawn": attributes.get("withdrawn"),
            }

    documents = list(documents_by_id.values())

    logger.info(
        "Found %d document(s) across %d docket(s) for Federal Register ID %s",
        len(documents),
        len(docket_ids),
        fr_doc_id,
    )

    return documents


def fetchComments(
    document: dict[str, Any],
    api_key: str,
) -> list[dict[str, Any]]:
    """
    Fetch all comment summaries attached to one Regulations.gov document.

    Comments are retrieved using the document's internal object_id.
    """

    object_id = document.get("object_id")
    document_id = document.get("document_id")

    if not object_id:
        logger.warning(
            "Skipping document %s because it has no object_id.",
            document_id,
        )
        return []

    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    headers = {
        "X-Api-Key": api_key,
        "Accept": "application/json",
    }

    first_response = req.get(
        f"{REGULATIONS_API_URL}/comments",
        headers=headers,
        params={
            "filter[commentOnId]": object_id,
            "page[number]": 1,
            "page[size]": PAGE_SIZE,
            "sort": "documentId",
        },
        timeout=30,
    )

    if not first_response.ok:
        logger.error(
            "Comment lookup failed for document %s. Status: %s. Body: %s",
            document_id,
            first_response.status_code,
            first_response.text[:1000],
        )

    first_response.raise_for_status()
    first_payload = first_response.json()

    comments = list(first_payload.get("data", []))

    total_elements = first_payload.get("meta", {}).get(
        "totalElements",
        len(comments),
    )

    total_pages = math.ceil(total_elements / PAGE_SIZE)

    if total_pages > MAX_STANDARD_PAGES:
        raise RuntimeError(
            f"Document {document_id} has {total_elements} comments, "
            f"which exceeds the normal {MAX_STANDARD_PAGES}-page API limit. "
            "A date-window collection strategy is required."
        )

    for page_number in range(2, total_pages + 1):
        response = req.get(
            f"{REGULATIONS_API_URL}/comments",
            headers=headers,
            params={
                "filter[commentOnId]": object_id,
                "page[number]": page_number,
                "page[size]": PAGE_SIZE,
                "sort": "documentId",
            },
            timeout=30,
        )

        if not response.ok:
            logger.error(
                "Comment page %d failed for document %s. Status: %s. Body: %s",
                page_number,
                document_id,
                response.status_code,
                response.text[:1000],
            )

        response.raise_for_status()
        comments.extend(response.json().get("data", []))

    logger.info(
        "Found %d comment(s) for document %s",
        len(comments),
        document_id,
    )

    return comments


def collectComments(fr_doc_id: str) -> list[dict[str, Any]]:
    """
    Collect all comments associated with a Federal Register document number.

    Process:
    1. Resolve the FR document number to one or more docket IDs.
    2. Fetch every document inside those dockets.
    3. Fetch comments for each document using its object_id.
    4. Add source document metadata to every comment.
    5. Deduplicate comments by Regulations.gov comment ID.

    Returns a list of unique comment summary records.
    """

    api_key = loadRegKey()

    documents = getDocuments(
        fr_doc_id=fr_doc_id,
        api_key=api_key,
    )

    if not documents:
        logger.warning(
            "No documents were found for Federal Register ID %s",
            fr_doc_id,
        )
        return []

    comments_by_id: dict[str, dict[str, Any]] = {}

    for document in documents:
        # Withdrawn documents are normally not useful comment targets.
        if document.get("withdrawn") is True:
            logger.info(
                "Skipping withdrawn document %s",
                document.get("document_id"),
            )
            continue

        document_comments = fetchComments(
            document=document,
            api_key=api_key,
        )

        for comment in document_comments:
            comment_id = comment.get("id")

            if not comment_id:
                logger.warning(
                    "Skipping comment without an ID from document %s",
                    document.get("document_id"),
                )
                continue

            comment_record = {
                **comment,
                "source": {
                    "fr_doc_id": fr_doc_id,
                    "document_id": document.get("document_id"),
                    "docket_id": document.get("docket_id"),
                    "object_id": document.get("object_id"),
                    "document_type": document.get("document_type"),
                    "document_title": document.get("title"),
                },
            }

            existing = comments_by_id.get(comment_id)

            if existing is None:
                comments_by_id[comment_id] = comment_record
                continue

            # Preserve every source document if the same comment appears twice.
            existing_sources = existing.setdefault(
                "additional_sources",
                [],
            )

            new_source = comment_record["source"]

            if new_source != existing.get("source"):
                existing_sources.append(new_source)

    comments = list(comments_by_id.values())

    logger.info(
        "Collected %d unique comment(s) from %d document(s) for Federal Register ID %s",
        len(comments),
        len(documents),
        fr_doc_id,
    )

    return comments


COMMENT_HEADERS = [
    "Comment ID",
    "Title",
    "Comment",
    "Posted Date",
    "Submitted By",
    "Organization",
]


def _stripHtml(html: str) -> str:
    if not html:
        return ""
    text = bs(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def getNoticeTitle(fr_doc_id: str) -> str | None:
    url = f"https://www.federalregister.gov/api/v1/documents/{fr_doc_id}.json?fields[]=title"
    try:
        res = req.get(url, timeout=30)
        res.raise_for_status()
        return res.json().get("title")
    except req.RequestException as e:
        logger.error("Failed to fetch notice title for %s: %s", fr_doc_id, e)
        return None


def processComment(comment: dict) -> list:
    attrs = comment.get("attributes", {})
    return [
        comment.get("id", ""),
        attrs.get("title", ""),
        _stripHtml(attrs.get("comment", "")),
        attrs.get("postedDate", ""),
        attrs.get("submittedBy", ""),
        attrs.get("organization", ""),
    ]


def _sanitizeTabName(name: str) -> str:
    name = re.sub(r"[\\/?*\[\]:]+", "", name)
    return name.strip()[:100]


def collectAndStoreComments(fr_doc_id: str) -> list[dict[str, Any]]:
    comments = collectComments(fr_doc_id)

    if not comments:
        return comments

    title = getNoticeTitle(fr_doc_id)
    tab_label = f"{title} ({fr_doc_id})" if title else fr_doc_id
    tab_name = _sanitizeTabName(tab_label)

    spreadsheet, _default_ws = setupGoogleSheets(loadCommentSheetUrl())
    worksheet = getOrCreateTab(spreadsheet, tab_name)

    if not worksheet.get_all_values():
        addRow(worksheet, COMMENT_HEADERS)

    for comment in comments:
        addRow(worksheet, processComment(comment))

    logger.info("Wrote %d comment(s) to tab '%s'", len(comments), tab_name)

    return comments


if __name__ == "__main__":
    fr_doc_id = "2026-08281"
    collectAndStoreComments(fr_doc_id)
