import logging
import math

import requests as req

from .documents import PAGE_SIZE, REGULATIONS_API_URL, getDocuments, loadAPIKey

logger = logging.getLogger(__name__)

MAX_STANDARD_PAGES = 20
MAX_DATE_WINDOWS = 10_000


def _comment_page(
    *,
    object_id,
    api_key,
    page_number,
    last_modified_cursor=None,
):
    params = {
        "filter[commentOnId]": object_id,
        "page[number]": page_number,
        "page[size]": PAGE_SIZE,
        "sort": "lastModifiedDate,documentId",
    }
    if last_modified_cursor:
        params["filter[lastModifiedDate][ge]"] = last_modified_cursor
    response = req.get(
        f"{REGULATIONS_API_URL}/comments",
        headers={
            "X-Api-Key": api_key,
            "Accept": "application/json",
        },
        params=params,
        timeout=30,
    )
    return response


def fetchComments(document, api_key):
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

    first_response = _comment_page(
        object_id=object_id,
        api_key=api_key,
        page_number=1,
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

    total_elements = first_payload.get("meta", {}).get(
        "totalElements",
        len(first_payload.get("data", [])),
    )
    if (
        isinstance(total_elements, bool)
        or not isinstance(total_elements, int)
        or total_elements < 0
    ):
        raise ValueError(
            f"Comment search returned invalid totalElements for {document_id}"
        )

    comments_by_id = {}
    last_modified_cursor = None
    window_number = 0
    pending_first_payload = first_payload

    while len(comments_by_id) < total_elements:
        window_number += 1
        if window_number > MAX_DATE_WINDOWS:
            raise RuntimeError(
                f"Comment pagination exceeded {MAX_DATE_WINDOWS} date windows "
                f"for document {document_id}"
            )
        window_start_count = len(comments_by_id)
        first_window_payload = pending_first_payload
        pending_first_payload = None
        if first_window_payload is None:
            response = _comment_page(
                object_id=object_id,
                api_key=api_key,
                page_number=1,
                last_modified_cursor=last_modified_cursor,
            )
            if not response.ok:
                logger.error(
                    "Comment window %d page 1 failed for document %s. "
                    "Status: %s. Body: %s",
                    window_number,
                    document_id,
                    response.status_code,
                    response.text[:1000],
                )
            response.raise_for_status()
            first_window_payload = response.json()

        window_total = first_window_payload.get("meta", {}).get(
            "totalElements",
            len(first_window_payload.get("data", [])),
        )
        window_pages = min(
            MAX_STANDARD_PAGES,
            max(1, math.ceil(window_total / PAGE_SIZE)),
        )
        last_page_comments = []
        for page_number in range(1, window_pages + 1):
            if page_number == 1:
                payload = first_window_payload
            else:
                response = _comment_page(
                    object_id=object_id,
                    api_key=api_key,
                    page_number=page_number,
                    last_modified_cursor=last_modified_cursor,
                )
                if not response.ok:
                    logger.error(
                        "Comment window %d page %d failed for document %s. "
                        "Status: %s. Body: %s",
                        window_number,
                        page_number,
                        document_id,
                        response.status_code,
                        response.text[:1000],
                    )
                response.raise_for_status()
                payload = response.json()
            page_comments = payload.get("data", [])
            if not isinstance(page_comments, list):
                raise ValueError(
                    f"Comment page returned invalid data for {document_id}"
                )
            last_page_comments = page_comments
            for comment in page_comments:
                if not isinstance(comment, dict):
                    continue
                comment_id = comment.get("id")
                if comment_id:
                    comments_by_id[comment_id] = comment
            if len(page_comments) < PAGE_SIZE:
                break

        if len(comments_by_id) >= total_elements:
            break
        if len(comments_by_id) == window_start_count:
            raise RuntimeError(
                f"Date-window pagination made no progress for document "
                f"{document_id} at cursor {last_modified_cursor!r}"
            )
        if not last_page_comments:
            break
        last_modified_cursor = (
            last_page_comments[-1]
            .get("attributes", {})
            .get("lastModifiedDate")
        )
        if not last_modified_cursor:
            raise RuntimeError(
                f"Cannot continue high-volume comment pagination for "
                f"{document_id}: lastModifiedDate is missing"
            )
        logger.info(
            "Collected %d/%d comment(s) for %s after date window %d",
            len(comments_by_id),
            total_elements,
            document_id,
            window_number,
        )

    comments = list(comments_by_id.values())
    if len(comments) < total_elements:
        logger.warning(
            "Regulations.gov reported %d comments for %s but returned %d unique IDs",
            total_elements,
            document_id,
            len(comments),
        )

    logger.info(
        "Found %d comment(s) for document %s",
        len(comments),
        document_id,
    )

    return comments


def collectComments(fr_doc_id):
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

    api_key = loadAPIKey()

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

    comments_by_id = {}

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
        "Collected %d unique comment(s) from %d document(s) "
        "for Federal Register ID %s",
        len(comments),
        len(documents),
        fr_doc_id,
    )

    return comments
