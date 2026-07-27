import logging
import os

import requests as req
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REGULATIONS_API_URL = "https://api.regulations.gov/v4"
PAGE_SIZE = 250
MAX_DOCUMENT_PAGES = 10_000


def loadAPIKey():
    """Load the Regulations.gov API key from the environment."""
    api_key = os.getenv("REGULATION_API_KEY", "").strip()

    if not api_key:
        raise ValueError("REGULATION_API_KEY not set")

    return api_key


def _document_pages(headers, filters, label):
    documents = []
    page_number = 1
    reported_total = None
    while page_number <= MAX_DOCUMENT_PAGES:
        response = req.get(
            f"{REGULATIONS_API_URL}/documents",
            headers=headers,
            params={
                **filters,
                "page[number]": page_number,
                "page[size]": PAGE_SIZE,
            },
            timeout=30,
        )
        if not response.ok:
            logger.error(
                "%s page %d failed. Status: %s. Body: %s",
                label,
                page_number,
                response.status_code,
                response.text[:1000],
            )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"{label} returned a non-object response")
        page_documents = payload.get("data", [])
        if not isinstance(page_documents, list):
            raise ValueError(f"{label} returned invalid document data")
        documents.extend(page_documents)

        meta = payload.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"{label} returned invalid pagination metadata")
        total_elements = meta.get("totalElements")
        if (
            isinstance(total_elements, int)
            and not isinstance(total_elements, bool)
            and total_elements >= 0
        ):
            reported_total = total_elements
            if len(documents) >= total_elements:
                break
        if len(page_documents) < PAGE_SIZE:
            break
        page_number += 1
    else:
        raise RuntimeError(
            f"{label} exceeded {MAX_DOCUMENT_PAGES} pages"
        )

    if reported_total is not None and len(documents) < reported_total:
        raise RuntimeError(
            f"{label} reported {reported_total} documents but only "
            f"{len(documents)} were retrieved"
        )
    return documents


def getDocuments(fr_doc_id, api_key):
    """Resolve a Federal Register number and fetch every docket document."""
    if not api_key:
        raise RuntimeError("Regulations.gov API key is missing.")

    fr_doc_id = fr_doc_id.strip()
    if not fr_doc_id:
        raise ValueError("Federal Register document ID cannot be empty.")

    headers = {
        "X-Api-Key": api_key,
        "Accept": "application/json",
    }
    fr_results = _document_pages(
        headers,
        {"filter[frDocNum]": fr_doc_id},
        f"FR document lookup for {fr_doc_id}",
    )

    if not fr_results:
        logger.warning(
            "No Regulations.gov documents found for Federal Register ID %s",
            fr_doc_id,
        )
        return []

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

    documents_by_id = {}
    for docket_id in sorted(docket_ids):
        for document in _document_pages(
            headers,
            {"filter[docketId]": docket_id},
            f"Docket lookup for {docket_id}",
        ):
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
