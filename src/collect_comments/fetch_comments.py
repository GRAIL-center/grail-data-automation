from __future__ import annotations

import re
from typing import Any

import requests

from src.services.config import loadRegKey
from src.collect_comments.analyze_comments import analyzeComment
from src.collect_comments.retrieve_comment_body import getCommentText

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
    document_fr_number = (
        document.get("attributes", {}).get("frDocNum")
    )

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
        document
        for document in documents
        if matchesFRNumber(document, fr_number)
    ]

    if exact_matches:
        return dedupeDocuments(exact_matches)

    search_results = getDocuments(
        session,
        {"filter[searchTerm]": fr_number},
    )

    exact_search_matches = [
        document
        for document in search_results
        if matchesFRNumber(document, fr_number)
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
        object_id = (
            document.get("attributes", {}).get("objectId")
        )

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
        raise ValueError(
            "Federal Register number cannot be empty."
        )

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

            federal_register_documents = (
                getFederalRegisterDocuments(
                    fr_session,
                    regulations_session,
                    fr_number,
                )
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
                    "No Regulations.gov documents found for "
                    f"{fr_number}."
                )

            print(
                f"Federal Register route: "
                f"{federal_register_count} comments\n"
                f"Regulations.gov route: "
                f"{regulations_count} comments\n"
                f"Using: {source}"
            )

            comments: dict[str, JsonDict] = {}

            for object_id in getObjectIDs(
                selected_documents
            ):
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

if __name__ == "__main__":
    comments = getComments("2021-25145")

    print(f"Found {len(comments)} comments")

    comment = comments[1]
    print(getCommentText(comment.get("id")))
