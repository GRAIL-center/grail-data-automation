# ask ai to fill in blanks from metadata

import html
import json
import logging
import re
from typing import Any

from comment_schema import CommentSchema
from src.collect_comments.process_comment_text import processCommentText
from src.collect_comments.retrieve_comment_body import getCommentText
from src.services.ai_client import PROVIDER_ERRORS, generate_json, get_client

logger = logging.getLogger(__name__)

def initComment() -> CommentSchema:
    return CommentSchema()

def prettify(obj: dict) -> str:
    return "".join([f"{key}: {value}\n" for key, value in obj.items()])

# uses the api to fill in blanks from metadata
def fillMetadata(
    metadata: dict[str, Any],
    comment: CommentSchema,
) -> CommentSchema:
    first_name = str(metadata.get("firstName") or "").strip()
    last_name = str(metadata.get("lastName") or "").strip()

    submitter_name = " ".join(part for part in (first_name, last_name) if part)

    title = str(metadata.get("title") or "")
    fr_match = re.search(
        r"FR\s+Doc\s*#?\s*([0-9]{4}-[0-9]+)",
        title,
        re.IGNORECASE,
    )

    organization_name = metadata.get("organization") or metadata.get("govAgency")

    contact_information = [
        str(value).strip()
        for value in (
            metadata.get("email"),
            metadata.get("phone"),
            metadata.get("address1"),
            metadata.get("address2"),
            metadata.get("city"),
            metadata.get("stateProvinceRegion"),
            metadata.get("zip"),
            metadata.get("country"),
        )
        if value and str(value).strip()
    ]

    comment.comment_id = metadata.get("id")
    comment.docket_id = metadata.get("docketId")
    comment.comment_on_document_id = metadata.get("commentOnDocumentId")
    comment.fr_doc_number = fr_match.group(1) if fr_match else None

    comment.organization_name = organization_name
    comment.submitter_name = submitter_name or None
    comment.submitter_role = metadata.get("submitterRep")
    comment.organization_type = metadata.get("category")
    comment.organization_subtype = metadata.get("govAgencyType")

    comment.contact_information = contact_information

    comment.date_submitted = metadata.get("receiveDate")
    comment.date_posted = metadata.get("postedDate")
    comment.date_modified = metadata.get("modifyDate")

    raw_comment_text = metadata.get("comment")

    comment.comment_text = (
        html.unescape(str(raw_comment_text)).strip() if raw_comment_text else None
    )

    comment.comment_text_source = (
        "Regulations.gov comment metadata" if comment.comment_text else None
    )

    return comment

# uses an ai with the api to fill in blanks from metadata
def analyzeMetadata(
    metadata: dict[str, Any],
    comment: CommentSchema,
):
    logger.info("Enriching metadata for comment %s", metadata.get("id", "unknown"))
    emptyFields = comment.findEmptyFields()
    emptyFieldsDict = {emptyField: getattr(comment, emptyField) for emptyField in emptyFields}

    prompt = f"Use the metadata to fill in the empty fields in the comment if possible. If not possible, leave the field empty.\nMetadata: {prettify(metadata)}\nEmpty Fields: {prettify(emptyFieldsDict)}"

    blocked_reason = get_client().provider_block_reason()
    if blocked_reason is not None:
        logger.info(
            "Skipping AI metadata enrichment for comment %s: %s",
            metadata.get("id", "unknown"),
            blocked_reason,
        )
        return comment

    try:
        result = generate_json(prompt, schema=emptyFieldsDict)
    except (*PROVIDER_ERRORS, json.JSONDecodeError, TypeError, ValueError) as error:
        logger.warning(
            "AI metadata enrichment failed for comment %s; preserving source data: %s",
            metadata.get("id", "unknown"),
            error,
        )
        return comment

    for field, value in result.items():
        setattr(comment, field, value)

    return comment

# # uses an ai with the api to fill in blanks from metadata
# def analyzeCommentBody(
#     comment: CommentSchema,
# ):


def analyzeComment(metadata: dict):
    logger.info("Analyzing standalone comment %s", metadata.get("id", "unknown"))
    comment = initComment()

    fillMetadata(metadata, comment)
    analyzeMetadata(metadata, comment)

    artifact_identifier = str(
        comment.fr_doc_number or metadata.get("docketId") or "unassigned"
    )
    bodyText = getCommentText(metadata.get("id", ""), artifact_identifier)
    comment.comment_text = bodyText

    res = processCommentText(text=bodyText, metadata=metadata)
    print(res)
