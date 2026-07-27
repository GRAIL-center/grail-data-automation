COMMENT_SCHEMA = {
    # Identity
    "comment_id": None,
    "docket_id": None,
    "comment_on_document_id": None,
    "fr_doc_number": None,

    # Submitter
    "organization_name": None,
    "submitter_name": None,
    "submitter_role": None,
    "organization_type": None,
    "organization_subtype": None,

    # Contact and web presence
    "contact_information": [],
    "website_url": None,

    # Dates
    "date_submitted": None,
    "date_posted": None,
    "date_modified": None,

    # Source material
    "comment_text": None,
    "comment_text_source": None,
    "link_to_comment_text": None,
    "attachment_urls": [],
    "attachment_count": 0,
    "attachment_file_types": [],

    # Analysis
    "brief_summary_of_comment": None,
    "relevant_issues_addressed": [],
    "position_or_stance": None,
    "recommendations": [],
    "policy_requests": [],
    "evidence_or_sources_cited": [],
    "ai_topics": [],
    "affected_stakeholders": [],

    # Reliability
    "extraction_confidence": None,
    "field_metadata": {},
    "review_required": False,
    "review_fields": [],
    "review_reasons": [],
    "fields_inferred": [],
    "analysis_notes": [],
}
