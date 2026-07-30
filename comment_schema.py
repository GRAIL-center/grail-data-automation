# COMMENT_SCHEMA = {
#     # Identity
#     "comment_id": None,
#     "docket_id": None,
#     "comment_on_document_id": None,
#     "fr_doc_number": None,

#     # Submitter
#     "organization_name": None,
#     "submitter_name": None,
#     "submitter_role": None,
#     "organization_type": None,
#     "organization_subtype": None,

#     # Contact and web presence
#     "contact_information": [],
#     "website_url": None,

#     # Dates
#     "date_submitted": None,
#     "date_posted": None,
#     "date_modified": None,

#     # Source material
#     "comment_text": None,
#     "comment_text_source": None,
#     "link_to_comment_text": None,
#     "attachment_urls": [],
#     "attachment_count": 0,
#     "attachment_file_types": [],

#     # Analysis
#     "brief_summary_of_comment": None,
#     "relevant_issues_addressed": [],
#     "position_or_stance": None,
#     "recommendations": [],
#     "policy_requests": [],
#     "evidence_or_sources_cited": [],
#     "ai_topics": [],
#     "affected_stakeholders": [],

#     # Reliability
#     "extraction_confidence": None,
#     "field_metadata": {},
#     "review_required": False,
#     "review_fields": [],
#     "review_reasons": [],
#     "fields_inferred": [],
#     "analysis_notes": [],
# }

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CommentSchema(BaseModel):
    # identity
    comment_id: Optional[str] = None
    docket_id: Optional[str] = None
    comment_on_document_id: Optional[str] = None
    fr_doc_number: Optional[str] = None

    # submitter
    organization_name: Optional[str] = None
    submitter_name: Optional[str] = None
    submitter_role: Optional[str] = None
    organization_type: Optional[str] = None
    organization_subtype: Optional[str] = None

    # contact and web presence
    contact_information: List[str] = Field(default_factory=list)
    website_url: Optional[str] = None

    # dates
    date_submitted: Optional[datetime] = None
    date_posted: Optional[datetime] = None
    date_modified: Optional[datetime] = None

    # source material
    comment_text: Optional[str] = None
    comment_text_source: Optional[str] = None
    link_to_comment_text: Optional[str] = None
    attachment_urls: List[str] = Field(default_factory=list)
    attachment_count: int = 0
    attachment_file_types: List[str] = Field(default_factory=list)

    # analysis
    brief_summary_of_comment: Optional[str] = None
    relevant_issues_addressed: List[str] = Field(default_factory=list)
    position_or_stance: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    policy_requests: List[str] = Field(default_factory=list)
    evidence_or_sources_cited: List[str] = Field(default_factory=list)
    ai_topics: List[str] = Field(default_factory=list)
    affected_stakeholders: List[str] = Field(default_factory=list)

    # reliability
    extraction_confidence: Optional[float] = None  # Assumed float for a 0.0-1.0 score
    field_metadata: Dict[str, Any] = Field(default_factory=dict)
    review_required: bool = False
    review_fields: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    fields_inferred: List[str] = Field(default_factory=list)
    analysis_notes: List[str] = Field(default_factory=list)

    def findEmptyFields(self) -> List[str]:
        empty_fields = []
        for field, value in self.__dict__.items():
            if not value:
                empty_fields.append(field)
        return empty_fields
