"""Federal Register API client and deterministic notice relevance scoring."""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.audit import AuditTrail, atomic_write_json, atomic_write_text, sha256_file

logger = logging.getLogger(__name__)

FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1"
NOTICE_FIELDS = [
    "abstract",
    "action",
    "agencies",
    "agency_names",
    "body_html_url",
    "cfr_references",
    "cfr_topics",
    "citation",
    "comment_url",
    "comments_close_on",
    "dates",
    "docket_id",
    "docket_ids",
    "document_number",
    "effective_on",
    "html_url",
    "json_url",
    "pdf_url",
    "publication_date",
    "raw_text_url",
    "regulation_id_numbers",
    "regulations_dot_gov_info",
    "regulations_dot_gov_url",
    "subtype",
    "title",
    "topics",
    "type",
]


def buildSession(retries: int = 3) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "GRAIL/4.0 federal-register-notice-monitor",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetchFederalRegisterDocument(
    documentNumber: str,
    *,
    timeout: int = 30,
    retries: int = 3,
    fields: list[str] | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch and validate Federal Register metadata for one document number."""
    if not isinstance(documentNumber, str) or not re.fullmatch(
        r"\d{4}-\d{4,6}",
        documentNumber.strip(),
    ):
        raise ValueError("Invalid Federal Register document number")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    client = session or buildSession(retries)
    selected_fields = fields or NOTICE_FIELDS
    response = client.get(
        f"{FEDERAL_REGISTER_API}/documents/{documentNumber.strip()}.json",
        params=[("fields[]", field) for field in selected_fields],
        timeout=timeout,
    )
    response.raise_for_status()
    metadata = response.json()
    if not isinstance(metadata, dict):
        raise ValueError("Federal Register detail response must be an object")
    if metadata.get("document_number") != documentNumber.strip():
        raise ValueError(
            "Federal Register returned an unexpected document number"
        )
    return metadata


def _safe_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return normalized[:80] or "query"


def _request_record(response: requests.Response) -> dict[str, Any]:
    return {
        "url": response.url,
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "content_length": len(response.content),
    }


def searchFederalRegister(
    settings: dict[str, Any],
    runDirectory: str | Path,
    *,
    audit: AuditTrail,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Search configured terms, save every API page, and deduplicate candidates."""
    search = settings["search"]
    processing = settings["processing"]
    timeout = processing["request_timeout_seconds"]
    client = session or buildSession(processing["request_retries"])
    search_dir = Path(runDirectory) / "notice_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict[str, Any]] = {}
    request_records: list[dict[str, Any]] = []
    search_artifacts: list[str] = []

    for query_index, query in enumerate(search["query_terms"], start=1):
        for page in range(1, search["max_pages_per_query"] + 1):
            params: list[tuple[str, Any]] = [
                ("conditions[term]", query),
                ("conditions[publication_date][gte]", search["start_date"]),
                ("order", search["order"]),
                ("per_page", search["per_page"]),
                ("page", page),
            ]
            if search.get("end_date"):
                params.append(
                    ("conditions[publication_date][lte]", search["end_date"])
                )
            params.extend(
                ("conditions[type][]", document_type)
                for document_type in search["document_types"]
            )
            params.extend(("fields[]", field) for field in NOTICE_FIELDS)

            audit.record(
                "notice_search_request_started",
                query=query,
                page=page,
            )
            response = client.get(
                f"{FEDERAL_REGISTER_API}/documents.json",
                params=params,
                timeout=timeout,
            )
            record = {
                "query": query,
                "page": page,
                **_request_record(response),
            }
            request_records.append(record)
            if not response.ok:
                audit.record(
                    "notice_search_request_finished",
                    status="failed",
                    **record,
                    response_excerpt=response.text[:1_000],
                )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(
                payload.get("results"),
                list,
            ):
                raise ValueError("Federal Register search returned an invalid payload")

            artifact = (
                search_dir
                / f"{query_index:02d}_{_safe_slug(query)}_page_{page:03d}.json"
            )
            atomic_write_json(artifact, payload)
            search_artifacts.append(str(artifact))
            audit.record(
                "notice_search_request_finished",
                query=query,
                page=page,
                result_count=len(payload["results"]),
                artifact_path=str(artifact),
                artifact_sha256=sha256_file(artifact),
            )

            for result in payload["results"]:
                if not isinstance(result, dict):
                    continue
                document_number = result.get("document_number")
                if not isinstance(document_number, str) or not document_number:
                    continue
                existing = candidates.setdefault(
                    document_number,
                    {
                        **result,
                        "_discovery": {
                            "matched_query_terms": [],
                            "search_artifacts": [],
                        },
                    },
                )
                discovery = existing["_discovery"]
                if query not in discovery["matched_query_terms"]:
                    discovery["matched_query_terms"].append(query)
                if str(artifact) not in discovery["search_artifacts"]:
                    discovery["search_artifacts"].append(str(artifact))

            total_pages = payload.get("total_pages")
            if not payload["results"] or (
                isinstance(total_pages, int) and page >= total_pages
            ):
                break

    return {
        "candidates": list(candidates.values()),
        "request_records": request_records,
        "search_artifacts": search_artifacts,
    }


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _phrase_matches(text: str, terms: list[str]) -> list[str]:
    matches = []
    for term in terms:
        normalized_term = _normalized(term)
        if not normalized_term:
            continue
        # Short terms such as "AI" require word boundaries; longer phrases can
        # be matched literally after whitespace normalization.
        if len(normalized_term) <= 3:
            matched = bool(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
                    text,
                )
            )
        else:
            matched = normalized_term in text
        if matched:
            matches.append(term)
    return matches


def _agency_text(notice: dict[str, Any]) -> str:
    agencies = notice.get("agencies") or []
    values = []
    for agency in agencies:
        if isinstance(agency, dict):
            values.extend(
                value
                for value in [agency.get("name"), agency.get("raw_name")]
                if value
            )
        elif isinstance(agency, str):
            values.append(agency)
    values.extend(notice.get("agency_names") or [])
    return " ".join(str(value) for value in values)


def _is_open_for_comment(notice: dict[str, Any]) -> bool | None:
    close_value = notice.get("comments_close_on")
    if not close_value:
        return None
    try:
        return date.fromisoformat(str(close_value)[:10]) >= date.today()
    except ValueError:
        return None


def scoreNotice(
    notice: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    search = settings["search"]
    title = _normalized(notice.get("title"))
    body = _normalized(
        " ".join(
            str(value or "")
            for value in [
                notice.get("abstract"),
                notice.get("action"),
                _agency_text(notice),
            ]
        )
    )
    combined = f"{title} {body}"
    title_topics = _phrase_matches(title, search["topic_terms"])
    body_topics = _phrase_matches(body, search["topic_terms"])
    matched_topics = list(
        dict.fromkeys([*title_topics, *body_topics])
    )
    matched_actions = _phrase_matches(combined, search["action_terms"])
    matched_exclusions = _phrase_matches(combined, search["exclude_terms"])
    open_for_comment = _is_open_for_comment(notice)

    score = (
        len(title_topics) * 3
        + len([term for term in body_topics if term not in title_topics])
        + len(matched_actions) * 2
        + (2 if open_for_comment is True else 0)
        + len(notice.get("_discovery", {}).get("matched_query_terms", []))
    )
    eligible = (
        len(matched_topics) >= search["minimum_topic_matches"]
        and not matched_exclusions
        and (
            not search["require_action_term"]
            or bool(matched_actions)
        )
        and (
            not search["only_open_for_comment"]
            or open_for_comment is True
        )
    )
    return {
        "score": float(score),
        "eligible": eligible,
        "matched_topics": matched_topics,
        "matched_action_terms": matched_actions,
        "matched_exclude_terms": matched_exclusions,
        "open_for_comment": open_for_comment,
    }


def rankNotices(
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        score = scoreNotice(candidate, settings)
        if not score["eligible"]:
            continue
        ranked.append({**candidate, "_relevance": score})
    ranked.sort(
        key=lambda item: (
            item["_relevance"]["score"],
            str(item.get("publication_date") or ""),
            str(item.get("document_number") or ""),
        ),
        reverse=True,
    )
    return ranked[: settings["search"]["max_results"]]


def fetchNoticeDetail(
    documentNumber: str,
    settings: dict[str, Any],
    artifactDirectory: str | Path,
    *,
    audit: AuditTrail,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    processing = settings["processing"]
    timeout = processing["request_timeout_seconds"]
    client = session or buildSession(processing["request_retries"])
    artifact_dir = Path(artifactDirectory).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    response = client.get(
        f"{FEDERAL_REGISTER_API}/documents/{documentNumber}.json",
        params=[("fields[]", field) for field in NOTICE_FIELDS],
        timeout=timeout,
    )
    if not response.ok:
        audit.record(
            "notice_detail_downloaded",
            status="failed",
            document_number=documentNumber,
            **_request_record(response),
            response_excerpt=response.text[:1_000],
        )
    response.raise_for_status()
    metadata = response.json()
    if not isinstance(metadata, dict):
        raise ValueError("Federal Register detail response must be an object")
    returned_number = metadata.get("document_number")
    if returned_number != documentNumber:
        raise ValueError(
            f"Federal Register returned {returned_number!r}; expected {documentNumber!r}"
        )
    metadata_path = artifact_dir / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    audit.record(
        "notice_detail_downloaded",
        document_number=documentNumber,
        artifact_path=str(metadata_path),
        artifact_sha256=sha256_file(metadata_path),
    )

    body_html_path: Path | None = None
    body_text_path: Path | None = None
    body_text = ""
    body_error = None
    body_url = metadata.get("body_html_url")
    if processing["fetch_full_text"] and body_url:
        parsed = urlparse(str(body_url))
        if parsed.scheme != "https" or parsed.hostname not in {
            "www.federalregister.gov",
            "federalregister.gov",
        }:
            body_error = f"Unsupported Federal Register body URL: {body_url}"
        else:
            try:
                body_response = client.get(str(body_url), timeout=timeout)
                body_response.raise_for_status()
                body_html_path = artifact_dir / "body.html"
                atomic_write_text(body_html_path, body_response.text)
                soup = BeautifulSoup(body_response.text, "html.parser")
                for node in soup(["script", "style", "noscript"]):
                    node.decompose()
                body_text = "\n".join(
                    line
                    for line in (
                        re.sub(r"\s+", " ", line).strip()
                        for line in soup.get_text("\n").splitlines()
                    )
                    if line
                )
                body_text_path = artifact_dir / "body.txt"
                atomic_write_text(body_text_path, body_text)
                audit.record(
                    "notice_body_downloaded",
                    document_number=documentNumber,
                    html_path=str(body_html_path),
                    text_path=str(body_text_path),
                    text_character_count=len(body_text),
                    html_sha256=sha256_file(body_html_path),
                    text_sha256=sha256_file(body_text_path),
                )
            except Exception as exc:
                body_error = str(exc)
                logger.warning(
                    "Unable to fetch full notice text for %s: %s",
                    documentNumber,
                    exc,
                    extra={
                        "event": "notice_body_download_failed",
                        "document_number": documentNumber,
                    },
                )
                audit.record(
                    "notice_body_downloaded",
                    status="failed",
                    document_number=documentNumber,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

    return {
        "metadata": metadata,
        "metadata_path": str(metadata_path),
        "body_html_path": str(body_html_path) if body_html_path else None,
        "body_text_path": str(body_text_path) if body_text_path else None,
        "body_text": body_text,
        "body_error": body_error,
    }
