import html
import json
import logging
import math
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.audit import atomic_write_json, atomic_write_text, sha256_file
from src.ocr import extract_pdf_text

logger = logging.getLogger(__name__)

ATTACHMENT_DOWNLOAD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
ATTACHMENT_DOWNLOAD_REFERER = "https://www.regulations.gov/"
ATTACHMENT_AUTH_HOSTS = frozenset(
    {
        "api.regulations.gov",
        "api-staging.regulations.gov",
        "downloads.regulations.gov",
    }
)
ATTACHMENT_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
ATTACHMENT_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
ATTACHMENT_CONNECT_TIMEOUT_SECONDS = 10
ATTACHMENT_READ_TIMEOUT_SECONDS = 60
ATTACHMENT_DOWNLOAD_RETRIES = 3
ATTACHMENT_MAX_REDIRECTS = 4
ATTACHMENT_CHUNK_SIZE = 1024 * 1024

PAGE_MARKER = re.compile(r"^--- PAGE \d+ ---$", re.MULTILINE)
PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", re.IGNORECASE)
CONFIDENTIALITY_NOTICE = re.compile(
    r"(?:confidentiality notice|intended recipient|privileged and confidential)",
    re.IGNORECASE,
)
HTML_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
HTML_IGNORED_TAGS = frozenset({"script", "style", "noscript"})


class _CommentHTMLTextExtractor(HTMLParser):
    """Turn API comment HTML into readable text without executing markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in HTML_IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if normalized == "br" or normalized in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        normalized = tag.casefold()
        if (
            not self._ignored_depth
            and normalized not in HTML_IGNORED_TAGS
            and (normalized == "br" or normalized in HTML_BLOCK_TAGS)
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in HTML_IGNORED_TAGS:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if not self._ignored_depth and normalized in HTML_BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def normalizeCommentHTML(text: str) -> str:
    """Decode entities and preserve visible HTML boundaries as line breaks."""
    if not isinstance(text, str):
        raise TypeError("Comment text must be a string")
    parser = _CommentHTMLTextExtractor()
    parser.feed(text)
    parser.close()
    # A second, bounded unescape handles API values such as ``&amp;#39;``
    # without modifying the separately persisted raw comment text.
    return html.unescape("".join(parser.parts))


def _trusted_attachment_auth_host(hostname: str | None) -> bool:
    return bool(hostname and hostname.casefold() in ATTACHMENT_AUTH_HOSTS)


def _attachment_request_headers(url: str) -> tuple[dict[str, str], bool]:
    parsed = urlparse(url)
    headers = {
        "Accept": (
            "application/pdf,"
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document,"
            "image/png,image/jpeg,text/plain,application/octet-stream,*/*"
        ),
        "Referer": ATTACHMENT_DOWNLOAD_REFERER,
        "User-Agent": ATTACHMENT_DOWNLOAD_USER_AGENT,
    }
    api_key = os.getenv("REGULATION_API_KEY", "").strip()
    authenticated = bool(
        api_key and _trusted_attachment_auth_host(parsed.hostname)
    )
    if authenticated:
        headers["X-Api-Key"] = api_key
    return headers, authenticated


def _safe_download_url(url: str) -> str:
    """Return a query-free URL suitable for logs."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def buildAttachmentSession(
    retries: int = ATTACHMENT_DOWNLOAD_RETRIES,
) -> requests.Session:
    """Build a bounded-retry session for public attachment downloads."""
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("Attachment retries must be a non-negative integer")
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        redirect=0,
        other=0,
        backoff_factor=0.5,
        status_forcelist=ATTACHMENT_RETRY_STATUSES,
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=8,
        pool_maxsize=8,
    )
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def _retry_attempt_count(response: requests.Response) -> int:
    retries = getattr(getattr(response, "raw", None), "retries", None)
    history = getattr(retries, "history", ())
    return 1 + len(history or ())


def _open_attachment_response(
    session: requests.Session,
    url: str,
    *,
    timeout: tuple[int, int],
    max_redirects: int = ATTACHMENT_MAX_REDIRECTS,
) -> tuple[requests.Response, dict[str, int | bool | str]]:
    """Open one attachment while preventing API-key leakage on redirects."""
    current_url = url
    redirect_count = 0
    request_attempts = 0
    authenticated_request = False

    while True:
        parsed = urlparse(current_url)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise ValueError(
                "Attachment URLs and redirects must use HTTPS with a hostname"
            )
        headers, authenticated = _attachment_request_headers(current_url)
        authenticated_request = authenticated_request or authenticated
        response = session.get(
            current_url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
        request_attempts += _retry_attempt_count(response)
        if response.status_code not in ATTACHMENT_REDIRECT_STATUSES:
            return response, {
                "authenticated_request": authenticated_request,
                "download_attempts": request_attempts,
                "redirect_count": redirect_count,
                "response_host": parsed.hostname.casefold(),
            }

        location = response.headers.get("Location")
        response.close()
        if not location:
            raise requests.HTTPError(
                "Attachment redirect did not include a Location header"
            )
        if redirect_count >= max_redirects:
            raise requests.TooManyRedirects(
                f"Attachment exceeded {max_redirects} redirects"
            )
        current_url = urljoin(current_url, location)
        redirect_count += 1


def getAttachmentURLs(metadata):
    """Return every downloadable attachment URL from a comment response."""
    urls = []
    for attachment in metadata.get("included", []):
        if attachment.get("type") != "attachments":
            continue

        for file_format in attachment.get("attributes", {}).get("fileFormats") or []:
            file_url = file_format.get("fileUrl")
            if file_url:
                urls.append(file_url)
    return urls


def downloadAttachments(
    metadata,
    downloadRoot="downloads",
    *,
    session: requests.Session | None = None,
):
    """Download all API attachments into a directory named after the comment ID."""
    comment_id = metadata.get("data", {}).get("id")
    if not isinstance(comment_id, str) or not comment_id.strip():
        raise ValueError("Comment metadata does not include a comment ID")
    comment_id = comment_id.strip()
    download_root = Path(downloadRoot).resolve()
    artifact_dir = (download_root / comment_id).resolve()
    try:
        artifact_dir.relative_to(download_root)
    except ValueError as exc:
        raise ValueError("Comment ID produced an unsafe artifact path") from exc
    if artifact_dir == download_root or Path(comment_id).name != comment_id:
        raise ValueError("Comment ID produced an unsafe artifact path")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    attachments = []
    client = session or buildAttachmentSession()
    owns_session = session is None

    try:
        used_destinations = set()
        for attachment in metadata.get("included", []):
            if attachment.get("type") != "attachments":
                continue

            attachment_id = attachment.get("id", "attachment")
            for file_index, file_format in enumerate(
                attachment.get("attributes", {}).get("fileFormats") or [],
                start=1,
            ):
                file_url = file_format.get("fileUrl")
                if not file_url:
                    continue

                parsed_url = urlparse(file_url)
                filename = (
                    Path(parsed_url.path).name
                    or f"{attachment_id}-{file_index}"
                )
                destination = artifact_dir / filename
                if destination in used_destinations:
                    destination = (
                        artifact_dir
                        / f"{attachment_id}-{file_index}-{filename}"
                    )
                used_destinations.add(destination)

                record = {
                    "attachment_id": attachment_id,
                    "url": file_url,
                    "request_host": (
                        parsed_url.hostname.casefold()
                        if parsed_url.hostname
                        else None
                    ),
                    "format": file_format.get("format", ""),
                    "size": file_format.get("size", ""),
                }
                _, initially_authenticated = _attachment_request_headers(
                    file_url
                )
                record["authenticated_request"] = initially_authenticated
                raw_expected_size = file_format.get("size")
                try:
                    expected_size = int(raw_expected_size)
                except (TypeError, ValueError):
                    expected_size = None
                if (
                    destination.exists()
                    and destination.is_file()
                    and destination.stat().st_size > 0
                    and (
                        expected_size is None
                        or expected_size <= 0
                        or destination.stat().st_size == expected_size
                    )
                ):
                    record["path"] = str(destination)
                    record["download_status"] = "reused"
                    record["download_reused"] = True
                    record["downloaded_bytes"] = destination.stat().st_size
                    record["content_sha256"] = sha256_file(destination)
                    attachments.append(record)
                    continue

                temporary_destination = destination.with_name(
                    destination.name + ".part"
                )
                response: requests.Response | None = None
                try:
                    response, request_details = _open_attachment_response(
                        client,
                        file_url,
                        timeout=(
                            ATTACHMENT_CONNECT_TIMEOUT_SECONDS,
                            ATTACHMENT_READ_TIMEOUT_SECONDS,
                        ),
                    )
                    record.update(request_details)
                    record["http_status"] = response.status_code
                    content_type = response.headers.get("Content-Type", "")
                    record["content_type"] = content_type
                    response.raise_for_status()
                    if (
                        content_type.casefold().startswith("text/html")
                        and destination.suffix.casefold()
                        not in {".html", ".htm"}
                    ):
                        raise OSError(
                            "Attachment endpoint returned HTML instead of "
                            "the requested file"
                        )

                    downloaded_bytes = 0
                    with temporary_destination.open("wb") as stream:
                        for chunk in response.iter_content(
                            chunk_size=ATTACHMENT_CHUNK_SIZE
                        ):
                            if not chunk:
                                continue
                            stream.write(chunk)
                            downloaded_bytes += len(chunk)
                    if downloaded_bytes == 0:
                        raise OSError("Attachment response was empty")
                    if (
                        expected_size is not None
                        and expected_size > 0
                        and downloaded_bytes != expected_size
                    ):
                        raise OSError(
                            "Attachment size mismatch: expected "
                            f"{expected_size} bytes, received "
                            f"{downloaded_bytes}"
                        )

                    temporary_destination.replace(destination)
                    record["path"] = str(destination)
                    record["download_status"] = "downloaded"
                    record["downloaded_bytes"] = downloaded_bytes
                    record["content_sha256"] = sha256_file(destination)
                    logger.info(
                        "Attachment downloaded from %s",
                        _safe_download_url(file_url),
                        extra={
                            "event": "attachment_downloaded",
                            "comment_id": comment_id,
                            "attachment_id": attachment_id,
                            "request_host": record.get("request_host"),
                            "response_host": record.get("response_host"),
                            "http_status": record.get("http_status"),
                            "download_attempts": record.get(
                                "download_attempts",
                                1,
                            ),
                            "redirect_count": record.get(
                                "redirect_count",
                                0,
                            ),
                            "downloaded_bytes": downloaded_bytes,
                        },
                    )
                except (requests.RequestException, OSError, ValueError) as exc:
                    logger.warning(
                        "Unable to download attachment from %s: %s",
                        _safe_download_url(file_url),
                        exc,
                        extra={
                            "event": "attachment_download_failed",
                            "comment_id": comment_id,
                            "attachment_id": attachment_id,
                            "request_host": record.get("request_host"),
                            "response_host": record.get("response_host"),
                            "http_status": record.get("http_status"),
                            "download_attempts": record.get(
                                "download_attempts",
                                None,
                            ),
                            "redirect_count": record.get(
                                "redirect_count",
                                0,
                            ),
                            "error_type": type(exc).__name__,
                        },
                    )
                    record["download_status"] = "failed"
                    record["download_error_type"] = type(exc).__name__
                    record["download_error"] = str(exc)
                finally:
                    if response is not None:
                        response.close()
                    if temporary_destination.exists():
                        try:
                            temporary_destination.unlink()
                        except OSError:
                            logger.warning(
                                "Unable to remove partial attachment %s",
                                temporary_destination,
                            )

                attachments.append(record)
    finally:
        if owns_session:
            client.close()

    return artifact_dir, attachments


def extractPdfText(pdfPath):
    """Extract native text from a PDF and retain page markers for evidence tracing."""
    from src.ocr import extract_pdf_text_native

    extraction = extract_pdf_text_native(pdfPath)
    return extraction["text"], extraction["page_count"]


def needsOCR(text, pageCount):
    """Return whether sparse native text indicates that OCR is needed."""
    from src.ocr import needs_ocr

    return needs_ocr(text, pageCount)


def extractPdfTextWithOCR(pdfPath):
    """Render PDF pages and extract text with Tesseract only when native text is sparse."""
    return extract_pdf_text(pdfPath)["text"]


def extractAttachmentText(filePath):
    """Extract attachment text, using OCR only for PDFs with insufficient native text."""
    path = Path(filePath)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(path)

    if suffix == ".txt":
        return {
            "text": path.read_text(encoding="utf-8", errors="replace"),
            "page_count": 0,
            "ocr_used": False,
            "ocr_pages": [],
            "ocr_errors": [],
        }

    if suffix == ".docx":
        import docx

        document = docx.Document(path)
        return {
            "text": "\n".join(paragraph.text for paragraph in document.paragraphs),
            "page_count": 0,
            "ocr_used": False,
            "ocr_pages": [],
            "ocr_errors": [],
        }

    logger.warning("Skipping unsupported attachment format: %s", path)
    return {
        "text": "",
        "page_count": 0,
        "ocr_used": False,
        "ocr_pages": [],
        "ocr_errors": [],
        "unsupported_format": True,
    }


def cleanText(text):
    """Remove low-value repeated formatting while preserving page markers and evidence."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)

    page_sections = PAGE_MARKER.split(text)
    repeated_lines = set()
    if len(page_sections) > 2:
        counts = {}
        for section in page_sections[1:]:
            section_lines = [
                re.sub(r"\s+", " ", line).strip().casefold()
                for line in section.splitlines()
                if line.strip()
            ]
            boundary_lines = section_lines[:3] + section_lines[-3:]
            unique_lines = set(boundary_lines)
            for line in unique_lines:
                counts[line] = counts.get(line, 0) + 1

        threshold = max(2, math.ceil((len(page_sections) - 1) * 0.6))
        repeated_lines = {
            line
            for line, count in counts.items()
            if count >= threshold and not PAGE_NUMBER.fullmatch(line)
        }

    cleaned_lines = []
    previous_line = None
    retained_repeated_lines = set()
    for line in text.splitlines():
        normalized = re.sub(r"[ \t]+", " ", line).strip()
        normalized_key = normalized.casefold()

        if PAGE_MARKER.fullmatch(normalized):
            cleaned_lines.append(normalized)
            previous_line = None
            continue
        if not normalized or PAGE_NUMBER.fullmatch(normalized):
            continue
        if CONFIDENTIALITY_NOTICE.search(normalized):
            continue
        if normalized_key in repeated_lines:
            if normalized_key in retained_repeated_lines:
                continue
            retained_repeated_lines.add(normalized_key)
        if normalized_key == previous_line:
            continue

        cleaned_lines.append(normalized)
        previous_line = normalized_key

    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


def chunkText(text, chunkSize=12000, overlap=1000):
    """Split text into overlapping chunks without discarding page-marker context."""
    if chunkSize <= 0:
        raise ValueError("chunkSize must be greater than zero")
    if overlap < 0 or overlap >= chunkSize:
        raise ValueError("overlap must be zero or greater and less than chunkSize")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunkSize, len(text))
        chunk = text[start:end]
        if start:
            preceding_markers = list(PAGE_MARKER.finditer(text, 0, start))
            if preceding_markers:
                active_marker = preceding_markers[-1].group(0)
                if not chunk.startswith(active_marker):
                    chunk = f"{active_marker}\n{chunk}"
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def saveJSON(path, value):
    atomic_write_json(path, value)


def prepareCommentText(metadata, downloadRoot="downloads"):
    """Download attachments and persist raw and cleaned text artifacts for one comment."""
    artifact_dir, attachments = downloadAttachments(metadata, downloadRoot)
    previous_attachments = []
    previous_attachments_path = artifact_dir / "attachments.json"
    if previous_attachments_path.is_file():
        try:
            previous_value = json.loads(
                previous_attachments_path.read_text(encoding="utf-8")
            )
            if isinstance(previous_value, list):
                previous_attachments = previous_value
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Ignoring unreadable attachment cache at %s",
                previous_attachments_path,
            )
    previous_by_url = {
        item.get("url"): item
        for item in previous_attachments
        if isinstance(item, dict) and item.get("url")
    }
    saveJSON(artifact_dir / "metadata.json", metadata)

    attributes = metadata.get("data", {}).get("attributes", {})
    raw_sections = []
    cleaned_sections = []
    comment_text = attributes.get("comment") or ""
    if comment_text:
        raw_sections.append(f"--- COMMENT BODY ---\n{comment_text}")
        cleaned_comment_text = normalizeCommentHTML(comment_text)
        cleaned_sections.append(
            f"--- COMMENT BODY ---\n{cleanText(cleaned_comment_text)}"
        )

    for attachment in attachments:
        if not attachment.get("path"):
            continue

        text_path = Path(attachment["path"] + ".txt")
        raw_text_path = Path(attachment["path"] + ".raw.txt")
        previous = previous_by_url.get(attachment.get("url"), {})
        current_raw_hash = (
            sha256_file(raw_text_path) if raw_text_path.is_file() else None
        )
        current_cleaned_hash = (
            sha256_file(text_path) if text_path.is_file() else None
        )
        reusable_extraction = (
            previous.get("content_sha256")
            and previous.get("content_sha256") == attachment.get("content_sha256")
            and raw_text_path.is_file()
            and text_path.is_file()
            and previous.get("raw_text_sha256") == current_raw_hash
            and previous.get("cleaned_text_sha256") == current_cleaned_hash
        )
        if reusable_extraction:
            raw_text = raw_text_path.read_text(encoding="utf-8", errors="replace")
            cleaned_text = text_path.read_text(encoding="utf-8", errors="replace")
            extraction = {
                "page_count": previous.get("page_count", 0),
                "ocr_used": previous.get("ocr_used", False),
                "ocr_pages": previous.get("ocr_pages", []),
                "ocr_errors": previous.get("ocr_errors", []),
                "unsupported_format": previous.get("unsupported_format", False),
            }
            attachment["extraction_reused"] = True
        else:
            try:
                extraction = extractAttachmentText(attachment["path"])
            except Exception as exc:
                logger.exception("Unable to extract attachment %s", attachment["path"])
                attachment["extraction_error"] = str(exc)
                continue

            raw_text = extraction["text"]
            cleaned_text = cleanText(raw_text)
            atomic_write_text(raw_text_path, raw_text)
            atomic_write_text(text_path, cleaned_text)
            current_raw_hash = sha256_file(raw_text_path)
            current_cleaned_hash = sha256_file(text_path)

        attachment.update(
            {
                "page_count": extraction["page_count"],
                "ocr_used": extraction["ocr_used"],
                "ocr_pages": extraction.get("ocr_pages", []),
                "ocr_errors": extraction.get("ocr_errors", []),
                "unsupported_format": extraction.get("unsupported_format", False),
                "raw_text_path": str(raw_text_path),
                "text_path": str(text_path),
                "raw_text_sha256": current_raw_hash,
                "cleaned_text_sha256": current_cleaned_hash,
            }
        )
        if raw_text:
            filename = Path(attachment["path"]).name
            raw_sections.append(f"--- ATTACHMENT {filename} ---\n{raw_text}")
            cleaned_sections.append(f"--- ATTACHMENT {filename} ---\n{cleaned_text}")

    raw_text = "\n\n".join(raw_sections).strip()
    cleaned_text = "\n\n".join(cleaned_sections).strip()
    atomic_write_text(artifact_dir / "raw_text.txt", raw_text)
    atomic_write_text(artifact_dir / "cleaned_text.txt", cleaned_text)
    saveJSON(artifact_dir / "attachments.json", attachments)

    return {
        "artifact_dir": str(artifact_dir),
        "attachments": attachments,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
    }
