"""Fetch a Regulations.gov comment and its complete body text.

Public function:
    getCommentText(comment_id: str, fr_number: str) -> str

For each comment, this file creates:

    data/<fr_number>/<comment_id>/
        metadata.json
        comment_body.txt
        full_comment.txt
        manifest.json
        attachments/
        extracted/

The file is standalone. It does not import any project modules.

Required environment variable:
    REGULATIONS_API_KEY

Required packages:
    requests
    pymupdf
    pillow
    pytesseract
    python-docx

Optional packages:
    striprtf       Better RTF extraction
    openpyxl       XLSX extraction
    python-pptx    PPTX extraction
"""

from __future__ import annotations

import html
import io
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import time
import uuid
from email.message import Message
from html.parser import HTMLParser
from pathlib import Path
from plistlib import load
from typing import Any
from urllib.parse import unquote, urlparse

import docx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.services.config import loadRegKey

REGULATIONS_API_URL = "https://api.regulations.gov/v4"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

TIMEOUT = 60
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
OCR_DPI = 300
MINIMUM_CHARACTERS_PER_PAGE = 100
OCR_ALL_PDF_PAGES = True
TESSERACT_CONFIG = "--oem 3 --psm 3"

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def buildSession(api_key: str | None = None) -> requests.Session:
    """Create an HTTP session with basic retry handling."""
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "application/pdf,application/octet-stream;q=0.8,*/*;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    )

    if api_key:
        session.headers.update(
            {
                "X-Api-Key": api_key,
                "Accept": "application/json",
            }
        )

    return session


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
        raise TypeError(f"Expected a JSON object from {url}.")

    return payload


def fetchCommentPayload(
    session: requests.Session,
    comment_id: str,
) -> JsonDict:
    """Fetch one comment and ask Regulations.gov to include attachments."""
    payload = requestJSON(
        session,
        f"{REGULATIONS_API_URL}/comments/{comment_id}",
        {"include": "attachments"},
    )

    if not isinstance(payload.get("data"), dict):
        raise LookupError(f"Comment not found: {comment_id}")

    return payload


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def sanitizeFilename(filename: str) -> str:
    """Return a Windows-safe filename."""
    filename = Path(unquote(filename)).name
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    filename = filename.strip(" .")

    if not filename:
        return "attachment"

    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }

    if Path(filename).stem.upper() in reserved:
        filename = f"_{filename}"

    return filename[:180]


def uniquePath(folder: Path, filename: str) -> Path:
    path = folder / filename

    if not path.exists():
        return path

    counter = 2

    while True:
        candidate = folder / f"{path.stem}_{counter}{path.suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


def writeJSON(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def firstNonempty(
    mapping: dict[str, Any],
    *keys: str,
) -> str | None:
    for key in keys:
        value = mapping.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return None


def validURL(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    value = value.strip()
    parsed = urlparse(value)

    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return value

    return None


# ---------------------------------------------------------------------------
# Attachment discovery and download
# ---------------------------------------------------------------------------


def findAttachmentURLs(value: Any) -> list[str]:
    """Recursively find likely file URLs inside attachment metadata."""
    urls: list[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower()

            looks_like_file_url = normalized_key in {
                "fileurl",
                "downloadurl",
                "attachmenturl",
                "contenturl",
                "href",
                "url",
            } or (
                "url" in normalized_key
                and any(
                    word in normalized_key
                    for word in (
                        "file",
                        "download",
                        "attachment",
                        "content",
                    )
                )
            )

            if looks_like_file_url:
                url = validURL(child)

                if url:
                    urls.append(url)
                    continue

            urls.extend(findAttachmentURLs(child))

    elif isinstance(value, list):
        for item in value:
            urls.extend(findAttachmentURLs(item))

    return urls


def extractAttachmentCandidates(
    payload: JsonDict,
) -> list[dict[str, str | None]]:
    """Normalize included JSON:API attachment records."""
    included = payload.get("included", [])

    if not isinstance(included, list):
        return []

    candidates: list[dict[str, str | None]] = []
    seen_urls: set[str] = set()

    for attachment_number, resource in enumerate(included, start=1):
        if not isinstance(resource, dict):
            continue

        resource_type = str(resource.get("type") or "").lower()

        if "attachment" not in resource_type:
            continue

        attributes = resource.get("attributes")

        if not isinstance(attributes, dict):
            continue

        attachment_id = str(resource.get("id")) if resource.get("id") else None

        attachment_name = firstNonempty(
            attributes,
            "fileName",
            "filename",
            "title",
            "name",
            "label",
        )

        file_formats = attributes.get("fileFormats")

        if isinstance(file_formats, dict):
            format_records = [file_formats]
        elif isinstance(file_formats, list):
            format_records = [item for item in file_formats if isinstance(item, dict)]
        else:
            format_records = []

        records = format_records or [attributes]

        for record_number, record in enumerate(records, start=1):
            filename = (
                firstNonempty(
                    record,
                    "fileName",
                    "filename",
                    "name",
                    "title",
                )
                or attachment_name
            )

            file_format = firstNonempty(
                record,
                "format",
                "fileType",
                "mimeType",
                "contentType",
                "type",
            )

            for url in findAttachmentURLs(record):
                if url in seen_urls:
                    continue

                seen_urls.add(url)

                candidates.append(
                    {
                        "attachment_id": attachment_id,
                        "url": url,
                        "filename": filename,
                        "format": file_format,
                        "label": (
                            attachment_name
                            or f"attachment_{attachment_number}_{record_number}"
                        ),
                    }
                )

    return candidates


def extensionFromFormat(value: str | None) -> str:
    if not value:
        return ""

    normalized = value.lower().split(";", maxsplit=1)[0].strip()

    known = {
        "pdf": ".pdf",
        "application/pdf": ".pdf",
        "doc": ".doc",
        "application/msword": ".doc",
        "docx": ".docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "txt": ".txt",
        "text": ".txt",
        "text/plain": ".txt",
        "html": ".html",
        "text/html": ".html",
        "rtf": ".rtf",
        "application/rtf": ".rtf",
        "csv": ".csv",
        "text/csv": ".csv",
        "json": ".json",
        "application/json": ".json",
        "xml": ".xml",
        "application/xml": ".xml",
        "png": ".png",
        "image/png": ".png",
        "jpg": ".jpg",
        "jpeg": ".jpg",
        "image/jpeg": ".jpg",
        "tif": ".tif",
        "tiff": ".tif",
        "image/tiff": ".tif",
        "xlsx": ".xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "pptx": ".pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    }

    if normalized in known:
        return known[normalized]

    guessed = mimetypes.guess_extension(normalized) or ""

    return ".jpg" if guessed == ".jpe" else guessed


def filenameFromResponse(
    response: requests.Response,
    candidate: dict[str, str | None],
    attachment_number: int,
) -> str:
    content_disposition = response.headers.get("Content-Disposition")

    if content_disposition:
        message = Message()
        message["content-disposition"] = content_disposition
        filename = message.get_filename()

        if filename:
            return sanitizeFilename(filename)

    filename = candidate.get("filename")

    if not filename:
        filename = Path(urlparse(str(candidate.get("url") or "")).path).name

    filename = sanitizeFilename(filename)

    if not Path(filename).suffix:
        content_type = (
            response.headers.get(
                "Content-Type",
                "",
            )
            .split(";", maxsplit=1)[0]
            .strip()
        )

        extension = extensionFromFormat(candidate.get("format")) or extensionFromFormat(
            content_type
        )

        filename = f"{filename}{extension}"

    if filename == "attachment":
        filename = f"attachment_{attachment_number}"

    return filename


def downloadAttachment(
    session: requests.Session,
    candidate: dict[str, str | None],
    folder: Path,
    attachment_number: int,
) -> Path:
    """Download one attachment using browser-like request headers."""
    url = candidate.get("url")

    if not url:
        raise ValueError("Attachment has no download URL.")

    parsed_url = urlparse(url)
    is_regulations_download = parsed_url.hostname == "downloads.regulations.gov"

    request_headers = {
        "Referer": "https://www.regulations.gov/",
        "Accept": (
            "application/pdf,application/octet-stream,"
            "application/msword,"
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document,*/*;q=0.8"
        ),
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": ("same-site" if is_regulations_download else "cross-site"),
    }

    def makeRequest() -> requests.Response:
        return session.get(
            url,
            headers=request_headers,
            stream=True,
            allow_redirects=True,
            timeout=TIMEOUT,
        )

    response = makeRequest()

    if response.status_code == 403 and is_regulations_download:
        response.close()

        try:
            session.get(
                "https://www.regulations.gov/",
                headers={
                    "Referer": "https://www.regulations.gov/",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                },
                timeout=TIMEOUT,
            )
        except requests.RequestException:
            pass

        request_headers["Cache-Control"] = "no-cache"
        request_headers["Pragma"] = "no-cache"
        response = makeRequest()

    try:
        response.raise_for_status()

        destination = uniquePath(
            folder,
            filenameFromResponse(
                response,
                candidate,
                attachment_number,
            ),
        )

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachment exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit."
            )

        bytes_written = 0
        try:
            with destination.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=DOWNLOAD_CHUNK_SIZE,
                ):
                    if not chunk:
                        continue

                    bytes_written += len(chunk)
                    if bytes_written > MAX_ATTACHMENT_BYTES:
                        raise ValueError(
                            f"Attachment exceeds the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit."
                        )
                    output.write(chunk)
        except Exception:
            destination.unlink(missing_ok=True)
            raise

        return destination
    finally:
        response.close()


# ---------------------------------------------------------------------------
# PDF extraction with selective OCR
# ---------------------------------------------------------------------------


def pageMarker(page_number: int, text: str) -> str:
    return f"--- PAGE {page_number} ---\n{text.strip()}"


def needsOCR(
    text: str,
    page_count: int,
    minimum_characters_per_page: int = MINIMUM_CHARACTERS_PER_PAGE,
) -> bool:
    if page_count <= 0:
        return False

    normalized = "".join(text.split())

    return len(normalized) / page_count < minimum_characters_per_page


def configureTesseract() -> Any:
    """Locate Tesseract, including common Windows installation paths."""
    import pytesseract

    configured_path = os.getenv("TESSERACT_CMD", "").strip()

    candidates = [
        configured_path,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(
            Path(os.getenv("LOCALAPPDATA", ""))
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe"
        ),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or could not be found. "
            "Install Tesseract, add it to PATH, or set TESSERACT_CMD "
            "to the full tesseract.exe path."
        ) from exc

    return pytesseract


def ocrPDFPage(page: Any, dpi: int = OCR_DPI) -> str:
    """Render and OCR one PDF page."""
    from PIL import Image, ImageOps

    pytesseract = configureTesseract()

    pixmap = page.get_pixmap(
        dpi=dpi,
        alpha=False,
    )

    with Image.open(io.BytesIO(pixmap.tobytes("png"))) as source_image:
        image = ImageOps.grayscale(source_image)
        image = ImageOps.autocontrast(image)

        return pytesseract.image_to_string(
            image,
            config=TESSERACT_CONFIG,
        ).strip()


def extractPDFText(
    pdf_path: str | Path,
    minimum_characters_per_page: int = MINIMUM_CHARACTERS_PER_PAGE,
    dpi: int = OCR_DPI,
) -> dict[str, Any]:
    """Extract PDF text and OCR every page by default.

    OCR is attempted on every page when ``OCR_ALL_PDF_PAGES`` is true. The OCR
    result replaces native text when it contains at least as much meaningful
    text. This handles scanned PDFs and PDFs with broken or misleading hidden
    text layers without throwing away good native extraction.
    """
    import pymupdf

    pages: list[str] = []
    ocr_attempted_pages: list[int] = []
    ocr_pages: list[int] = []
    ocr_errors: list[dict[str, Any]] = []
    native_pages_used: list[int] = []

    with pymupdf.open(str(pdf_path)) as document:
        for page_number in range(len(document)):
            page = document[page_number]
            page_number += 1
            native_text = str(page.get_text("text")).strip()
            native_character_count = len("".join(native_text.split()))

            should_ocr = (
                OCR_ALL_PDF_PAGES
                or native_character_count < minimum_characters_per_page
            )

            selected_text = native_text

            if should_ocr:
                ocr_attempted_pages.append(page_number)

                try:
                    ocr_text = ocrPDFPage(page, dpi)
                    ocr_character_count = len("".join(ocr_text.split()))

                    # Prefer OCR for scanned/sparse pages. For digital PDFs,
                    # only replace native text when OCR recovered at least as
                    # much meaningful content.
                    if ocr_text and (
                        native_character_count < minimum_characters_per_page
                        or ocr_character_count >= native_character_count
                    ):
                        selected_text = ocr_text
                        ocr_pages.append(page_number)
                    else:
                        native_pages_used.append(page_number)

                except Exception as exc:
                    logger.warning(
                        "OCR failed for %s page %d: %s",
                        pdf_path,
                        page_number,
                        exc,
                    )
                    native_pages_used.append(page_number)
                    ocr_errors.append(
                        {
                            "page": page_number,
                            "error": str(exc),
                        }
                    )
            else:
                native_pages_used.append(page_number)

            pages.append(
                pageMarker(
                    page_number,
                    selected_text,
                )
            )

    return {
        "text": "\n\n".join(pages),
        "page_count": len(pages),
        "ocr_attempted": bool(ocr_attempted_pages),
        "ocr_attempted_pages": ocr_attempted_pages,
        "ocr_used": bool(ocr_pages),
        "ocr_pages": ocr_pages,
        "native_pages_used": native_pages_used,
        "ocr_errors": ocr_errors,
    }


# ---------------------------------------------------------------------------
# Other attachment extractors
# ---------------------------------------------------------------------------


class HTMLTextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
        "section",
        "table",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def getText(self) -> str:
        text = html.unescape("".join(self.parts))

        return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def extractDOCXText(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    sections: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            sections.append(text)

    for table_number, table in enumerate(
        document.tables,
        start=1,
    ):
        rows = [
            "\t".join(cell.text.strip() for cell in row.cells) for row in table.rows
        ]

        if rows:
            sections.append(f"--- TABLE {table_number} ---\n" + "\n".join(rows))

    return "\n\n".join(sections).strip()


def extractHTMLText(path: Path) -> str:
    parser = HTMLTextExtractor()
    parser.feed(
        path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    return parser.getText()


def extractRTFText(path: Path) -> str:
    raw = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    try:
        from striprtf.striprtf import rtf_to_text

        return rtf_to_text(raw).strip()
    except ImportError:
        raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
        raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", raw)
        raw = raw.replace("{", " ").replace("}", " ")

        return re.sub(r"\s+", " ", raw).strip()


def extractImageText(path: Path) -> str:
    import pytesseract
    from PIL import Image

    with Image.open(path) as image:
        return pytesseract.image_to_string(image).strip()


def extractLegacyDOCText(path: Path) -> str:
    """Use antiword for old .doc files when it is installed."""
    if not shutil.which("antiword"):
        raise RuntimeError("Legacy .doc extraction requires the antiword program.")

    result = subprocess.run(
        ["antiword", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    return result.stdout.strip()


def extractXLSXText(path: Path) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("XLSX extraction requires openpyxl.") from exc

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True,
    )
    sections: list[str] = []

    try:
        for worksheet in workbook.worksheets:
            rows: list[str] = []

            for row in worksheet.iter_rows(values_only=True):
                values = ["" if value is None else str(value) for value in row]

                if any(value.strip() for value in values):
                    rows.append("\t".join(values))

            if rows:
                sections.append(f"--- SHEET: {worksheet.title} ---\n" + "\n".join(rows))
    finally:
        workbook.close()

    return "\n\n".join(sections).strip()


def extractPPTXText(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise RuntimeError("PPTX extraction requires python-pptx.") from exc

    presentation = Presentation(str(path))
    slides: list[str] = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):
        parts: list[str] = []

        for shape in slide.shapes:
            text = getattr(shape, "text", "")

            if text and str(text).strip():
                parts.append(str(text).strip())

        slides.append(f"--- SLIDE {slide_number} ---\n" + "\n".join(parts))

    return "\n\n".join(slides).strip()


def extractAttachmentText(path: Path) -> dict[str, Any]:
    """Extract text from one downloaded attachment."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extractPDFText(path)

    if suffix == ".docx":
        return {"text": extractDOCXText(path)}

    if suffix == ".doc":
        return {"text": extractLegacyDOCText(path)}

    if suffix in {
        ".txt",
        ".md",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
    }:
        return {
            "text": path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
        }

    if suffix in {".html", ".htm"}:
        return {"text": extractHTMLText(path)}

    if suffix == ".rtf":
        return {"text": extractRTFText(path)}

    if suffix in {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }:
        return {
            "text": extractImageText(path),
            "ocr_used": True,
        }

    if suffix == ".xlsx":
        return {"text": extractXLSXText(path)}

    if suffix == ".pptx":
        return {"text": extractPPTXText(path)}

    raise ValueError(f"Unsupported attachment type: {suffix or '[no extension]'}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def textSection(title: str, text: str) -> str:
    return f"===== {title} =====\n{text.strip()}"


def commentFolder(fr_number: str, comment_id: str) -> Path:
    safe_fr_number = sanitizeFilename(fr_number)
    safe_comment_id = sanitizeFilename(comment_id)

    if not safe_fr_number or not safe_comment_id:
        raise ValueError("Federal Register number and comment ID are required.")

    return DATA_ROOT / safe_fr_number / safe_comment_id


def loadCompletedCommentText(comment_folder: Path) -> str | None:
    manifest_path = comment_folder / "manifest.json"
    text_path = comment_folder / "full_comment.txt"

    if not manifest_path.is_file() or not text_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(manifest, dict) or not manifest.get("complete"):
        return None

    return text_path.read_text(encoding="utf-8")


def replaceCommentFolder(staging_folder: Path, comment_folder: Path) -> None:
    backup_folder = comment_folder.with_name(
        f".{comment_folder.name}.backup-{uuid.uuid4().hex}"
    )

    if comment_folder.exists():
        comment_folder.replace(backup_folder)

    retry_delays = (0.1, 0.2, 0.4, 0.8, 1.6)

    try:
        for attempt in range(len(retry_delays) + 1):
            try:
                staging_folder.replace(comment_folder)
                break
            except PermissionError:
                # Another run may have promoted the same comment first.
                if loadCompletedCommentText(comment_folder) is not None:
                    shutil.rmtree(staging_folder, ignore_errors=True)
                    break
                if attempt == len(retry_delays):
                    raise
                time.sleep(retry_delays[attempt])
    except Exception:
        if backup_folder.exists():
            backup_folder.replace(comment_folder)
        raise
    else:
        if backup_folder.exists():
            shutil.rmtree(backup_folder)


def getCommentText(comment_id: str, fr_number: str) -> str:
    """Download a comment into data/<FR number>/<comment ID> and return its text."""
    comment_id = comment_id.strip()
    fr_number = fr_number.strip()

    if not comment_id:
        raise ValueError("Comment ID cannot be empty.")
    if not fr_number:
        raise ValueError("Federal Register number cannot be empty.")

    comment_folder = commentFolder(fr_number, comment_id)
    logger.info("Preparing local artifact for comment %s under %s", comment_id, fr_number)
    cached_text = loadCompletedCommentText(comment_folder)
    if cached_text is not None:
        logger.info("Reusing downloaded comment %s from %s", comment_id, comment_folder)
        return cached_text

    comment_folder.parent.mkdir(parents=True, exist_ok=True)
    staging_folder = comment_folder.with_name(
        f".{comment_folder.name}.staging-{uuid.uuid4().hex}"
    )
    attachments_folder = staging_folder / "attachments"
    extracted_folder = staging_folder / "extracted"
    attachments_folder.mkdir(parents=True)
    extracted_folder.mkdir(parents=True)

    try:
        with buildSession(loadRegKey()) as api_session:
            payload = fetchCommentPayload(api_session, comment_id)

        writeJSON(staging_folder / "metadata.json", payload)

        attributes = payload["data"].get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        inline_text = html.unescape(str(attributes.get("comment") or "")).strip()
        (staging_folder / "comment_body.txt").write_text(
            inline_text,
            encoding="utf-8",
        )

        sections: list[str] = []
        if inline_text:
            sections.append(textSection("INLINE COMMENT", inline_text))

        attachment_candidates = extractAttachmentCandidates(payload)
        attachment_results: list[dict[str, Any]] = []
        logger.info(
            "Found %d attachment candidate(s) for comment %s",
            len(attachment_candidates),
            comment_id,
        )

        # Do not send the Regulations.gov API key to attachment hosts.
        with buildSession() as download_session:
            for number, candidate in enumerate(attachment_candidates, start=1):
                result: dict[str, Any] = {
                    "number": number,
                    "attachment_id": candidate.get("attachment_id"),
                    "label": candidate.get("label"),
                    "url": candidate.get("url"),
                    "format": candidate.get("format"),
                    "downloaded_file": None,
                    "extracted_text_file": None,
                    "download_error": None,
                    "extraction_error": None,
                    "extraction_details": {},
                }

                try:
                    downloaded_path = downloadAttachment(
                        download_session,
                        candidate,
                        attachments_folder,
                        number,
                    )
                    result["downloaded_file"] = str(
                        downloaded_path.relative_to(staging_folder)
                    )
                except Exception as exc:
                    result["download_error"] = str(exc)
                    attachment_results.append(result)
                    continue

                try:
                    extraction = extractAttachmentText(downloaded_path)
                    attachment_text = str(extraction.get("text") or "").strip()
                    extracted_path = uniquePath(
                        extracted_folder,
                        f"{downloaded_path.name}.txt",
                    )
                    extracted_path.write_text(attachment_text, encoding="utf-8")
                    result["extracted_text_file"] = str(
                        extracted_path.relative_to(staging_folder)
                    )
                    result["extraction_details"] = {
                        key: value for key, value in extraction.items() if key != "text"
                    }

                    if attachment_text:
                        sections.append(
                            textSection(
                                f"ATTACHMENT {number}: {downloaded_path.name}",
                                attachment_text,
                            )
                        )
                except Exception as exc:
                    result["extraction_error"] = str(exc)

                attachment_results.append(result)

        if not sections:
            sections.append(
                textSection(
                    "COMMENT",
                    "[No inline or extractable attachment text found.]",
                )
            )

        full_text = "\n\n".join(sections).strip()
        (staging_folder / "full_comment.txt").write_text(full_text, encoding="utf-8")
        writeJSON(
            staging_folder / "manifest.json",
            {
                "storage_version": 1,
                "complete": True,
                "fr_number": fr_number,
                "comment_id": comment_id,
                "inline_comment_present": bool(inline_text),
                "attachment_candidates_found": len(attachment_candidates),
                "attachments_downloaded": sum(
                    item["downloaded_file"] is not None for item in attachment_results
                ),
                "attachments_extracted": sum(
                    item["extracted_text_file"] is not None for item in attachment_results
                ),
                "attachments": attachment_results,
            },
        )
        replaceCommentFolder(staging_folder, comment_folder)
        logger.info(
            "Saved comment %s with %d downloaded attachment(s) to %s",
            comment_id,
            sum(item["downloaded_file"] is not None for item in attachment_results),
            comment_folder,
        )
        return full_text
    except Exception:
        shutil.rmtree(staging_folder, ignore_errors=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(getCommentText("FNS-2021-0038-0050", "2021-00000"))
