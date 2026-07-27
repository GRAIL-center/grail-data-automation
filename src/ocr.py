"""Document text extraction with selective PDF OCR.

The public functions in this module return page-marked text so downstream AI
evidence can be traced back to the source document.  The legacy ``extractFile``
entry point is retained for callers that only need the extracted text.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def needs_ocr(
    text: str,
    page_count: int,
    minimum_characters_per_page: int = 100,
) -> bool:
    """Return whether a PDF has too little native text to be trustworthy."""
    if page_count <= 0:
        return False

    normalized = "".join(text.split())
    return len(normalized) / page_count < minimum_characters_per_page


def _page_marker(page_number: int, text: str) -> str:
    return f"--- PAGE {page_number} ---\n{text.strip()}"


def extract_pdf_text_native(pdf_path: str | Path) -> dict[str, Any]:
    """Extract native PDF text without performing OCR."""
    import fitz

    pages: list[str] = []
    page_texts: list[str] = []
    with fitz.open(str(pdf_path)) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            page_texts.append(text)
            pages.append(_page_marker(page_number, text))

        return {
            "text": "\n\n".join(pages),
            "page_texts": page_texts,
            "page_count": len(document),
        }


def _ocr_page(page: Any, dpi: int) -> str:
    import pytesseract
    from PIL import Image

    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image).strip()


def extract_pdf_text(
    pdf_path: str | Path,
    minimum_characters_per_page: int = 100,
    dpi: int = 300,
) -> dict[str, Any]:
    """Extract a PDF, applying OCR only to pages whose native text is sparse.

    A document-level sparsity check catches fully scanned PDFs.  For mixed PDFs,
    an individual sparse page is OCRed when it contains a raster image.  OCR
    failures are recorded while the native text remains available.
    """
    import fitz

    native = extract_pdf_text_native(pdf_path)
    native_texts = native["page_texts"]
    document_needs_ocr = needs_ocr(
        native["text"],
        native["page_count"],
        minimum_characters_per_page,
    )

    pages: list[str] = []
    ocr_pages: list[int] = []
    ocr_errors: list[dict[str, Any]] = []

    with fitz.open(str(pdf_path)) as document:
        for page_number, (page, native_text) in enumerate(
            zip(document, native_texts),
            start=1,
        ):
            native_character_count = len("".join(native_text.split()))
            has_raster_image = bool(page.get_images(full=True))
            page_needs_ocr = (
                native_character_count < minimum_characters_per_page
                and (document_needs_ocr or has_raster_image)
            )
            selected_text = native_text

            if page_needs_ocr:
                try:
                    ocr_text = _ocr_page(page, dpi)
                    if ocr_text:
                        selected_text = ocr_text
                    ocr_pages.append(page_number)
                except Exception as exc:  # Tesseract availability varies by host.
                    logger.warning(
                        "OCR failed for %s page %d: %s",
                        pdf_path,
                        page_number,
                        exc,
                    )
                    ocr_errors.append(
                        {
                            "page": page_number,
                            "error": str(exc),
                        }
                    )

            pages.append(_page_marker(page_number, selected_text))

    return {
        "text": "\n\n".join(pages),
        "page_count": native["page_count"],
        "ocr_used": bool(ocr_pages),
        "ocr_pages": ocr_pages,
        "ocr_errors": ocr_errors,
    }


def extract_text_from_pdf(pdf_path: str) -> str:
    """Compatibility wrapper returning only PDF text."""
    return extract_pdf_text(pdf_path)["text"].strip()


def extract_text_from_docx(docx_path: str) -> str:
    import docx

    document = docx.Document(docx_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()


def extract_text_from_txt(txt_path: str) -> str:
    return Path(txt_path).read_text(encoding="utf-8", errors="replace").strip()


def extractFile(file_path: str) -> str:
    """Legacy dispatcher kept for the existing pipeline API."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported file type: {file_path}")
