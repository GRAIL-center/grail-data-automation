"""Reliable Google Sheets workbook and worksheet export helpers."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

import gspread as gs
from gspread.utils import rowcol_to_a1
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

GOOGLE_SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
MAX_WORKSHEET_TITLE_LENGTH = 100
INVALID_WORKSHEET_TITLE = re.compile(r"[\x00-\x1f\x7f:/\\?*\[\]]+")
_EXPORT_LOCK = threading.RLock()


class GoogleSheetsExportError(RuntimeError):
    """An actionable Google Sheets authentication or export failure."""


def sanitizeWorksheetTitle(
    value: str,
    *,
    fallback: str = "Sheet",
    maxLength: int = MAX_WORKSHEET_TITLE_LENGTH,
) -> str:
    """Return a deterministic Google-Sheets-safe worksheet title."""
    if isinstance(maxLength, bool) or not isinstance(maxLength, int) or maxLength <= 0:
        raise ValueError("maxLength must be a positive integer")
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    normalized = INVALID_WORKSHEET_TITLE.sub(" - ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*-\s*(?:-\s*)+", " - ", normalized)
    normalized = normalized.strip(" -'")
    if not normalized:
        normalized = re.sub(r"\s+", " ", fallback).strip(" '") or "Sheet"
    return normalized[:maxLength].rstrip(" '")


def buildCommentWorksheetTitle(
    frDocumentNumber: str,
    noticeTitle: str | None,
    *,
    template: str = "{fr_document_number} - {notice_title}",
) -> str:
    """Build a stable per-notice comment-tab title with the FR ID preserved."""
    fr_number = str(frDocumentNumber or "").strip()
    if not fr_number:
        raise ValueError("frDocumentNumber cannot be empty")
    if (
        not isinstance(template, str)
        or not template.lstrip().startswith("{fr_document_number}")
    ):
        raise ValueError(
            "Comment worksheet template must start with "
            "{fr_document_number} so notice tabs remain stable"
        )
    title = str(noticeTitle or "").strip() or "Comments"
    try:
        rendered = template.format(
            fr_document_number=fr_number,
            notice_title=title,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid comment worksheet template: {exc}") from exc
    rendered = rendered.strip()
    if not rendered.casefold().startswith(fr_number.casefold()):
        raise ValueError(
            "Comment worksheet template must render with the Federal Register "
            "document number first"
        )
    suffix = rendered[len(fr_number) :].lstrip(" \t-–—:|")
    return sanitizeWorksheetTitle(
        f"{fr_number} - {suffix or title}",
    )


def openGoogleSpreadsheet(
    spreadsheetUrl: str,
    *,
    credentialsPath: str = "service_account.json",
):
    """Authenticate and return a gspread Spreadsheet with actionable errors."""
    if not isinstance(spreadsheetUrl, str) or not spreadsheetUrl.strip():
        raise ValueError("Google spreadsheet URL cannot be empty")
    credentials_path = Path(credentialsPath)
    if not credentials_path.is_file():
        raise FileNotFoundError(
            "Google service-account file was not found: "
            f"{credentials_path.resolve()}. Set application.service_account_file "
            "to the downloaded JSON credential."
        )
    try:
        credentials = Credentials.from_service_account_file(
            str(credentials_path),
            scopes=[GOOGLE_SHEETS_SCOPE],
        )
    except Exception as exc:
        raise GoogleSheetsExportError(
            f"Unable to load Google service-account credentials from "
            f"{credentials_path}: {exc}"
        ) from exc

    service_account_email = getattr(
        credentials,
        "service_account_email",
        "",
    )
    try:
        client = gs.authorize(credentials)
        return client.open_by_url(spreadsheetUrl.strip())
    except gs.exceptions.NoValidUrlKeyFound as exc:
        raise ValueError(
            "The Google Sheet URL is invalid; copy the full spreadsheet URL "
            "from the browser."
        ) from exc
    except gs.exceptions.SpreadsheetNotFound as exc:
        account_hint = (
            f" Share it with {service_account_email}."
            if service_account_email
            else " Share it with the configured service-account email."
        )
        raise GoogleSheetsExportError(
            "The spreadsheet was not found or the service account cannot access it."
            + account_hint
        ) from exc
    except Exception as exc:
        raise GoogleSheetsExportError(
            f"Unable to open the Google spreadsheet: {exc}"
        ) from exc


def _worksheet_with_prefix(spreadsheet, prefix: str):
    normalized_prefix = prefix.casefold()
    for worksheet in spreadsheet.worksheets():
        title = str(getattr(worksheet, "title", ""))
        folded = title.casefold()
        if folded == normalized_prefix or folded.startswith(
            f"{normalized_prefix} -"
        ):
            return worksheet
    return None


def selectGoogleWorksheet(
    spreadsheet,
    *,
    worksheetTitle: str | None = None,
    create: bool = False,
    reusePrefix: str | None = None,
    rows: int = 1000,
    cols: int = 40,
):
    """Select the first tab or safely create/reuse an explicitly named tab."""
    if worksheetTitle is None:
        return spreadsheet.sheet1
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError("rows must be a positive integer")
    if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
        raise ValueError("cols must be a positive integer")
    title = sanitizeWorksheetTitle(worksheetTitle)

    try:
        return spreadsheet.worksheet(title)
    except gs.exceptions.WorksheetNotFound:
        pass

    if reusePrefix:
        existing = _worksheet_with_prefix(
            spreadsheet,
            sanitizeWorksheetTitle(reusePrefix),
        )
        if existing is not None:
            return existing
    if not create:
        raise gs.exceptions.WorksheetNotFound(title)

    try:
        return spreadsheet.add_worksheet(
            title=title,
            rows=rows,
            cols=cols,
        )
    except gs.exceptions.APIError:
        # Another worker/process may have created it between lookup and add.
        try:
            return spreadsheet.worksheet(title)
        except gs.exceptions.WorksheetNotFound:
            if reusePrefix:
                existing = _worksheet_with_prefix(
                    spreadsheet,
                    sanitizeWorksheetTitle(reusePrefix),
                )
                if existing is not None:
                    return existing
            raise


def setupGoogleSheets(
    spreadsheet_url,
    credentialsPath="service_account.json",
    *,
    worksheetTitle: str | None = None,
    createWorksheet: bool = False,
    reusePrefix: str | None = None,
    rows: int = 1000,
    cols: int = 40,
):
    """Compatibility helper returning a selected worksheet."""
    spreadsheet = openGoogleSpreadsheet(
        spreadsheet_url,
        credentialsPath=credentialsPath,
    )
    return selectGoogleWorksheet(
        spreadsheet,
        worksheetTitle=worksheetTitle,
        create=createWorksheet,
        reusePrefix=reusePrefix,
        rows=rows,
        cols=cols,
    )


def getRowCount(sheet):
    return max(
        0,
        sum(1 for cell in sheet.col_values(1) if str(cell).strip() != "") - 1,
    )


def getExistingDocketIDs(sheet):
    """Return Docket ID values without relying on a fragile column number."""
    headers = [str(header).strip() for header in sheet.row_values(1)]
    try:
        column = headers.index("Docket ID") + 1
    except ValueError:
        return []
    values = sheet.col_values(column)
    return [str(value).strip() for value in values[1:] if str(value).strip()]


def _ordered_required_headers(
    rows: Iterable[Mapping[str, Any]],
    canonicalHeaders: Iterable[str] | None,
) -> list[str]:
    required: list[str] = []
    seen: set[str] = set()
    for header in canonicalHeaders or []:
        normalized = str(header).strip()
        if not normalized:
            raise ValueError("Canonical worksheet headers cannot be empty")
        if normalized in seen:
            raise ValueError(f"Duplicate canonical worksheet header: {normalized}")
        seen.add(normalized)
        required.append(normalized)
    for row in rows:
        for field in row:
            normalized = str(field).strip()
            if not normalized:
                raise ValueError("Worksheet row fields cannot be empty")
            if normalized not in seen:
                seen.add(normalized)
                required.append(normalized)
    return required


def _appendMappingRowsDetailed(
    sheet,
    rows,
    *,
    strictHeaders=False,
    batchSize=500,
    canonicalHeaders: Iterable[str] | None = None,
    deduplicateBy: str | None = None,
    upsertBy: str | None = None,
) -> dict[str, Any]:
    materialized_rows = list(rows or [])
    if not all(isinstance(row, Mapping) for row in materialized_rows):
        raise TypeError("Each worksheet row must be a mapping")
    if isinstance(batchSize, bool) or not isinstance(batchSize, int) or batchSize <= 0:
        raise ValueError("batchSize must be a positive integer")
    if deduplicateBy is not None and upsertBy is not None:
        raise ValueError("deduplicateBy and upsertBy cannot both be configured")

    required_headers = _ordered_required_headers(
        materialized_rows,
        canonicalHeaders,
    )
    headers = [str(header).strip() for header in sheet.row_values(1)]
    if not any(headers):
        if not required_headers:
            return {
                "appended": 0,
                "updated": 0,
                "skipped": 0,
                "headers": [],
                "appended_keys": [],
                "updated_keys": [],
                "skipped_keys": [],
            }
        headers = required_headers
        sheet.append_row(headers)

    missing_headers = [
        header
        for header in required_headers
        if header not in headers
    ]
    if missing_headers:
        if strictHeaders:
            raise ValueError(
                "Worksheet is missing required column headers: "
                + ", ".join(missing_headers)
            )
        headers.extend(missing_headers)
        sheet.update(range_name="1:1", values=[headers])
        logger.info(
            "Added %d analysis column(s) to worksheet %s",
            len(missing_headers),
            getattr(sheet, "title", ""),
            extra={
                "event": "sheet_headers_extended",
                "worksheet_title": getattr(sheet, "title", ""),
                "added_headers": missing_headers,
            },
        )

    rows_to_append = materialized_rows
    rows_to_update: list[tuple[int, Mapping[str, Any]]] = []
    appended_keys: list[str] = []
    updated_keys: list[str] = []
    skipped_keys: list[str] = []
    key_field = upsertBy or deduplicateBy
    if key_field is not None and rows_to_append:
        if key_field not in headers:
            raise ValueError(
                f"Key column is missing from worksheet: {key_field}"
            )
        key_column = headers.index(key_field) + 1
        existing_rows: dict[str, list[int]] = {}
        for row_number, value in enumerate(
            sheet.col_values(key_column)[1:],
            start=2,
        ):
            normalized = str(value).strip().casefold()
            if normalized:
                existing_rows.setdefault(normalized, []).append(row_number)
        filtered_rows = []
        input_keys: set[str] = set()
        for row in rows_to_append:
            raw_key = row.get(key_field, "")
            display_key = str(raw_key).strip()
            normalized_key = display_key.casefold()
            if normalized_key and normalized_key in input_keys:
                skipped_keys.append(display_key)
                continue
            if normalized_key:
                input_keys.add(normalized_key)
            matching_rows = existing_rows.get(normalized_key, [])
            if normalized_key and matching_rows and upsertBy is not None:
                rows_to_update.extend(
                    (row_number, row)
                    for row_number in matching_rows
                )
                updated_keys.append(display_key)
                continue
            if normalized_key and matching_rows:
                skipped_keys.append(display_key)
                continue
            filtered_rows.append(row)
            if normalized_key:
                existing_rows[normalized_key] = []
                appended_keys.append(display_key)
        rows_to_append = filtered_rows

    update_payloads: list[dict[str, Any]] = []
    for row_number, row in rows_to_update:
        segment_start = None
        segment_values: list[Any] = []
        for column_number, header in enumerate(headers, start=1):
            if header in row:
                if segment_start is None:
                    segment_start = column_number
                segment_values.append(row[header])
                continue
            if segment_start is not None:
                update_payloads.append(
                    {
                        "range": (
                            f"{rowcol_to_a1(row_number, segment_start)}:"
                            f"{rowcol_to_a1(row_number, column_number - 1)}"
                        ),
                        "values": [segment_values],
                    }
                )
                segment_start = None
                segment_values = []
        if segment_start is not None:
            update_payloads.append(
                {
                    "range": (
                        f"{rowcol_to_a1(row_number, segment_start)}:"
                        f"{rowcol_to_a1(row_number, len(headers))}"
                    ),
                    "values": [segment_values],
                }
            )
    for start in range(0, len(update_payloads), batchSize):
        sheet.batch_update(update_payloads[start : start + batchSize])

    values = [
        [row.get(header, "") for header in headers]
        for row in rows_to_append
    ]
    for start in range(0, len(values), batchSize):
        sheet.append_rows(values[start : start + batchSize])
    logger.info(
        "Updated %d and appended %d row(s) to worksheet %s; "
        "skipped %d row(s)",
        len(rows_to_update),
        len(values),
        getattr(sheet, "title", ""),
        len(skipped_keys),
        extra={
            "event": "sheet_rows_written",
            "worksheet_title": getattr(sheet, "title", ""),
            "updated_count": len(rows_to_update),
            "appended_count": len(values),
            "skipped_count": len(skipped_keys),
        },
    )
    return {
        "appended": len(values),
        "updated": len(rows_to_update),
        "skipped": len(skipped_keys),
        "headers": headers,
        "appended_keys": appended_keys,
        "updated_keys": updated_keys,
        "skipped_keys": skipped_keys,
    }


def appendMappingRows(
    sheet,
    rows,
    *,
    strictHeaders=False,
    batchSize=500,
    canonicalHeaders: Iterable[str] | None = None,
    deduplicateBy: str | None = None,
    upsertBy: str | None = None,
):
    """Write mapping records in physical header order and return appended count."""
    return _appendMappingRowsDetailed(
        sheet,
        rows,
        strictHeaders=strictHeaders,
        batchSize=batchSize,
        canonicalHeaders=canonicalHeaders,
        deduplicateBy=deduplicateBy,
        upsertBy=upsertBy,
    )["appended"]


def exportMappingRows(
    spreadsheetUrl: str,
    rows,
    *,
    credentialsPath: str = "service_account.json",
    worksheetTitle: str | None = None,
    createWorksheet: bool = False,
    reusePrefix: str | None = None,
    worksheetRows: int = 1000,
    worksheetCols: int = 40,
    canonicalHeaders: Iterable[str] | None = None,
    deduplicateBy: str | None = None,
    upsertBy: str | None = None,
    batchSize: int = 500,
) -> dict[str, Any]:
    """Select/create a tab and append, deduplicate, or upsert mapping rows."""
    with _EXPORT_LOCK:
        worksheet = setupGoogleSheets(
            spreadsheetUrl,
            credentialsPath=credentialsPath,
            worksheetTitle=worksheetTitle,
            createWorksheet=createWorksheet,
            reusePrefix=reusePrefix,
            rows=worksheetRows,
            cols=worksheetCols,
        )
        details = _appendMappingRowsDetailed(
            worksheet,
            rows,
            batchSize=batchSize,
            canonicalHeaders=canonicalHeaders,
            deduplicateBy=deduplicateBy,
            upsertBy=upsertBy,
        )
        details["worksheet_title"] = str(
            getattr(worksheet, "title", worksheetTitle or ""),
        )
        details["worksheet_id"] = getattr(worksheet, "id", None)
        return details
