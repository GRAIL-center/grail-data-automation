from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import gspread as gs

logger = logging.getLogger(__name__)


def setupGoogleSheets(
    spreadsheet_url: str,
) -> tuple[gs.Spreadsheet, gs.Worksheet]:
    if not spreadsheet_url.strip():
        raise ValueError("spreadsheet_url is required")

    client = gs.service_account(filename="service_account.json")
    spreadsheet = client.open_by_url(spreadsheet_url)
    return spreadsheet, spreadsheet.sheet1


def getRowCount(worksheet: gs.Worksheet) -> int:
    values = worksheet.col_values(1)

    if not values:
        logger.warning("No values found in worksheet %s", worksheet.title)
        return 0

    return sum(bool(str(value).strip()) for value in values[1:])


def getExistingFRIDs(worksheet: gs.Worksheet) -> set[str]:
    """Return non-empty values below the header in column A.

    Despite the legacy function name, this also works for comment IDs when
    column A contains "Comment ID".
    """
    values = worksheet.col_values(1)

    if len(values) <= 1:
        return set()

    return {str(value).strip() for value in values[1:] if str(value).strip()}


def addRow(worksheet: gs.Worksheet, values: Sequence[Any]) -> None:
    worksheet.append_row(
        list(values),
        value_input_option=gs.utils.ValueInputOption.user_entered,
        table_range="A1",
    )


def addRows(
    worksheet: gs.Worksheet,
    rows: Sequence[Sequence[Any]],
) -> None:
    if not rows:
        return

    worksheet.append_rows(
        [list(row) for row in rows],
        value_input_option=gs.utils.ValueInputOption.user_entered,
        table_range="A1",
    )


def createTab(
    spreadsheet: gs.Spreadsheet,
    tab_name: str,
) -> gs.Worksheet:
    tab_name = tab_name.strip()

    if not tab_name:
        raise ValueError("tab_name is required")

    return spreadsheet.add_worksheet(
        title=tab_name,
        rows=1000,
        cols=30,
    )


def getTab(
    spreadsheet: gs.Spreadsheet,
    tab_name: str,
) -> gs.Worksheet:
    tab_name = tab_name.strip()

    if not tab_name:
        raise ValueError("tab_name is required")

    try:
        return spreadsheet.worksheet(tab_name)
    except gs.WorksheetNotFound:
        logger.info("Creating worksheet %s", tab_name)
        return createTab(spreadsheet, tab_name)


def ensureHeaders(
    worksheet: gs.Worksheet,
    headers: Sequence[str],
) -> None:
    expected = list(headers)
    current = worksheet.row_values(1)

    if not current:
        addRow(worksheet, expected)
        return

    if current != expected:
        raise RuntimeError(
            f"Tab {worksheet.title!r} has the wrong headers. "
            f"Expected {expected}, found {current}."
        )
