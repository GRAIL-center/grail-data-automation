import logging

import gspread as gs

logger = logging.getLogger(__name__)


def setupGoogleSheets(spreadsheet_url: str):
    if not spreadsheet_url:
        raise ValueError("spreadsheet_url is required")

    client = gs.service_account(filename="service_account.json")
    spreadsheet = client.open_by_url(spreadsheet_url)

    return spreadsheet, spreadsheet.sheet1


def getRowCount(worksheet: gs.Worksheet) -> int:
    values = worksheet.col_values(1)

    if not values:
        logger.warning("No values found in the sheet")
        return 0

    return sum(bool(value.strip()) for value in values[1:])


def getExistingFRIDs(worksheet: gs.Worksheet) -> set[str]:
    values = worksheet.col_values(1)

    if len(values) <= 1:
        logger.warning("No FR IDs found in the sheet")
        return set()

    return {value.strip() for value in values[1:] if value.strip()}


def addRow(worksheet: gs.Worksheet, values: list) -> None:
    worksheet.append_row(list(values), value_input_option="USER_ENTERED")


# def createTab(spreadsheet: gs.Spreadsheet, tab_name: str) -> gs.Worksheet:
#     tab_name = tab_name.strip()

#     if not tab_name:
#         raise ValueError("tab_name is required")

#     return spreadsheet.add_worksheet(title=tab_name, rows=1000, cols=20)


def getTab(spreadsheet: gs.Spreadsheet, tab_name: str) -> gs.Worksheet:
    tab_name = tab_name.strip()

    if not tab_name:
        raise ValueError("tab_name is required")

    try:
        return spreadsheet.worksheet(tab_name)
    except gs.WorksheetNotFound:
        return createTab(spreadsheet, tab_name)


# testing
# if __name__ == "__main__":
#     logger.info("Testing sheets module")
#     notice_ss = setupGoogleSheets(
#         "https://docs.google.com/spreadsheets/d/1G1YjFpOYcAnBcHTZq5ySl956VeT6fJRmrSlFiVESCec/edit?gid=0#gid=0"
#     )
#     comment_ss = setupGoogleSheets(
#         "https://docs.google.com/spreadsheets/d/1n7l_velbqHHLpHI_V0NYwU8m7KeDGIShvdbxJ4sX5Co/edit?gid=0#gid=0"
#     )

#     logger.info(getRowCount(notice_ss))
#     logger.info(getExistingFRIDs(notice_ss))

#     new_ws = createTab(comment_ss, "testing")
#     addRow(new_ws, [1, 2, 3, 4, 5])
