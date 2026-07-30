import logging
import re
from datetime import datetime

import requests as req
from bs4 import BeautifulSoup as bs

from src.services.ai_client import generate_text
from src.services.config import loadNoticeConfig, loadNoticeSheetUrl
from src.services.sheets import addRow, getExistingFRIDs, setupGoogleSheets

logger = logging.getLogger(__name__)
REQUEST_TIMEOUT = 30



def extractCommentDate(text: str) -> str | None:
    """
    Attempts to extract the comment closing date from text using Regex.
    Returns a formatted date string 'MM/DD/YYYY' or None if not found.
    """
    if not text:
        return None

    # This regex looks for phrases like "comments must be received on or before"
    # or "due by" followed by a Date (e.g., January 1, 2024 or Jan. 1, 2024)
    pattern = re.compile(
        r"(?:comments?.*?received|comments?.*?due).*?(?:on or before|by|no later than)\s+"
        r"([A-Z][a-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4})",
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(text)
    if match:
        date_str = match.group(1).replace(".", "")  # clean up "Jan." to "Jan"

        # Strip suffixes like 'st', 'nd', 'rd', 'th' (e.g., "January 1st" -> "January 1")
        date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)

        try:
            # Parse the matched string "January 1, 2024" into a datetime
            parsed_date = datetime.strptime(date_str, "%B %d, %Y")
            return parsed_date.strftime("%m/%d/%Y")  # return in standard format
        except ValueError:
            # Fallback for abbreviated months like "Jan 1, 2024"
            try:
                parsed_date = datetime.strptime(date_str, "%b %d, %Y")
                return parsed_date.strftime("%m/%d/%Y")
            except ValueError:
                return None

    return None


def getBodyInfo(docId: str):
    """
    Retrieves the agency name for a given docId.

    Parameters
    ----------
    docId : str
        The document number to look up.

    Returns
    -------
    tuple[str, str] or tuple[None, None]
        Body text and agency name as a title-cased string if found,
        or (None, None) if extraction fails.
    """
    url = f"https://www.federalregister.gov/api/v1/documents/{docId}.json?fields[]=body_html_url"
    res = req.get(url, timeout=REQUEST_TIMEOUT)
    res.raise_for_status()
    res = res.json()

    # retrieve soup
    bodyUrl = res.get("body_html_url")
    if not bodyUrl:
        logger.error("No body URL found for docId: %s", docId)
        return None, None

    bodyRes = req.get(bodyUrl, timeout=REQUEST_TIMEOUT)

    if bodyRes.status_code != 200:
        logger.error("Failed to retrieve body for docId: %s", docId)
        return None, None

    bodyText = bodyRes.text
    soup = bs(bodyText, "html.parser")

    agency = soup.find("div", id="agency")
    if agency:
        agency = (
            agency.text.strip()
            .lower()
            .replace("agency:", "")
            .strip()
            .replace(".", "")
            .title()
        )
        return agency, bodyText

    logger.error("No agency found for docId: %s", docId)
    return None, None


def formatDate(
    dateStr: str, inputFormat: str = "%Y-%m-%d", outputFormat: str = "%m/%d/%Y"
):
    """
    Formats a date string from one format to another.

    Parameters
    ----------
    dateStr : str
        The date string to format.
    inputFormat : str, optional
        The format of the input date string (default is "%Y-%m-%d").
    outputFormat : str, optional
        The desired output format for the date string (default is "%m/%d/%Y").

    Returns
    -------
    str or None
        The formatted date string, or None if the parsing fails.
    """
    try:
        return datetime.strptime(dateStr, inputFormat).strftime(outputFormat)
    except (ValueError, TypeError) as e:
        logger.error("Failed to format date: %s", e)
        return None


def processNotice(notice: dict):
    """
    Processes a single notice and returns a list of values for the sheet.

    Parameters
    ----------
    notice : dict
        A dictionary containing the notice metadata from the Federal Register API.

    Returns
    -------
    list
        A list of values formatted for a spreadsheet row, or None on failure.
    """
    docNum = notice.get("document_number")
    if not docNum:
        logger.error("No document number found for notice: %s", notice)
        return None

    commentUrl = f"https://www.federalregister.gov/api/v1/documents/{docNum}.json?fields[]=comments_close_on"
    commentsCloseOn = None
    try:
        commentData = req.get(commentUrl, timeout=REQUEST_TIMEOUT).json()
        commentsCloseOn = commentData.get("comments_close_on")
        if not commentsCloseOn:
            logger.warning("No comments close on found for docId: %s", docNum)
    except Exception as e:
        logger.error(
            "Failed to retrieve comments close on for docId: %s", docNum, exc_info=e
        )

    pubDate = formatDate(notice.get("publication_date", "")) or "N/A"
    if commentsCloseOn:
        commentsCloseOn = formatDate(commentsCloseOn)

    agency, bodyText = getBodyInfo(docNum)
    if not agency:
        logger.error("No agency found for docId: %s", docNum)
        return None

    textToSummarize = notice.get("abstract") or bodyText or ""
    summary = ""
    if textToSummarize:
        try:
            summary = generate_text(
                f"Summarize the following federal register notice in 2-3 sentences "
                f"focused on key regulatory actions and their impact:\n\n{textToSummarize[:4000]}"
            )
        except Exception as e:
            logger.error("Failed to summarize docId: %s", docNum, exc_info=e)
            summary = "Summary unavailable."

    commentEndDate = commentsCloseOn or extractCommentDate(bodyText or "") or "N/A"

    return [
        notice.get("title", ""),
        agency,
        pubDate,
        f"{pubDate} - {commentEndDate}",
        f"AI Summary:\n{summary}" if summary else "",
        notice.get("html_url", ""),
        "Needs comment end date." if commentEndDate == "N/A" else "",
        notice.get("type", ""),
        docNum,
    ]


def scrapeNotices(
    sheet,
    searchTerms: list[str],
    maxNotices: int = 5,
    docketType: str = "NOTICE",
    order: str = "relevance",
    startDate: str = "2021-01-01",
) -> None:
    existingIds = set(getExistingFRIDs(sheet))
    addedCount = 0
    rowsToAppend = []

    for term in searchTerms:
        if addedCount >= maxNotices:
            break

        logger.info("Searching Federal Register for term %r", term)
        url = (
            f"https://www.federalregister.gov/api/v1/documents.json?"
            f"per_page={maxNotices}&order={order}&conditions[term]={term}"
            f"&conditions[publication_date][gte]={startDate}&conditions[type]={docketType}"
        )

        try:
            res = req.get(url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            data = res.json()
            notices = data.get("results", [])
            logger.info("Federal Register returned %d notices for term %r", len(notices), term)

            relevantNotices = [
                n
                for n in notices
                if any(t in n.get("title", "").lower() for t in searchTerms)
                and n.get("document_number") not in existingIds
            ]

            for notice in relevantNotices:
                if addedCount >= maxNotices:
                    break

                rowData = processNotice(notice)
                if rowData:
                    rowsToAppend.append(rowData)
                    existingIds.add(notice.get("document_number"))
                    addedCount += 1
                    logger.info(
                        "Added notice: %s [%s]",
                        notice.get("title"),
                        notice.get("document_number"),
                    )

        except Exception as e:
            logger.error("Failed to retrieve notices for term: %s", term, exc_info=e)
            continue

    if rowsToAppend:
        for row in rowsToAppend:
            addRow(sheet, row)
        logger.info("Added %s notices to the sheet", len(rowsToAppend))


# main function
def collectNotices(user_settings=None, user_terms=None):
    """
    Collect Federal Register notices using configured or user-provided inputs.

    This function loads the default Google Sheet and notice configuration, then
    optionally replaces either configuration block with user-supplied values.

    Parameters
    ----------
    user_settings : dict | None, optional
        Full notice collection settings to use instead of the saved config.
        If provided, this dictionary must include all of the following keys:

        - ``max_notices``: int
        - ``docket_type``: str
        - ``order``: str
        - ``start_date``: str

        These values are passed directly into ``scrapeNotices``.
    user_terms : dict | None, optional
        Search-term configuration to use instead of the saved config. If
        provided, this dictionary should include both of the following keys:

        - ``default_terms``: list[str]
        - ``search_terms``: list[str]

        Both lists are flattened into a single search-term list before scraping.

    Notes
    -----
    If ``user_settings`` or ``user_terms`` is provided, that value completely
    replaces the corresponding loaded configuration. Missing required keys will
    raise a ``KeyError`` when the function tries to use them.
    """
    logger.info("Opening notice spreadsheet and loading configuration")
    _, SHEET = setupGoogleSheets(loadNoticeSheetUrl())
    SETTINGS, SEARCH_TERMS_DATA = loadNoticeConfig()

    if user_terms is not None:
        SEARCH_TERMS_DATA = user_terms

    if user_settings is not None:
        SETTINGS = user_settings

    # Combine default_terms and search_terms into a single flat list
    allTerms = []
    if SEARCH_TERMS_DATA:
        configured_terms = (
            (SEARCH_TERMS_DATA.get("default_terms", []) or [])
            + (SEARCH_TERMS_DATA.get("search_terms", []) or [])
        )
        allTerms.extend(
            str(term).strip().lower()
            for term in configured_terms
            if str(term).strip()
        )

    logger.info(
        "Starting notice collection with %d search terms; max_notices=%s, "
        "docket_type=%s, order=%s, start_date=%s",
        len(allTerms),
        SETTINGS["max_notices"],
        SETTINGS["docket_type"],
        SETTINGS["order"],
        SETTINGS["start_date"],
    )

    scrapeNotices(
        SHEET,
        allTerms,
        SETTINGS["max_notices"],
        SETTINGS["docket_type"],
        SETTINGS["order"],
        SETTINGS["start_date"],
    )


# testing
if __name__ == "__main__":
    collectNotices()
