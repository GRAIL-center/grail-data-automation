import requests as req
import gspread as gs
import logging
import yaml
import os

from oauth2client.service_account import ServiceAccountCredentials
from services.ollama_client import OllamaClient
from pipeline.metrics_tracker import MetricsTracker
from bs4 import BeautifulSoup as bs
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()
logging.basicConfig(level=logging.INFO)


def setupGoogleSheets():
    """
    Sets up the Google Sheets client and returns the sheet object.

    Returns
    -------
    gspread.models.Worksheet
        The initialized Google Sheet object.
    """

    SPREADSHEET_URL = os.getenv("NOTICE_SPREADSHEET_URL")
    CREDS = ServiceAccountCredentials.from_json_keyfile_name(
        "service_account.json",
        [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    GCLIENT = gs.authorize(CREDS)
    SHEET = GCLIENT.open_by_url(SPREADSHEET_URL).sheet1
    return SHEET


def getRowCount(sheet):
    """
    Returns the number of rows in the sheet.

    Parameters
    ----------
    sheet : gspread.models.Worksheet
        The Google Sheet object to count rows for.

    Returns
    -------
    int
        The number of non-empty rows in the first column, excluding the header.
    """
    return (
        sum(1 for cell in sheet.col_values(1) if cell.strip() != "") - 1
    )  # subtract header


def getExistingDocIds(sheet):
    """
    Returns a list of existing doc ids from the sheet.

    Parameters
    ----------
    sheet : gspread.models.Worksheet
        The Google Sheet object containing notice data.

    Returns
    -------
    list of str
        A list of document IDs found in column 9 of the sheet.
    """
    # Fetch entire column 9 in one API call instead of one request per row
    all_values = sheet.col_values(9)  # includes header at index 0
    if len(all_values) <= 1:
        logging.info(
            "No existing doc IDs found while attempting to retrieve list of doc ids."
        )
        return []

    ids = all_values[1:]  # skip header row
    return [doc_id.strip() for doc_id in ids if doc_id]


def extractCommentDate(text: str) -> str:
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


def loadSearchTerms(config_path: str = "config/search_terms.yaml"):
    """
    Loads search terms from a YAML config file.

    Parameters
    ----------
    config_path : str, optional
        The path to the YAML configuration file (default is "config/search_terms.yaml").

    Returns
    -------
    dict or None
        A dictionary containing "default_terms" and "search_terms" if successful,
        or None if the config file does not exist.
    """
    if not os.path.exists(config_path):
        logging.error("Config file not found: %s", config_path)
        return None

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    default_terms = data.get("notice_search_terms").get("default_terms")
    search_terms = data.get("notice_search_terms").get("search_terms")

    return {"default_terms": default_terms, "search_terms": search_terms}


def loadSettings(config_path: str = "config/experiment_settings.yaml"):
    """
    Loads experiment settings from a YAML config file.

    Parameters
    ----------
    config_path : str, optional
        The path to the YAML configuration file (default is "config/experiment_settings.yaml").

    Returns
    -------
    dict or None
        A dictionary containing experiment settings if successful,
        or None if the config file does not exist.
    """
    if not os.path.exists(config_path):
        logging.error("Config file not found: %s", config_path)
        return None

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    data = data.get("notice_collection_settings")
    maxNotices = data.get("maxNotices")
    docketType = data.get("docketType")
    order = data.get("order")
    startDate = data.get("startDate")

    return {
        "maxNotices": maxNotices,
        "docketType": docketType,
        "order": order,
        "startDate": startDate,
    }


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
    res = req.get(url).json()

    # retrieve soup
    bodyUrl = res.get("body_html_url")
    if not bodyUrl:
        logging.error("No body URL found for docId: %s", docId)
        return None, None

    bodyRes = req.get(bodyUrl)

    if bodyRes.status_code != 200:
        logging.error("Failed to retrieve body for docId: %s", docId)
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

    logging.error("No agency found for docId: %s", docId)
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
        logging.error("Failed to format date: %s", e)
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
        A list of values formatted for a spreadsheet row:
        [formatted date, agency, title, url, document number].
    """
    docNum = notice.get("document_number")
    if not docNum:
        logging.error("No document number found for notice: %s", notice)
        return None

    commentUrl = f"https://www.federalregister.gov/api/v1/documents/{docNum}.json?fields[]=comments_close_on"
    commentsCloseOn = None
    try:
        commentData = req.get(commentUrl).json()
        commentsCloseOn = commentData.get("comments_close_on")
        if not commentsCloseOn:
            logging.warning("No comments close on found for docId: %s", docNum)
    except Exception as e:
        logging.error("Failed to retrieve comments close on for docId: %s", docNum)

    pubDate = formatDate(notice.get("publication_date", ""))
    if commentsCloseOn:
        commentsCloseOn = formatDate(commentsCloseOn)

    agency, bodyText = getBodyInfo(docNum)
    if not agency:
        logging.error("No agency found for docId: %s", docNum)
        return None

    textToSummarize = notice.get("abstract") or bodyText
    summary = OllamaClient().summarize(
        textToSummarize,
        "This is part of a federal register notice and should be analyzed as such.",
    )

    commentEndDate = commentsCloseOn or extractCommentDate(bodyText) or "N/A"

    docId = notice.get("document_number")

    # info that will be added to the sheet
    return [
        notice.get("title", ""),
        agency,
        pubDate,
        f"{pubDate} - {commentEndDate}",
        f"AI Summary:\n{summary}",
        notice.get("html_url", ""),
        "Needs comment end date." if commentEndDate == "N/A" else "",
        notice.get("type", ""),  # docket type
        docId,
    ]


def scrapeNotices(
    sheet,
    searchTerms: list[str],
    maxNotices: int = 5,
    docketType: str = "NOTICE",
    order: str = "relevance",
    startDate: str = "2021-01-01",
    metrics_tracker: MetricsTracker = None,
) -> None:
    """
    Scrapes notices from the Federal Register based on search terms.

    Parameters
    ----------
    sheet : gspread.models.Worksheet
        The Google Sheet object to append new notices to.
    searchTerms : list of str
        A list of search terms to filter the Federal Register notices.
    maxNotices : int, optional
        The maximum number of notices to scrape (default is 5).
    docketType : str, optional
        The type of document to search for (default is "NOTICE").
    order : str, optional
        The order in which to retrieve results (default is "relevance").
    startDate : str, optional
        The earliest date of publication to consider, formatted as "YYYY-MM-DD"
        (default is "2021-01-01").
    metrics_tracker : MetricsTracker
        The metrics tracker to log events to.

    Returns
    -------
    None
    """
    existingIds = set(getExistingDocIds(sheet))
    addedCount = 0
    rowsToAppend = []

    for term in searchTerms:
        if addedCount >= maxNotices:
            break

        url = (
            f"https://www.federalregister.gov/api/v1/documents.json?"
            f"per_page={maxNotices}&order={order}&conditions[term]={term}"
            f"&conditions[publication_date][gte]={startDate}&conditions[type]={docketType}"
        )

        try:
            res = req.get(url)
            res.raise_for_status()
            data = res.json()
            notices = data.get("results", [])

            relevantNotices = [
                n
                for n in notices
                if any(term in n.get("title", "").lower() for term in searchTerms)
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
                    logging.info(
                        "Added notice: %s [%s]",
                        notice.get("title"),
                        notice.get("document_number"),
                    )

                    if metrics_tracker:
                        metrics_tracker.log_event("notice_added", count=addedCount)
        except Exception as e:
            if metrics_tracker:
                metrics_tracker.log_event("notice_added", error=str(e))
            logging.error("Failed to retrieve notices for term: %s", term)
            continue

        if res.status_code != 200:
            if metrics_tracker:
                metrics_tracker.log_event(
                    "notice_added", error=f"Failed to retrieve notices for term: {term}"
                )
            logging.error("Failed to retrieve notices for term: %s", term)
            continue

    if rowsToAppend:
        sheet.append_rows(rowsToAppend)
        logging.info("Added %s notices to the sheet", len(rowsToAppend))


# main function
def collectNotices(metrics_tracker: MetricsTracker = None):
    """
    Collects notices from the Federal Register based on search terms.

    Parameters
    ----------
    metrics_tracker : MetricsTracker
        The metrics tracker to log events to.

    Returns
    -------
    None
    """

    SHEET = setupGoogleSheets()
    SETTINGS = loadSettings()
    SEARCH_TERMS_DATA = loadSearchTerms()

    # Combine default_terms and search_terms into a single flat list
    allTerms = []
    if SEARCH_TERMS_DATA:
        allTerms.extend(SEARCH_TERMS_DATA.get("default_terms", []) or [])
        allTerms.extend(SEARCH_TERMS_DATA.get("search_terms", []) or [])

    logging.info("Starting notice collection with %d search terms", len(allTerms))

    scrapeNotices(
        SHEET,
        allTerms,
        SETTINGS["maxNotices"],
        SETTINGS["docketType"],
        SETTINGS["order"],
        SETTINGS["startDate"],
        metrics_tracker,
    )
