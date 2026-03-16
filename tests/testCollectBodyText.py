from pipeline.notice_collection.collect import getAgency
import re
from datetime import datetime


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


agency, body = getAgency("2023-07776")
print(extractCommentDate(""))
