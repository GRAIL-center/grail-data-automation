import os

import yaml
from dotenv import load_dotenv

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config.yaml",
)

load_dotenv()


def loadConfig() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def loadNoticeConfig() -> tuple:
    data = loadConfig()
    return data.get("notice_collection_settings"), data.get("notice_search_terms")


def loadAISettings() -> dict:
    return loadConfig().get("ai", {})


def loadRegKey() -> str:
    key = os.getenv("REGULATION_API_KEY")

    if not key:
        raise ValueError("REGULATION_API_KEY not set")

    return key


def loadNoticeSheetUrl() -> str:
    url = os.getenv("NOTICE_SHEET_URL")

    if not url:
        raise ValueError("NOTICE_SHEET_URL not set")

    return url


def loadCommentSheetUrl() -> str:
    url = os.getenv("COMMENT_SHEET_URL")

    if not url:
        raise ValueError("COMMENT_SHEET_URL not set")

    return url
