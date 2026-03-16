from dotenv import load_dotenv
import os

import requests as req
import logging
import time

load_dotenv()

BASE_URL = "https://api.regulations.gov/v4"
FR_BASE_URL = "https://www.federalregister.gov/api/v1/documents/"


def loadAPIKey():
    api_key = os.getenv("REGULATION_API_KEY")
    if api_key is None:
        raise ValueError("REGULATION_API_KEY not found in environment variables")
    return api_key


# attempt 1
def getDocketID(fr_doc_id: str, api_key: str):
    try:
        res = req.get(
            "https://api.regulations.gov/v4/documents?filter[frDocNum]="
            + fr_doc_id
            + "&&api_key="
            + api_key
        )
    except Exception as e:
        logging.error("Failed to fetch FR document %s: %s", fr_doc_id, e)
        return []

    return [
        {
            "title": i.get("attributes").get("title"),
            "id": i.get("id"),
            "objectId": i.get("attributes").get("objectId"),
            "method": "getDocketID",
        }
        for i in res.json()["data"]
    ]


def getDocumentId(docket_id: str, api_key: str):
    try:
        res = req.get(
            "https://api.regulations.gov/v4/documents?filter[docketId]="
            + docket_id
            + "&&api_key="
            + api_key
        )
    except Exception as e:
        logging.error("Failed to fetch FR document %s: %s", docket_id, e)
        return []

    return [
        {
            "title": i.get("attributes").get("title"),
            "id": i.get("id"),
            "objectId": i.get("attributes").get("objectId"),
            "method": "getDocumentId",
        }
        for i in res.json()["data"]
    ]


def getFRData(fr_doc_id: str):
    try:
        res = req.get(FR_BASE_URL + fr_doc_id)
    except Exception as e:
        logging.error("Failed to fetch FR document %s: %s", fr_doc_id, e)
        return []

    return res.json()


# attempt 2
def searchDocuments(query: str, api_key):
    url = f"{BASE_URL}/documents"
    params = {"filter[searchTerm]": query, "api_key": api_key}

    try:
        res = req.get(url, params=params).json()
    except Exception as e:
        logging.error("Failed to fetch dockets for query %s: %s", query, e)
        return []

    print(res)

    return [
        {
            "title": i.get("attributes").get("title"),
            "id": i.get("id"),
            "objectId": i.get("attributes").get("objectId"),
            "method": "searchDocuments",
        }
        for i in res["data"]
        if i.get("id")
    ]


def fetchComments(docket_id: str, api_key: str, metrics_tracker=None):
    # attempt 1: direct frDocNum lookup
    dockets = getDocketID(docket_id, api_key)
    docketInfo = getFRData(docket_id)

    # attempt 2: use docket IDs from the FR API
    if len(dockets) == 0:
        # try both fields and pick whichever gives more results
        nested_ids = [
            i.get("docket_id")
            for i in docketInfo.get("dockets", [])
            if i.get("docket_id")
        ]
        flat_ids = [i for i in docketInfo.get("docket_ids", []) if i]

        all_docket_ids = nested_ids if len(nested_ids) >= len(flat_ids) else flat_ids

        for did in all_docket_ids:
            results = getDocumentId(did, api_key)
            if len(results) > 0:
                dockets = results
                break

    # attempt 3: search by FR document number (NOT title — much more precise)
    if len(dockets) == 0:
        dockets = searchDocuments(docket_id, api_key)

    if len(dockets) == 0:
        if metrics_tracker:
            metrics_tracker.increment("failed_dockets")
        logging.error("Failed to fetch dockets for docket_id %s", docket_id)
        return

    logging.info(
        "Found %s dockets for docket_id %s. Sleeping 3 seconds to avoid rate limiting.",
        len(dockets),
        docket_id,
    )
    time.sleep(3)

    for docket in dockets:
        print(docket)
