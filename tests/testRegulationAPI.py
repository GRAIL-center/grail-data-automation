import requests as req
import os
from dotenv import load_dotenv

load_dotenv()

res = req.get(
    "https://api.regulations.gov/v4/documents?filter[frDocNum]=2021-05281&&api_key="
    + os.getenv("REGULATION_API_KEY")
)
print(res.json()["data"])
print(
    [
        {"title": i.get("attributes").get("title"), "id": i.get("id")}
        for i in res.json()["data"]
    ]
)
