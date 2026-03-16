from pipeline.comment_collection.collect import fetchComments
import os
from dotenv import load_dotenv

load_dotenv()

links = [
    "2021-05281",
    "2022-23255",
    "2022-20000",
    "2024-09645",
    "2024-15377",
    "2021-19287",
    "2023-25128",
    "2023-20480",
    "2023-18624",
    "2024-09132",
]
for i in links:
    comment = fetchComments(i, os.getenv("REGULATION_API_KEY"))
