from pipeline.comment_collection.collect import searchDocuments, getFRData
from dotenv import load_dotenv
import os

load_dotenv()


def testSearchDockets():
    print(searchDocuments("2023-18624", os.getenv("REGULATION_API_KEY")))
    print(getFRData("2023-18624"))


if __name__ == "__main__":
    testSearchDockets()
