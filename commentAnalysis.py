from PIL import Image
import os
import json
import fitz  # PyMuPDF
from ollama import Client
import pytesseract
import io
import gspread as gs
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from commentManager import CommentManager
from logger_config import Logger

import dotenv

from queryFile import QueryFile
dotenv.load_dotenv()


class Environment:
    def __init__(self, ollama_key, url, ollama_model="llama3.2:latest", creds_file="service_account.json", sheetNum=1):
        self.sheetNum = sheetNum - 1
        self.url = url
        self.OLLAMA_MODEL = ollama_model  # ✅ Cloud model to use
        self.FIELDS = [
            "COMMENT-ID",
            "Filename",
            "Date Submitted",
            "Submitter Name",
            "Organization Name",
            "Organization Type",
            "501c Status",
            "Organization Type_NTEE",
            "Organization Role",
            "Contact Information",
            "Relevant Keywords",
            "Brief Summary of Comment",
            "Relevant Issues Addressed",
        ]
        self.cred_file = creds_file
        self.ollama_key = ollama_key

        self.setupLogger("env")
        self.setupOllama()
        self.setupGoogleSheet()

    def setupLogger(self, docNum):
        log_folder = f"./logs/analysis/{docNum}"
        self.logger = Logger(log_folder=log_folder)

    def setupOllama(self):
        try:
            self.ollama = Client(
                host="https://ollama.com",
                headers={
                    "Authorization": self.ollama_key,
                    "Content-Type": "application/json"
                }
            )
        except Exception as e:
            self.logger.log(f"Error setting up Ollama: {e}", level="ERROR")
            raise e

    def setupGoogleSheet(self):
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.cred_file, [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ])
            client = gs.authorize(creds)
            self.spreadsheet = client.open_by_url(self.url)
        except Exception as e:
            self.logger.log(f"Error setting up Google Sheet: {e}", level="ERROR")
            raise e

    def chat(self, prompt):
        try:
            response = self.ollama.chat(model=self.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            raw_output = response["message"]["content"].strip()
            self.logger.log(f"Response: {raw_output}", level="INFO")
            return raw_output
        except Exception as e:
            self.logger.log(f"Error in chat: {e}", level="ERROR")
            raise e

    def addSheetRow(self, rowData):
        sheet = self.spreadsheet.get_worksheet(self.sheetNum)
        if sheet is None:
            print("❌ No worksheet found at index 0")
            return
        rowData = [str(x).strip() if x else "N/A" for x in rowData]
        sheet.append_row(rowData, value_input_option='USER_ENTERED')
        print("✅ Row added successfully.")


class CommentAnalysis:
    def __init__(self, env: Environment, docId: str):
        self.env = env
        self.docId = docId
        self.env.setupLogger(docId)
        self.logger = self.env.logger
        self.commentPath = f"./comments/{self.docId}"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    text += page_text + "\n"
                else:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img)
                    text += ocr_text + "\n"
            doc.close()
        except Exception as e:
            self.logger.log(f"Error extracting {pdf_path}: {e}", level="ERROR")
        return text.strip()

    @staticmethod
    def getBase(file):
        """
        Returns the base filename without attachment suffix or extension
        """
        base = os.path.splitext(file)[0]
        if "__attachment" in base:
            base = base.rsplit("__attachment", 1)[0]
        return base

    @staticmethod
    def organizeDuplicates(folder):
        """
        Groups files by base name (attachments included)
        """
        groups = {}
        for f in folder:
            base = CommentAnalysis.getBase(f)
            groups.setdefault(base, []).append(f)
        return list(groups.values())

    def getOrganizedFolder(self):
        files = [f for f in os.listdir(self.commentPath) if f.lower().endswith(".pdf")]
        return self.organizeDuplicates(files)

    def getMetadata(self, filename):
        base = os.path.splitext(filename)[0]
        metadataFile = f"{self.commentPath}/metadata.json"
        with open(metadataFile, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata.get(self.getBase(filename), None)

    def extractFolder(self):
        for group in self.getOrganizedFolder():
            self.analyzeComments(group)

    def analyzeComments(self, group):
        context = ""
        for file in group:
            res = self.analyzeComment(file, context)
            context += f"\nFilename: {res[1]}   Summary: {res[-2]}   Relevant Issues: {res[-1]}"

    def analyzeComment(self, file, relevantInfo=""):
        filePath = f"{self.commentPath}/{file}"
        text = self.extract_text_from_pdf(filePath)
        metadata = self.getMetadata(file)

        response = self.interpolateResponse(metadata, file)
        response.append(text[:49000])
        self.env.addSheetRow(response)
        return response

    def interpolateResponse(self, metadata, filename):
        arr = []
        
        qf = QueryFile(self.docId, filename)
        naw = "If the data cannot be found, simply write 'N/A' and nothing else."
        
        arr.append(metadata.get("comment_id", ""))
        arr.append(filename)
        arr.append(metadata.get("date", ""))
        arr.append(qf.generateAnswer(f"Who is the commenter? Enter only the name. {naw} {metadata.get("commenter", "")}"))
        arr.append(qf.generateAnswer(f"What is the name of the organization submitting the comment? Enter only the name. {naw}"))
        
        if 'n/a' in arr[4]:
            arr.append("N/A")
            arr.append("N/A")
            arr.append("N/A")
            arr.append("N/A")
        else:
            arr.append(qf.generateAnswer(f"What is the type of the organization submitting the comment? Enter only the type. {naw}"))
            arr.append(qf.generateAnswer(f"What is the 501c Status of the organization submitting the comment? Enter only the status. {naw}"))
            arr.append(qf.generateAnswer(f"What is the NTEE type of the organization submitting the comment? Enter only the type. {naw}"))
            arr.append(qf.generateAnswer(f"What is the role of the organization submitting the comment? (select from Citizen Engagement, Individual Expression/Specialization, Innovation, Political Advocacy, Service Provision, Social Capital Creation, Products, Services, Government Institution, Citizen, Infastructure) Enter only the role. {naw}"))
            
        arr.append(qf.generateAnswer(f"What is the contact information of the comment's creators? Enter only the contact information. {naw}"))
        arr.append(qf.generateAnswer(f"What are some relevant keywords related to the comment? Enter only the keywords in this format: a, b, c, d."))
        arr.append(qf.generateAnswer(f"What is the summary of the comment? Enter only the summary in 1-2 brief sentences."))
        arr.append(qf.generateAnswer(f"What are some relevant issues addressed in the comment? Enter only the issues in this format: a, b, c, d."))

        return arr


if __name__ == "__main__":
    docNum = "2023-07776"
    env = Environment(os.getenv("OLLAMA_KEY"), os.getenv("GOOGLE_SHEET_URL"), sheetNum=4)
    analyzer = CommentAnalysis(env, docNum)
    analyzer.extractFolder()
