from ollama import embed
from dotenv import load_dotenv
from logger_config import Logger

import os

load_dotenv()

class CommentEmbedder:    
    MODEL = os.getenv("EMBED_MODEL")
    
    def __init__(self, docNum):
        self.docNum = docNum
        
        self.commentFolder = f"./comments/{docNum}"
        self.targetFolder=f"./embeddings/{docNum}"
        
        self.logger = self.setupLogger()
        
    def setupLogger(self):
        return Logger(log_folder=f"./logs/embeddings/{self.docNum}")
    
    def getFiles(self):
        return [i for i in os.listdir(self.commentFolder) if i.endswith(".pdf")]
