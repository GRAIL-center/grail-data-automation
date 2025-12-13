from extractText import extractFile
from ollama import embed

import numpy as np
import os

import faiss
import dotenv
import io


dotenv.load_dotenv()

class ReadFiles:
    def __init__(self, docId, file):
        self.embed_model = os.getenv("EMBED_MODEL")
        self.llm_model = os.getenv("LLM_MODEL")
        self.chunk_size = int(os.getenv("CHUNK_SIZE"))
        self.folder = f"./comments/{docId}"

        self.file = os.path.join(self.folder, file)
        self.embeddings = []
        self.documents = []
        
        self.loadFile()

    def embedText(self, text):
        res = embed(model=self.embed_model, input=text)
        emb = np.array(res["embeddings"], dtype=np.float32)

        return emb
    
    def storeEmbed(self):
        emb_matrix = np.vstack(self.embeddings)

        dim = emb_matrix.shape[1]
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(emb_matrix)
        
    
    def loadFile(self):
        text = extractFile(self.file)
        chunks = self.chunkText(text)

        for chunk in chunks:
            self.documents.append(chunk)
            self.embeddings.append(self.embedText(chunk))

        self.storeEmbed()
        
    def chunkText(self, text):
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            while end < len(text) and text[end] not in [".", "!", "?"]:
                end += 1

            if end < len(text):
                end += 1

            chunks.append(text[start:end])
            start = end

        return chunks
            
    def search(self, query, k=5):
        query_emb = self.embedText(query) 
        D, I = self.index.search(query_emb, k=k)
        res = [self.documents[i] for i in I[0]]
        return res

if __name__ == "__main__":
    res = ReadFiles("2023-07776", "N__CommentonFRDoc202307776__20230615__NTIA-2023-0005-0040.pdf").search("Who is the submitter of the comment?")
    print(res)