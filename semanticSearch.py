import json
import numpy as np
from ollama import embed

class SemanticSearch:
    MODEL = "mxbai-embed-large"
    FOLDER = "./embeddings"
    TOP_K = 5
        
    def __init__(self, query):  
        self.query = query
        self.query_embedding = np.array(embed(input=self.query, model=self.MODEL))
        
        self.loadEmbeddings()


    def loadEmbeddings(self):
        embeddings = []
        
        for file in os.listdir(self.FOLDER):
            if file.endswith(".json"):
                with open(os.path.join(self.FOLDER, file), "r") as f:
                    embeddings.append(json.load(f))
                    
        self.embeddings = np.array(embeddings, dtype="float32")
        
    @staticmethod
    def cosineSimilarity(a, b):
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.dot(a_norm, b_norm)
    
    def fetchResults(self, top_k=TOP_K):
        indices = np.argsort(self.cosineSimilarity(self.query_embedding, self.embeddings))
        return indices[-top_k:][::-1]
    

def main():
    query = input("Enter query to search: ")
    search = SemanticSearch(query)
    results = search.fetchResults()
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")
    
    
if __name__ == "__main__":
    main()