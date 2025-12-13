from embedFile import ReadFiles
from ollama import chat
from dotenv import load_dotenv

import os

load_dotenv()

class QueryFile:
    def __init__(self, docId, filename):
        self.LLM_MODEL = os.getenv("LLM_MODEL")

        self.readFiles = ReadFiles(docId, filename)

    def getContext(self, query, k=5):
        context = "\n".join(self.readFiles.search(query, k))
        print("Top results:\n", context)
        return context

    def rewriteQuery(self, query, k=5):
        context = self.getContext(query, k)
        prompt = f"Rewrite the following query using context from the documents. Remember to include exactly what the query asks for in the rewritten query. Also, keep in note that this query is for analysis regarding a response to a federal RFI/RFC regarding AI.\n\nContext:\n{context}\n\nQuery: {query}\nRewritten Query:"
        resp = chat(model=self.LLM_MODEL, stream=False, messages=[{"role": "user", "content": prompt}])

        return resp['message']['content'], context

    def generateAnswer(self, query):
        query, context = self.rewriteQuery(query)
        print("\n\nRewritten query: " + query + "\n\n")
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        resp = chat(model=self.LLM_MODEL, stream=False, messages=[{"role": "user", "content": "Answer the following questions using the context from the documents below. Don't write in bullet points. Write in forms of short-concise sentences or phrases."}, {"role": "user", "content": prompt}])
        
        return resp['message']['content']
        

if __name__ == "__main__":
    qf = QueryFile("2023-07776", "N__CommentonFRDoc202307776__20230615__NTIA-2023-0005-0040.pdf")

    while True:
        inp = input("Enter your query: ")
        if inp.lower() == "exit" or inp.lower() == "q":
            break
        print("\n\nAnswer:\n", qf.generateAnswer(inp) + "\n\n")
