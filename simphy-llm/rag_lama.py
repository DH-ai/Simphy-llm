from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class SimplePDFRetriever:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.retriever = self._setup_retriever()

    def _load_documents(self):
        loader = PyPDFLoader(self.pdf_path)
        return loader.load()

    def _split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(docs)

    def _setup_retriever(self):
        docs = self._load_documents()
        splits = self._split_documents(docs)
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        return vectorstore.as_retriever()

    def search(self, query: str, top_k: int = 6):
        return self.retriever.get_relevant_documents(query)[:top_k]

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(SCRIPT_DIR, "docs", "SimpScriptGPart3.pdf")
    
    retriever = SimplePDFRetriever(pdf_path)
    query = "Create a triangular wedge with sides 6, 8 and 10 and mass 5"
    matches = retriever.search(query)

    for i, doc in enumerate(matches):
        print(f"\n--- Match {i+1} ---\n")
        print(doc.page_content)
