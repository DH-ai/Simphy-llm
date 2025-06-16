# simphy-llm/embeddings_simphy.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os

class SimphyEmbedding:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    
    def load_and_embed(self):
        # Load the PDF document
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load_and_split()

        
        # and split it into chunks
        # need to split it into smaller chunks for processing, and playing with chunk size for better results
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,      # each chunk has 800 characters
            chunk_overlap=100    # overlap to preserve context between chunks
        )
        chunks = splitter.split_documents(docs)

        # create an embedding model, also here test different embedding models
        # to see which one gives better results
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Store the embeddings in a vector store
        
        client = QdrantClient(":memory:")  # Use in-memory; for production use persistent storage ?? what does this line do?

        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="simphy_guide",
            client=client
        )

        return vectorstore



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "docs", "SimpScriptG.pdf")
    print(pdf_path)  # Path to your PDF file
    simphy_embedding = SimphyEmbedding(pdf_path)
    vectorstore = simphy_embedding.load_and_embed()

    # Example query to test the retriever
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents
    # You can change the query to test different questions
    
    query = "How do I print to console in SimPhy?"
    docs = retriever.get_relevant_documents(query)

    for d in docs:
        print("---")
        print(d.page_content[:300])