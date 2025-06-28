#Embedding + caching logic (new)


from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)
class EmbeddingsSimphy:
    """
    Class to handle embedding generation and vector store creation using HuggingFace embeddings.
    """
    def __init__(self, model_name="BAAI/bge-small-en", qdrantdb_path=None):
        self.model_name = model_name
        self.qdrantdb_path = qdrantdb_path if qdrantdb_path else ":memory:"  # Default to in-memory if no path is provided
        self.vectorstore = None  # Initialize vectorstore attribute

    def create_vectorstore(self, chunks):
        """
        Create vectorstore for the given chunks of text.
        """
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
            )

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model,
                # collection_name="simphy_guide",  # Not needed for FAISS
            )  

            # embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
            # logging.info("Embeddings created successfully.")
            self.vectorstore = vectorstore # Initialize the vectorstore attribute

            return vectorstore
        except Exception as e:
            logging.error(f"Failed to create embeddings: {e}")
            return None

    



