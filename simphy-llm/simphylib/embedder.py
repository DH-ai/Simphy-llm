#Embedding + caching logic (new)


from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import os
import logging
import os 
# from simphylib.config import CACHED_INDEX_PATH
import pickle
CACHED_INDEX_PATH = "vectorstore_new.pkl"  # Path to save the cached vector store

logger = logging.getLogger(__name__)
class EmbeddingsSimphy:
    """
    Class to handle embedding generation and vector store creation using HuggingFace embeddings.
    """
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", qdrantdb_path=None):
        self.model_name = model_name
        self.qdrantdb_path = qdrantdb_path if qdrantdb_path else ":memory:"  # Default to in-memory if no path is provided
        self.vectorstore = None  # Initialize vectorstore attribute

    def create_vectorstore(self, chunks) -> FAISS | None:
        """
        Create vectorstore for the given chunks of text.
        """
        if self.check_vectorstore():
            try:
                logger.info("Loading cached vector store from disk...")
                with open(CACHED_INDEX_PATH, "rb") as f:
                    vectorstore = pickle.load(f)
                return vectorstore
            except Exception as e:
                logger.error(f"Failed to load cached vector store: {e}")
                return None
        else:
            try:
                embedding_model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True},  # Normalize embeddings for better similarity search
                    # show_progress=True,
                      # Show progress bar during embedding generation)
                )
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    # collection_name="simphy_guide",  # Not needed for FAISS
                )  

                # embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
                # logging.info("Embeddings created successfully.")
                self.vectorstore = vectorstore # Initialize the vectorstore attribute
                
                
                with open(CACHED_INDEX_PATH, "wb") as f:
                    pickle.dump(vectorstore, f)
                return vectorstore
            except Exception as e:
                logging.error(f"Failed to create embeddings: {e}")
                
                return None

    def load_vectorstore(self)->FAISS |None:
        """Load the vectorstore from disk."""
        try:
            if not os.path.exists(CACHED_INDEX_PATH):
                raise FileNotFoundError(f"Vector store file not found at {CACHED_INDEX_PATH}")
            with open(CACHED_INDEX_PATH, "rb") as f:
                self.vectorstore = pickle.load(f)
            logger.info("Vector store loaded successfully.")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return None
    def check_vectorstore(self):
        """Check if the vectorstore is already created."""
        # if os.path.exists(CACHED_INDEX_PATH):
        #     # logger.info("Vector store already exists.")
        #     return True
        # else:
        #     # logger.warning("No vector store found. Please create one first.")
        #     return False
        return False
    



