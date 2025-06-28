#Embedding + caching logic (new)

import pickle
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
import os
import logging
FAISS_INDEX_PATH = "vectorstore_index"
FAISS_INDEX_METADATA_PATH = "vectorstore.pkl"
# Set up logging
logger = logging.getLogger(__name__)
# WARNING: Pickle deserialization can execute arbitrary code. Only load trusted files.

def vector_store( chunks):
        """
        Create a vector store from the chunks using HuggingFace embeddings.
        """


        try:

            if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_METADATA_PATH):
                print("Loading FAISS vector store from disk...")
                with open(FAISS_INDEX_METADATA_PATH, "rb") as f:
                    stored_embeddings = pickle.load(f,)
                vectorstore = FAISS.load_local(FAISS_INDEX_PATH, stored_embeddings,allow_dangerous_deserialization=True)
                return vectorstore
            embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-base-en-v1.5", # try BAAI/bge-small-en,intfloat/multilingual-e5-small
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}, # Need to see about torch # Need to see about torch
                encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search, maybe look for cosine similarity if avail
            )

            # client = QdrantClient(":memory:")  # Use in-memory
            # client = QdrantClient(path=self.qdrantdb_path)  # testing disk storage this time
            print("unable to Load FAISS vector store from disk...")
            
            ## using qdrant 
            # vectorstore = Qdrant.from_documents(
            #     documents=chunks,
            #     embedding=embedding_model,
            #     collection_name="simphy_guide",
            #     # client=client
            #     location=":memory:", # Check if this works with self.qdrantdb_path
            # )

            ## using FAISS for now
            # vectorstore = FAISS.from_documents(
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embedding_model,
                # collection_name="simphy_guide",  # Not needed for FAISS
            )            
            # logging.info("Vector store created successfully.")
            vectorstore.save_local(FAISS_INDEX_PATH)
            with open(FAISS_INDEX_METADATA_PATH, "wb") as f:
                pickle.dump(embedding_model, f)
            # Save the vectorstore to disk for later use
            return vectorstore
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            # return None
        

# def create_vectorstore(chunks):
#     """
#     Create vectorstore for the given chunks of text.
#     """
#     try:
#         embedding_model = HuggingFaceEmbeddings(
#             model_name="BAAI/bge-base-en-v1.5",
#             model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
#             encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search
#         )
#         # Generate embeddings for each chunk
#         embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
#         # logging.info("Embeddings created successfully.")
        

#         return embeddings
#     except Exception as e:
#         logging.error(f"Failed to create embeddings: {e}")
#         return []

test_docs = [
    Document(page_content="Quantum mechanics explores the behavior of particles at atomic scales.", metadata={"source": "physics"}),
    Document(page_content="The French Revolution began in 1789 and drastically changed the course of history.", metadata={"source": "history"}),
    Document(page_content="Photosynthesis allows plants to convert sunlight into energy.", metadata={"source": "biology"}),
    Document(page_content="Machine learning is a subset of artificial intelligence focused on pattern recognition.", metadata={"source": "ai"}),
    Document(page_content="Rome was not built in a day — it's a proverb highlighting the need for patience.", metadata={"source": "proverbs"}),
    Document(page_content="Shakespeare wrote tragedies like Hamlet and Macbeth in the 16th century.", metadata={"source": "literature"}),
    Document(page_content="Blockchain is a distributed ledger technology behind cryptocurrencies.", metadata={"source": "technology"}),
    Document(page_content="Mount Everest is the tallest mountain above sea level on Earth.", metadata={"source": "geography"}),
    Document(page_content="The mitochondrion is known as the powerhouse of the cell.", metadata={"source": "cell-bio"}),
    Document(page_content="The Pythagorean theorem describes a fundamental relationship in Euclidean geometry.", metadata={"source": "math"})
]

