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

# import time
from time import time
# avoid saving the model
# Store embeddings only
# write my own retriver in place of langchain retriever
# So currently we are saving the embeddings model not the embeddings themselves therefore we are facing almost 
# same runtime, if we generate the embeddings there is no need of embeddings model, means no embeddings making from chunks therefare faster runtime
# Now we need to figure how we are going to store the embeddings -> maybe we did storing the entire vectore store damn
# can cache huggingface embeddings model cache folder arg to huggingface embeddings class -> further development says that saving vector store is not that efficient too took 4 seconds equival
# - to the first approach, noew we will try other approach will also enable caching
# time = time.time()
def vector_store( chunks)-> FAISS| None:
        """
        Create a vector store from the chunks using HuggingFace embeddings.
        """

        # lets try to load the vector store from disk first
        
        
        if os.path.exists("vectorstore_new.pkl") :
            print("Loading FAISS vector store from disk...")
            t = time()
            with open("vectorstore_new.pkl", "rb") as f:
                vector_store = pickle.load(f)
            print("Time req to load vectorstore from disk: {t}".format(t=time() - t))
            return vector_store
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5", # try BAAI/bge-small-en,intfloat/multilingual-e5-small
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}, # Need to see about torch # Need to see about torch
            encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search, maybe look for cosine similarity if avail
        )

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
            # collection_name="simphy_guide",  # Not needed for FAISS
        )
        with open("vectorstore_new.pkl", "wb") as f:
            pickle.dump(vectorstore, f)
        
        return vectorstore
        # pass 
        # try:

        #     if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_INDEX_METADATA_PATH):
        #         print("Loading FAISS vector store from disk...")
        #         with open(FAISS_INDEX_METADATA_PATH, "rb") as f:
        #             stored_embeddings = pickle.load(f,)
        #         vectorstore = FAISS.load_local(FAISS_INDEX_PATH, stored_embeddings,allow_dangerous_deserialization=True)
        #         return vectorstore
        #     embedding_model = HuggingFaceEmbeddings(
        #         model_name="BAAI/bge-base-en-v1.5", # try BAAI/bge-small-en,intfloat/multilingual-e5-small
        #         model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}, # Need to see about torch # Need to see about torch
        #         encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search, maybe look for cosine similarity if avail
        #     )

        #     # client = QdrantClient(":memory:")  # Use in-memory
        #     # client = QdrantClient(path=self.qdrantdb_path)  # testing disk storage this time
        #     print("unable to Load FAISS vector store from disk...")
            
        #     ## using qdrant 
        #     # vectorstore = Qdrant.from_documents(
        #     #     documents=chunks,
        #     #     embedding=embedding_model,
        #     #     collection_name="simphy_guide",
        #     #     # client=client
        #     #     location=":memory:", # Check if this works with self.qdrantdb_path
        #     # )

        #     ## using FAISS for now
        #     # vectorstore = FAISS.from_documents(
        #     vectorstore = FAISS.from_documents(
        #         documents=chunks,
        #         embedding=embedding_model,
        #         # collection_name="simphy_guide",  # Not needed for FAISS
        #     )            
        #     # logging.info("Vector store created successfully.")
        #     vectorstore.save_local(FAISS_INDEX_PATH)
        #     with open(FAISS_INDEX_METADATA_PATH, "wb") as f:
        #         pickle.dump(embedding_model, f)
        #     # Save the vectorstore to disk for later use
        #     return vectorstore
        # except Exception as e:
        #     logging.error(f"Failed to create vector store: {e}")
            
            # return None


def create_vectorstore(chunks):
    """
    Create vectorstore for the given chunks of text.
    """
    try:
        t = time()
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=True,  # Show progress bar during embedding generation
            # multi_process=True,  # Use multiple processes for faster embedding generation
            # Note: multi_process=True is not supported in all environments,

              # Normalize embeddings for better similarity search
        )
        print("Time req to run embedding model: {t}".format(t= time()-t))  # Generate embeddings for each chunk
        t = time()
        embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
        # logging.info("Embeddings created successfully.")
        print("Time req to create embeddings: {t}".format(t=time() - t))

        return embeddings
    except Exception as e:
        logging.error(f"Failed to create embeddings: {e}")
        return []

# def create_vectorstore_from_embeddigs(embeddings, chunks):
#     """
#     Create vectorstore from the given embeddings and chunks of text.
#     """
#     try:
        
#         # Create a vector store using the embeddings and chunks
#         # vectorstore = FAISS.from_embeddings(
#         #     documents=chunks,
#         #     embedding=embeddings,
#         #     # collection_name="simphy_guide",  # Not needed for FAISS
#         # ) 
#         return vectorstore
#     except Exception as e:
#         logging.error(f"Failed to create vector store from embeddings: {e}")
#         return None

test_docs = [
    Document(page_content="Quantum mechanics explores the behavior of particles at atomic scales.", metadata={"source": "physics"}), #1
    Document(page_content="The French Revolution began in 1789 and drastically changed the course of history.", metadata={"source": "history"}),#2
    Document(page_content="Photosynthesis allows plants to convert sunlight into energy.", metadata={"source": "biology"}),#3
    Document(page_content="Machine learning is a subset of artificial intelligence focused on pattern recognition.", metadata={"source": "ai"}),#4
    Document(page_content="Rome was not built in a day — it's a proverb highlighting the need for patience.", metadata={"source": "proverbs"}),#5
    Document(page_content="Shakespeare wrote tragedies like Hamlet and Macbeth in the 16th century.", metadata={"source": "literature"}),#6
    Document(page_content="Blockchain is a distributed ledger technology behind cryptocurrencies.", metadata={"source": "technology"}),#7
    Document(page_content="Mount Everest is the tallest mountain above sea level on Earth.", metadata={"source": "geography"}),#8
    Document(page_content="The mitochondrion is known as the powerhouse of the cell.", metadata={"source": "cell-bio"}),#9
    Document(page_content="The Pythagorean theorem describes a fundamental relationship in Euclidean geometry.", metadata={"source": "math"})#9
]

# create_vectorstore_from_embeddigs(create_vectorstore(test_docs), test_docs)
t = time()
vc= create_vectorstore(test_docs)
print("Time req to run functino")
print(time() - t)
# if vc is not None:
#     retriever = vc.as_retriever(search_type="similarity", search_kwargs={"k": 1})
#     t = time()

#     print(retriever.invoke("most historical documetns"))
#     print("Time req to run search : {t}".format(t=time() - t))

