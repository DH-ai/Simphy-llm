# Constants like model names and path

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLMINILM = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "../faiss_index")
FAISS_META_PATH = os.path.join(SCRIPT_DIR, "../faiss_embeddings.pkl")
CACHED_INDEX_PATH = os.path.join(SCRIPT_DIR, "../cached_index/vectorstore_new.pkl")

GENMODEL = "gemini-2.0-flash"
HUGGINGFACE_EMBEDDING_MODEL_BAAI = "BAAI/bge-base-en-v1.5"