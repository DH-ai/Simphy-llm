# Constants like model names and path

import os
CURRENT_DIR = os.getcwd()
ALLMINILM = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = os.path.join(CURRENT_DIR, "vectorstore_new.pkl")
GENMODEL = "gemini-2.0-flash"
HUGGINGFACE_EMBEDDING_MODEL_BAAI = "BAAI/bge-base-en-v1.5"