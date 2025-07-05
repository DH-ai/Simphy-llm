# Constants like model names and path

import os
CURRENT_DIR = os.getcwd()
ALLMINILM = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_PATH = os.path.join(CURRENT_DIR, "vectorstore_new.pkl")
GENMODEL = "gemini-2.0-flash"
HUGGINGFACE_EMBEDDING_MODEL_BAAI = "BAAI/bge-base-en-v1.5"
DEFAULT_LLMSHERPAURL = "http://172.17.0.2:5001/api/parseDocument?renderFormat=all&useNewIndentParser=true"
LLMSHERPA_IMAGE = "ghcr.io/nlmatics/nlm-ingestor:latest"
LLMSHERPA_CONTAINER_NAME = "LLMSherpaFileLoader"
DOCKER_ERROR_MESSAGE = (
    "Make sure Docker is running and you have the correct permissions.\n"
    "Run the following command to add your user to the docker group:\n"
    "sudo usermod -aG docker $USER"
)