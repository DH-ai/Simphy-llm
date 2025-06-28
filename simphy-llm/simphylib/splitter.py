# PDF loading + chunking

# from langchain.document_loaders import PyPDFLoader ## deprecated 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
logger = logging.getLogger(__name__)

# Avoid this entire thing if Embeddings are already created

class PDFChunker:

    """Class to handle loading and chunking of PDF documents.
    """
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=100):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self.chunks = []

    def load(self):
        """Load the PDF file and return the documents."""
        loader = PyPDFLoader(self.pdf_path)
        self.docs = loader.load()
        return self.docs

    def split(self):
        """Split the loaded documents into chunks."""
         # If docs are not loaded, load them first
         # This is to avoid loading the PDF multiple times unnecessarily
        
        
        
        if not self.docs:
            self.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.chunks = splitter.split_documents(self.docs)
        return self.chunks

    def format_chunks(self, prefix="Represent this passage for retrieval: "):
        for chunk in self.chunks:
            chunk.page_content = f"{prefix}{chunk.page_content}"
        return self.chunks