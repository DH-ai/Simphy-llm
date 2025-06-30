# PDF loading + chunking

# from langchain.document_loaders import PyPDFLoader ## deprecated 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
logger = logging.getLogger(__name__)
from langchain_core.documents import Document
from simphylib.embedder import EmbeddingsSimphy

import os 






class PDFChunker:

    """Class to handle loading and chunking of PDF documents.
    """
    def __init__(self, pdf_path:str="", chunk_size=1000, chunk_overlap=100):
        if not os.path.exists(pdf_path) and pdf_path!="":
            try:
                print(f"PDF path {pdf_path} does not exist. Attempting to resolve relative path.")
                pdf_path = os.path.join(os.getcwd(), pdf_path)
                os.path.exists(pdf_path)  # Check if the path exists after joining with SCRIPT_DIR
            except Exception as e:
                logger.error(f"Error in resolving PDF path: {e}")
                raise e
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs:list[Document] = []
        self.chunks:list[Document] = []

        
    def load(self):
        """Load the PDF file and return the documents."""
        try:
            if not self.pdf_path:
                raise ValueError("PDF path must be provided.")
            logger.info(f"Loading PDF from {self.pdf_path}")
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise e
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
        for chunk in self.chunks:
            chunk.page_content= f"Represent this passage for retrieval: {chunk.page_content}"
        return self.chunks

    
    def check_vectorstore_before_load(self):
        """Check if the vectorstore is already created."""
        return EmbeddingsSimphy().check_vectorstore()
        
    
    def __repr__(self):
        return f"PDFChunker(pdf_path={self.pdf_path}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
    