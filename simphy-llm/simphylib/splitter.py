# PDF loading + chunking

# from langchain.document_loaders import PyPDFLoader ## deprecated 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
logger = logging.getLogger(__name__)
from langchain_core.documents import Document
# from simphylib.embedder import EmbeddingsSimphy 
from embedder import EmbeddingsSimphy

import os 
from rake_nltk import Rake


import nltk 
nltk.download('stopwords')
nltk.download('punkt_tab')


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
    

page_prahse = []
if __name__ == "__main__":
    # Example usage
    pdf_path = "simphy-llm/docs/SimpScriptGPart4Ch4.pdf"  # Replace with your PDF file path
    # chunker = PDFChunker(pdf_path=pdf_path, chunk_size=1000, chunk_overlap=100)
    # docs = chunker.load()
    # chunks = chunker.split()

    docs = PyPDFLoader(pdf_path).load()

    print(docs[0])
    for doc in docs:
        r = Rake()
        r.extract_keywords_from_text(doc.page_content)
        phrases = r.get_ranked_phrases()
        page_prahse.append(phrases)
    

    print(f"Loaded {len(docs)} documents")
    print(docs[0].page_content)  # Print first 200 characters of the first document
    print(f"Extracted phrases from the first document: \n{page_prahse[0]}")

    for i in page_prahse:
        print("Phrases from document:")

        print(i)
        print("-" * 30+ f"{len(i)} different keywords")
        
    