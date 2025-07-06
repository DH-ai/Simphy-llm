# Document loading + chunking

# from langchain.document_loaders import PyPDFLoader ## deprecated 
from langchain_community.document_loaders import PyPDFLoader ## Now we are using LLLSherpaFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, SpacyTextSplitter, CharacterTextSplitter, NLTKTextSplitter,SentenceTransformersTokenTextSplitter
import logging
logger = logging.getLogger(__name__)
from langchain_core.documents import Document

try:
    from simphylib.embedder import EmbeddingsSimphy 
    from simphylib.retriever import RetrieverSimphy
    from simphylib.config import HUGGINGFACE_EMBEDDING_MODEL_BAAI
except ImportError:
    # If the import fails, it means the module is not in the path.
    # This is a workaround to avoid circular imports.
    from embedder import EmbeddingsSimphy
    from retriever import RetrieverSimphy
    from config import HUGGINGFACE_EMBEDDING_MODEL_BAAI
# from embedder import EmbeddingsSimphy
try:
    from simphylib.parser import SimphyFileLoader
except ImportError:
    from parser import SimphyFileLoader
import os 
from rake_nltk import Rake


import nltk 


# nltk.download('stopwords',download_dir=os.path.join(os.getcwd(), "nltk_data"),)

# nltk.download('punkt_tab',download_dir=os.path.join(os.getcwd(), "nltk_data"))


class PDFChunker:
    ## add a method for different type of loaders and splitters
    """Class to handle loading and chunking of PDF documents.
    """


    allowed_loaders = ["SimphyFileLoader", "PyPDFLoader"]
    allowed_splitters = [
        "RecursiveCharacterTextSplitter",
        "TokenTextSplitter",
        "SpacyTextSplitter",
        "CharacterTextSplitter",
        "NLTKTextSplitter",
        "SentenceTransformersTokenTextSplitter"
    ]
    def __init__(self, pdf_path:str="", chunk_size=1000, chunk_overlap=200, loader="SimphyFileLoader", splitter="RecursiveCharacterTextSplitter"):

        if not os.path.exists(pdf_path) and pdf_path!="":
            try:
                print(f"PDF path {pdf_path} does not exist. Attempting to resolve relative path.")
                pdf_path = os.path.join(os.getcwd(), pdf_path)
                os.path.exists(pdf_path)  # Check if the path exists after joining with SCRIPT_DIR
            except Exception as e:
                logger.error(f"Error in resolving PDF path: {e}")
                raise e
        
        self.loader = loader
        self.splitter = splitter
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
            # logger.info(f"Loading PDF from {self.pdf_path}")
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise e
        
        if self.loader == "SimphyFileLoader":
            # Use SimphyFileLoader to load the PDF
            loader = SimphyFileLoader(self.pdf_path)
        elif self.loader == "PyPDFLoader":
            # Use PyPDFLoader to load the PDF
            loader = PyPDFLoader(self.pdf_path)
        

         
        loader = SimphyFileLoader(self.pdf_path)
        self.docs = loader.load()
        return self.docs

    def split(self, splitter_name: str|None = None):
        """Split the loaded documents into chunks."""
         # If docs are not loaded, load them first
         # This is to avoid loading the PDF multiple times unnecessarily
        
        if splitter_name:
            # if splitter_name not in PDFChunker.allowed_splitters:
                self.splitter = splitter_name
                # logger.warning(f"Invalid splitter type '{splitter_name}', Allowed types {PDFChunker.allowed_splitters}")
                
        
        if not self.docs:
            self.load()

        if self.splitter == "RecursiveCharacterTextSplitter":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.splitter == "TokenTextSplitter":
            splitter = TokenTextSplitter(
                # encoding_name="gpt-2",  
                chunk_size=self.chunk_size,

                chunk_overlap=self.chunk_overlap
                
            )
        elif self.splitter == "SpacyTextSplitter":   
            splitter = SpacyTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.splitter == "CharacterTextSplitter":
            splitter = CharacterTextSplitter(

            )
        elif self.splitter == "NLTKTextSplitter":
            splitter = NLTKTextSplitter(

            )
        elif self.splitter == "SentenceTransformersTokenTextSplitter":   
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap,
                model_name=HUGGINGFACE_EMBEDDING_MODEL_BAAI
            )
        elif self.splitter not in PDFChunker.allowed_splitters:
            logger.warning(f"Invalid splitter type '{self.splitter}, Using RecursiveCharacterTextSplitter Allowed types {PDFChunker.allowed_splitters}' ")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        self.chunks = splitter.split_documents(self.docs)
        for chunk in self.chunks:
            chunk.page_content= f"Represent this passage for retrieval: {chunk.page_content}"
        return self.chunks

    @staticmethod
    def check_vectorstore_before_load():
        """Check if the vectorstore is already created."""
        return EmbeddingsSimphy().check_vectorstore()
    
    @staticmethod
    def delete_vectorstore():
        """Delete the existing vectorstore."""
        return EmbeddingsSimphy().delete_vectorstore()
    def modify_metadata(self):

        pass
    def __repr__(self):
        return f"PDFChunker(pdf_path={self.pdf_path}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
   

if __name__ == "__main__":
    page_prahse = []

    pdf_path = "simphy-llm/docs/SimpScriptGPart4Ch4.pdf"  # Replace with your PDF file path
    # chunker = PDFChunker(pdf_path=pdf_path, chunk_size=1000, chunk_overlap=100)
    # docs = chunker.load()
    # chunks = chunker.split()

    chunker = PDFChunker(pdf_path=pdf_path, chunk_size=1024, chunk_overlap=256, loader="PyPDFLoader", splitter="TokenTextSplitter")

    docs = chunker.load()
    chunks = chunker.split()
    print(f"Loaded {len(docs)} documents and split into {len(chunks)} chunks.")
    query = "Tell me about add and creating a dialog box"
    vectorstore=   EmbeddingsSimphy(save_vectorstore=False).create_vectorstore(chunks)  # Create vector store with the chunks
    retriever = RetrieverSimphy(vectorstore=vectorstore)
    results = retriever.retrieve(query=query, k=5)
    print(f"Retrieved {len(results)} documents for query '{query}':")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        # print(f"Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content:\n\n {doc.page_content} \n") # Print first 200 characters of content
   