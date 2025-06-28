# simphy-llm/embeddings_simphy.py
# from langchain.document_loaders import PyPDFLoader ## deprecated 
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings # Deprecated, use langchain_community.embeddings instead
# from langchain_community.embeddings import HuggingFaceEmbedding

# from :class:`~langchain_huggingface.HuggingFaceEmbeddings` import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain.vectorstores import Qdrant #Deprecated, use langchain_community.vectorstores instead
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os
import logging
from langchain_core.documents import Document
import torch
import argparse
from langchain_community.vectorstores import FAISS

## thigns to do
# 1. Store the embeddings model locally, so that it can be reused without reloading
# 4  Store the embeddings in a vector store, such as Qdrant or FAISS for now qdrant
# 5. Add a method to save the vector store to disk for later use
# 6. Add a method to load the vector store from disk
# 2. Add a method to search the vector store with a query and return relevant documents that might be a method for the retrivar class
# 7. Add a method to update the vector store with new documents
# 3. Add error handling for file loading and embedding processes
# 8. Add a method to delete documents from the vector store
# 9. Need to see the docket and internet version of the qdrant client, for now using in-memory for testing
# 10. Add a method to clear the vector store
# 11. Add a method to modify the metadata of individual chunksi
# 12. Add command-line argument parsing for PDF path
# 13. Add seperate logic for case when vectorstore is created 


ALLMINILMV6 = "sentence-transformers/all-MiniLM-L6-v2"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# SimphyEmbedding class to load a PDF document, split it into chunks, and create embeddings
class SimphyEmbedding:
    def __init__(self, pdf_path,model_name:str=""):
        self.pdf_path = pdf_path
        self.vectorstore= None
        self.model_name = model_name 
        if model_name=="": self.model_name = ALLMINILMV6

        
        self.qdrantdb_path = os.path.join(SCRIPT_DIR,"qdrant_data/")  # Path to store Qdrant data, can be changed to a different path if needed

    def setuprag(self):
        pdf_docs = self.load_pdf()
        chunks = self.splitter(pdf_docs)
        vectorstore = self.vector_store(chunks)
        if vectorstore is None:
            logging.error("Failed to create vector store. Exiting.")
            exit(1)
        logging.info("RAG setup completed successfully.")
        
        
    
    def _load_and_embed(self):

        # Load the PDF document
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load_and_split()
        logging.info(f"Loaded {len(docs)} pages from the PDF document.")
        
        # and split it into chunks
        # need to split it into smaller chunks for processing, and playing with chunk size for better results
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # each chunk has 800 characters
            chunk_overlap=100    # overlap to preserve context between chunks
        )
        chunks = splitter.split_documents(docs)
        logging.info(f"Split the document into {len(chunks)} chunks.")
        if not chunks:
            logging.error("No chunks were created from the document. Please check the PDF file.")
        # logging.info("Value of chunk 1: %s", chunks[0].page_content[:300])  # Pprint first 300 characters of the first chunk for debugging

        # logging.info("Metadata of first chunk: %s", chunks[0].metadata)  # Print metadata of the first chunk for debugging


        
        
        cleaned_chunks = []
        for doc in chunks:
            safe_meta = {}
            for k, v in doc.metadata.items():
                try:
                    safe_meta[str(k)] = str(v)  # force string-only metadata
                except Exception:
                    continue  # skip any unserializable field

            cleaned_chunks.append(Document(page_content=doc.page_content, metadata=safe_meta))
        # create an embedding model, also here test different embedding models
        # to see which one gives better results
        # logging.info("checking metadata for cleaned chunks and chunks")
        # if not cleaned_chunks:
        #     logging.error("No cleaned chunks were created. Please check the document processing.")

        # logging.info("Number of cleaned chunks: %d", len(cleaned_chunks))
        # logging.info("Metadata of first cleaned chunk: %s", cleaned_chunks[0].metadata)  # Print metadata of the first cleaned chunk 
        # logging.info("Content of first normal chunk: %s", chunks[0].metadata)  # Print metadata of the first cleaned chunk
         
        # basic check for metadata

        # if cleaned_chunks[0].metadata== chunks[0].metadata:
        #     logging.info("Metadata of cleaned chunks matches original chunks.")
        # else:
        #     logging.warning("Metadata of cleaned chunks does not match original chunks. Please check the processing steps.")
        
        # difference in metadata between cleaned chunks and original chunks
        # for i in range(min(3, len(chunks))):  # Check up to 3 chunks
        #     print(f"\n--- Chunk {i} Metadata Comparison ---")
        #     original_meta = chunks[i].metadata
        #     cleaned_meta = cleaned_chunks[i].metadata
            
        #     print("Original metadata types:")
        #     for k, v in original_meta.items():
        #         print(f"  {k}: {type(v).__name__} = {v!r}")
            
        #     print("Cleaned metadata types:")
        #     for k, v in cleaned_meta.items():
        #         print(f"  {k}: {type(v).__name__} = {v!r}")
            
        #     # Find differences
        #     print("Differences:")
        #     for k in set(original_meta.keys()) | set(cleaned_meta.keys()):
        #         if k not in original_meta:
        #             print(f"  {k}: Only in cleaned metadata")
        #         elif k not in cleaned_meta:
        #             print(f"  {k}: Only in original metadata")
        #         elif original_meta[k] != cleaned_meta[k]:
        #             print(f"  {k}: Original={original_meta[k]!r}, Cleaned={cleaned_meta[k]!r}")

        ## Make sure chunks don't contain file objects
        # for i, chunk in enumerate(chunks[:5]):  # Check first 5 chunks
        #     logging.info(f"Chunk {i} metadata: {chunk.metadata if hasattr(chunk, 'metadata') else 'No metadata'}")
        #     print(" ")
        # print(chunks[0].model_dump())
        # exit()

        logging.info("Metadata of first chunk before cleaning: %s", chunks[0].metadata)  
        # Clean metadata to ensure all values are strings and remove problematic fields
        
        chunks = self.cleanmetadata(chunks)

        logging.info("Metadata of first chunk after cleaning: %s", chunks[0].metadata)  


        
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",

        )

        # Store the embeddings in a vector store
        
        client = QdrantClient(path="qdrant_data/")  # Use in-memory; for production use persistent storage ?? what does this line do?

        vectorstore = Qdrant.from_documents(
            documents=cleaned_chunks,
            embedding=embedding_model,
            collection_name="simphy_guide",
            client=client
        )

        return vectorstore
    
    def cleanmetadata(self, chunks):
        """
        Clean metadata from PDF chunks to make them serializable for Qdrant
        """
        logging.info(f"Cleaning metadata for {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            # Id issue, its empy 
            if hasattr(chunk, 'id') and chunk.id is None:
                chunk.id = f"chunk_{i}"  # might use uuid: f"chunk_{uuid.uuid4()}" in future


            if hasattr(chunk, 'metadata') and chunk.metadata:
                # Create clean metadata with only necessary, serializable fields
                clean_metadata = {
                    'title': str(chunk.metadata.get('title', 'Simphy Scripting Guide')),
                    'source': str(chunk.metadata.get('source', '')),
                    'page': int(chunk.metadata.get('page', 0)),
                    'total_pages': int(chunk.metadata.get('total_pages', 271))
                }
                
                # Only add page_label if it exists and is not empty
                page_label = chunk.metadata.get('page_label')
                if page_label and page_label.strip():
                    clean_metadata['page_label'] = str(page_label)
                
                # Replace the original metadata
                chunk.metadata = clean_metadata
                
            # Also ensure the chunk has the correct type
            if hasattr(chunk, 'type'):
                chunk.type = 'Document'
        
        logging.info("Metadata cleaning completed.")
        return chunks
    
    def load_pdf(self)->list[Document]:
        """
        Load the PDF document and return the raw text content.
        """
        try:
            loader = PyPDFLoader(self.pdf_path)
            docs = loader.load()
            # logging.info(f"Loaded {len(docs)} pages from the PDF document.")
            return docs
        except Exception as e:
            logging.error(f"Failed to load PDF document: {e}")
            return []
    
    def splitter(self, docs):
        """
        Split the loaded documents into smaller chunks.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,      # each chunk has 800 characters
                chunk_overlap=100    # overlap to preserve context between chunks
            )
            chunks = splitter.split_documents(docs)
            
            # Applying chunk formatiing is a good practice
            for chunk in chunks:
                chunk.page_content=f"Represent this passage for retrieval: {chunk.page_content}"
            # logging.info(f"Split the document into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.error(f"Failed to split documents: {e}")
            return []
    
    def vector_store(self, chunks):
        """
        Create a vector store from the chunks using HuggingFace embeddings.
        """
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=self.model_name, # try BAAI/bge-small-en,intfloat/multilingual-e5-small
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}, # Need to see about torch # Need to see about torch
                encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for better similarity search, maybe look for cosine similarity if avail
            )

            # client = QdrantClient(":memory:")  # Use in-memory
            # client = QdrantClient(path=self.qdrantdb_path)  # testing disk storage this time
            
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
            logging.info("Vector store created successfully.")
            
            self.vectorstore = vectorstore # Initialize the vectorstore attribute
            # Save the vectorstore to disk for later use
            return vectorstore
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            return None
        
    def retriever(self, query, k=3):
        """
        Retrieve relevant documents from the vector store based on a query.
        """
        try:
            if self.vectorstore is None:
                logging.error("Vector store is not initialized.")
                return []
            
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})  # Retrieve top 3 relevant documents
            # docs = retriever.get_relevant_documents(query)
            if self.model_name!=ALLMINILMV6:
                
                query = f"Represent this question for searching relevant passages:{query}"
            docs = retriever.invoke(query)  # Use invoke to get documents
            logging.info(f"Retrieved {len(docs)} documents for query: {query}")
            return docs
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            return []


        
#Seeing the vector 

if __name__ == "__main__":

    pdf_path = os.path.join(SCRIPT_DIR, "docs", "SimpScriptGPart3.pdf")
    # print(pdf_path)  # Path to your PDF file
    simphy_embedding = SimphyEmbedding(pdf_path)
    # vectorstore = simphy_embedding.load_and_embed()

    

    # pdf_doc = simphy_embedding.load_pdf()
    # chunks = simphy_embedding.splitter(pdf_doc)
    # vectorstore = simphy_embedding.vector_store(chunks)


    simphy_embedding.setuprag()  # Set up the RAG system
    
    logging.info("Testing retrieval...")
    # Interactive query loop
    logging.info("Enter your queries below. Type 'quit', 'exit', or 'q' to end the session.")
    
     
    while True:
        query = input("\nEnter your query (type 'clear' to clear screen, 'exit' to quit): ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            logging.warning("Exiting query session.")
            break
        
        if query.lower() in ['clear', 'cls']:
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        
        if not query.strip():
            logging.warning("Please enter a valid query.")
            continue
        
        docs = simphy_embedding.retriever(query)
        
        if not docs:
            logging.warning("No relevant documents found for your query.")
        else:
            logging.info(f"\nFound {len(docs)} relevant document(s):")
            for i, doc in enumerate(docs, 1):
                
                logging.info(f"\n\n--- Result {i} ---")
                logging.info(f"Page: {doc.metadata.get('page', 'Unknown')}")
                logging.info(f"Content: \n{doc.page_content}...")  # Show first 200 chars