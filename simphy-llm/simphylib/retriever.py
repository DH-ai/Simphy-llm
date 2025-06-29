# Query interface
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

class RetrieverSimphy:
    """
    Class to handle retrieval of documents using a vector store.
    """
    
    def __init__(self, vectorstore: FAISS|None = None):
        self.vectorstore:FAISS|None = vectorstore

    def retrieve(self, query, k=5)->list[Document]:
        """
        Retrieve documents based on the query.
        
        Args:
            query (str): The query string to search for.
            k (int): The number of documents to retrieve.
        
        Returns:
            list: A list of retrieved documents.
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vector store is not initialized.")
        except Exception as e:
            raise ValueError(f"Error in retrieving documents: {e}")
        
        query = "Represent this question for searching relevant passages: {query}"
        ret = self.vectorstore.as_retriever(search_kwargs={"k": k})
        doc = ret.invoke(query) #1
        
        
        return doc