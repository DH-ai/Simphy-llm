# Query interface
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS



## custom retriever for Simphy using similartiy search + keyword matchinng implmenting hybrid search,
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
        
        query = f"Represent this question for searching relevant passages: {query}"
        ret = self.vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":k}) 
        doc = ret.invoke(query) #1
        # print(f"Query: {query}")
        # results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k) #2

        # for i, (d1c, score) in enumerate(results_with_scores):
        #     print(f"\nResult {i+1}")
        #     print(f"Score: {score}")
        #     print(f"Content: {d1c.page_content[:200]}")
        #     print(f"Metadata: {d1c.metadata}")
        
        
        return doc