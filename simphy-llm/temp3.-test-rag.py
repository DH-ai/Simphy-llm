from simphylib.splitter import PDFChunker
from simphylib.embedder import EmbeddingsSimphy
from simphylib.retriever import RetrieverSimphy
from simphylib.config import *
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


import logging
logging.basicConfig()
if __name__ == "__main__":
    


    
    logging.info("This is SLiPI, your SimPhy Scripting Assistant.")
    logging.info("Loading PDF and creating vector store...")
    if not PDFChunker().check_vectorstore_before_load():

        pdf_chunker = PDFChunker(pdf_path=SCRIPT_DIR+"/docs/SimpScriptGPart4Ch4.pdf", chunk_size=1000, chunk_overlap=100)
        pdf_chunker.load()
        chunks = pdf_chunker.split()
        # chunks = pdf_chunker.format_chunks()
        embedder = EmbeddingsSimphy(model_name=HUGGINGFACE_EMBEDDING_MODEL_BAAI)
        vectorstore = embedder.create_vectorstore(chunks)
    else:
        logging.info("Vector store already exists. Loading from cache...")
        embedder = EmbeddingsSimphy(model_name=HUGGINGFACE_EMBEDDING_MODEL_BAAI)
        vectorstore = embedder.load_vectorstore()
    if vectorstore is None:
        logging.error("Failed to load vector store. Exiting.")
        exit(1)
    retriever = RetrieverSimphy(vectorstore=vectorstore)
    

    logging.info("Enter your queries below. Type 'quit', 'exit', or 'q' to end the session.")
    while True:
        query = input("Query: ")
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query.lower() in ['clear', 'cls']:
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        if not query.strip():
            logging.warning("Please enter a valid query.")
            continue

        
        
        doc = retriever.retrieve(query=query, k=7)

        logging.info(f"Result of the query: {query}\n\n".format(query=query))
        for i, doc2 in enumerate(doc, 1):
                
                print(f"\n\n--- Result {i} ---\n\n")
                print(f"Page: {doc2.metadata.get('page', 'Unknown')}")
                print(f"Content: \n{doc2.page_content}...")
        logging.info("\n\n End of RAG Outout ")


            

