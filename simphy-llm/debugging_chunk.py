# Add this debug code before your Qdrant.from_documents() call
import pickle
import copy
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import traceback
# Configure logging
logging.basicConfig(level=logging.DEBUG)

logging.info("Debugging chunks for serialization issues...")

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "docs", "SimpScriptG.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load_and_split()
logging.info(f"Loaded {len(docs)} pages from the PDF document.")

# and split it into chunks
# need to split it into smaller chunks for processing, and playing with chunk size for better results
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # each chunk has 800 characters
    chunk_overlap=100    # overlap to preserve context between chunks
)
chunks = splitter.split_documents(docs)


# for i, chunk in enumerate(chunks[:5]):  # Test first 5 chunks
#     try:
#         # Test if the chunk can be pickled
#         pickle.dumps(chunk)
#         logging.debug(f"Chunk {i}: Pickle OK")
#     except Exception as e:
#         logging.error(f"Chunk {i}: Pickle FAILED - {e}")
    
#     try:
#         # Test if the chunk can be deep copied
#         copy.deepcopy(chunk)
#         logging.debug(f"Chunk {i}: Deep copy OK")
#     except Exception as e:
#         logging.error(f"Chunk {i}: Deep copy FAILED - {e}")
        
#     # Check individual attributes
#     for attr in ['id', 'metadata', 'page_content', 'type']:
#         if hasattr(chunk, attr):
#             try:
#                 copy.deepcopy(getattr(chunk, attr))
#                 logging.debug(f"Chunk {i}.{attr}: OK")
#             except Exception as e:
#                 logging.error(f"Chunk {i}.{attr}: FAILED - {e}")


def find_problematic_chunks(chunks):
    """
    Find which specific chunks are causing the serialization issue
    """
    print(f"Testing all {len(chunks)} chunks for serialization issues...")
    
    problematic_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
        # Test if the chunk can be pickled
            pickle.dumps(chunk)
            # logging.debug(f"Chunk {i}: Pickle OK")
        except Exception as e:
            logging.error(f"Chunk {i}: Pickle FAILED - {e}")
        
        try:
            # Test deep copy (this is what Qdrant does internally)
            copy.deepcopy(chunk)
            if i % 50 == 0:  # Progress indicator
                print(f"Tested chunk {i}/{len(chunks)} - OK")
        except Exception as e:
            print(f"PROBLEM FOUND: Chunk {i} failed deep copy: {e}")
            problematic_chunks.append(i)
            
            # Get more details about this chunk
            print(f"Chunk {i} details:")
            print(f"  Type: {type(chunk)}")
            print(f"  Has id: {hasattr(chunk, 'id')} - Value: {getattr(chunk, 'id', 'N/A')}")
            print(f"  Has metadata: {hasattr(chunk, 'metadata')}")
            
            if hasattr(chunk, 'metadata'):
                print(f"  Metadata keys: {list(chunk.metadata.keys()) if chunk.metadata else 'None'}")
                # Test each metadata field
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        try:
                            copy.deepcopy(value)
                        except Exception as meta_e:
                            print(f"    Metadata field '{key}' is problematic: {meta_e}")
            
            print(f"  Content preview: {str(chunk)[:100]}...")
            print("-" * 50)
    
    print(f"Found {len(problematic_chunks)} problematic chunks: {problematic_chunks}")
    return problematic_chunks

def test_embeddings_serial(model):
    logging.info("2. Testing embedding model serialization...")
    try:
        copy.deepcopy(embedding_model)
        logging.info("✓ Embedding model: OK")
    except Exception as e:
        logging.error(f"✗ Embedding model: FAILED - {e}")
        logging.error("This is likely the problem!")
        
        # Try to identify what's in the embedding model
        logging.debug(f"Embedding model type: {type(embedding_model)}")
        logging.debug(f"Embedding model attributes: {dir(embedding_model)}")
        
        # Check if it has device info or file handles
        if hasattr(embedding_model, 'device'):
            logging.debug(f"Device: {embedding_model.device}")
        if hasattr(embedding_model, '_modules'):
            logging.debug("Has _modules (PyTorch model)")
        if hasattr(embedding_model, '_target_device'):
            logging.debug(f"Target device: {embedding_model._target_device}")
            
        return False
    



if __name__ == "__main__":
    # problematic_chunks = find_problematic_chunks(chunks)
    
    # Optionally, you can save the problematic chunks for further analysis
    # with open("problematic_chunks.pkl", "wb") as f:
    #     pickle.dump([chunks[i] for i in problematic_chunks], f)
    #     logging.info(f"Saved problematic chunks to 'problematic_chunks.pkl'.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    # problematic_model = test_embeddings_serial(embedding_model)
    # exit()
    client = QdrantClient(":memory:")  # Use in-memory storage for testing; for production use persistent storage
    try:
        # Test if the client can connect
        # client.get_collections()
        # logging.info("Qdrant client connected successfully.")
        vectorstore = Qdrant.from_documents(
            documents=chunks,
            embedding=embedding_model,
            location=":memory:",  # This is the key - uses in-memory storage
            collection_name="simphy_docs"  ,
            # location=os.path.join(os.path.dirname(os.path.abspath(__file__)),"qdrant_data/"),  # Use persistent storage for production
            # collection_name="simphy_guide",
            # client=client
        )
    except Exception as e:
        print(f"Still failing: {e}")
        traceback.print_exc()

    logging.info("Vector store created successfully.")

    logging.info("testing retrieval...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents
    # You can change the query to test different questions
    
    query = "abstract Vector2" # polygon dena hota hai
    docs = retriever.get_relevant_documents(query)

    for d in docs:
        print("---")
        print(d.page_content[:300])