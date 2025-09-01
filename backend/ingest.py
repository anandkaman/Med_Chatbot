# backend/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define paths
DATA_PATH = 'data/Medical_book.pdf'
VECTOR_STORE_PATH = 'vector_store/faiss_index'

def create_vector_store():
    """Loads PDF, splits it into chunks, and creates a FAISS vector store."""
    print("Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    
    print("Splitting text into chunks...")
    # We use a smaller chunk_size to avoid context window overflow
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    print("Creating and saving FAISS vector store...")
    if not os.path.exists('vector_store'):
        os.makedirs('vector_store')
        
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_STORE_PATH)
    
    print(f"Vector store created and saved at {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    create_vector_store()