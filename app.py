# app.py
import streamlit as st
import os
from backend.rag_pipeline import RAGPipeline

# --- Setup and Ingestion (Only runs if needed) ---
# This part of the code ensures the vector store exists before the main app loads.
# It's the key to solving the deployment crash.

DATA_PATH = 'data/Medical_book.pdf'
VECTOR_STORE_PATH = 'vector_store/faiss_index'

def check_and_create_vector_store():
    """Checks if the vector store exists, and if not, creates it."""
    if not os.path.exists(VECTOR_STORE_PATH):
        st.info("Vector store not found. Starting one-time setup...")
        # Display a spinner and status during the long process
        with st.spinner("Processing PDF and creating vector store. This may take a few minutes..."):
            from backend.ingest import create_vector_store # Import the function
            create_vector_store()
        st.success("Setup complete! The application is now ready.")
        # Rerun the app to proceed to the main interface
        st.rerun()

# Run the setup check at the very beginning
check_and_create_vector_store()


# --- Main Application Logic ---

@st.cache_resource
def load_pipeline():
    """Loads the RAG pipeline. This will only run after the vector store is confirmed to exist."""
    return RAGPipeline()

# Set page configuration
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# --- Page Title and Description ---
st.title("Medical Chatbot 🩺")
st.markdown(
    "This chatbot uses a local AI model to answer questions based on a medical document. "
    "Your data remains private and is processed on your machine."
)
st.divider()

# --- Load Pipeline ---
with st.spinner("Initializing models..."):
    rag_pipeline = load_pipeline()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the medical text"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_pipeline.ask(prompt)
            answer = response["answer"]
            st.markdown(answer)
            
            with st.expander("Debugger View: See Retrieved Context"):
                st.write("The model based its answer on the following text chunks:")
                if response["retrieved_docs"]:
                    for i, doc in enumerate(response["retrieved_docs"]):
                        st.info(f"**Chunk {i+1} (Source: {doc.metadata.get('source')})**")
                        st.text(doc.page_content)
                        st.divider()
                else:
                    st.write("No relevant documents were retrieved.")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: grey;"><p>Built By Anand_K and Rishab_T</p></div>
    """,
    unsafe_allow_html=True
)