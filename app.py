import streamlit as st
import os
from backend.rag_pipeline import RAGPipeline

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* General body styling */
body {
    color: #fff;
    background-color: #111;
}

/* Sidebar styling */
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(0deg, rgba(20,20,30,1) 0%, rgba(30,30,45,1) 100%);
    border-right: 2px solid #3c3c3c;
}

/* Custom styling for the "Clear Cache" button in the sidebar */
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(45deg, #222 0%, #000 100%);
    color: white;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
    width: 100%; /* Make button fill sidebar width */
}

[data-testid="stSidebar"] .stButton button:hover {
    background: linear-gradient(45deg, #FF4B2B 0%, #FF416C 100%);
    border-color: #FF4B2B;
    box-shadow: 0 0 15px rgba(255, 75, 43, 0.5);
    transform: scale(1.02);
}

/* Style for the main description box */
.description-box {
    background-color: #1e1e2f;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #3a3a5a;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}

.description-box h1 {
    color: #e0e0ff;
}

.description-box p {
    color: #b0b0d0;
    line-height: 1.6;
}

/* Footer styling */
.footer {
    text-align: center;
    color: grey;
    font-size: 14px;
    padding-top: 20px;
}
.footer a {
    color: #FF4B2B; /* A color that matches the button hover */
    text-decoration: none;
    font-weight: bold;
}
.footer a:hover {
    text-decoration: underline;
    color: #FF416C;
}
</style>
""", unsafe_allow_html=True)

# (The check_and_create_vector_store function remains the same)
# ...

@st.cache_resource
def load_pipeline():
    """Loads the RAG pipeline. This will only run after the vector store is confirmed to exist."""
    return RAGPipeline()

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    st.markdown(
        "If the app becomes slow, use this button to reset the model and clear the chat."
    )
    if st.button("Clear Cache & Reset"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

# --- Page Title and Description ---
st.markdown("""
<div class="description-box">
    <h1>Medical Chatbot 🩺</h1>
    <p>This is a private, locally-hosted medical question-answering chatbot. Ask any question based on the loaded medical document, and the AI will provide an answer using a local RAG pipeline. Your data remains 100% private.</p>
</div>
""", unsafe_allow_html=True)


# --- Load Pipeline ---
# The check for the vector store needs to happen before loading the pipeline.
VECTOR_STORE_PATH = "vector_store/faiss_index"
if not os.path.exists(VECTOR_STORE_PATH):
    with st.spinner("First-time setup: Creating vector store from PDF... This may take a moment."):
        st.error("Vector store not found. Please run the `ingest.py` script first to process your data.")
        st.stop()
else:
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
    <div class="footer">
        <p>Built By <a href="https://github.com/anandkaman" target="_blank">Anand K</a> and Rishab T</p>
    </div>
    """,
    unsafe_allow_html=True
)
