# app.py
import streamlit as st
import os
from backend.rag_pipeline import RAGPipeline

# (The check_and_create_vector_store function remains the same)
# ...

@st.cache_resource
def load_pipeline():
    """Loads the RAG pipeline. This will only run after the vector store is confirmed to exist."""
    return RAGPipeline()

# Set page configuration
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")
    st.markdown(
        "If the app becomes slow or unresponsive after many questions, "
        "use this button to reset the AI model and clear the conversation."
    )
    if st.button("Clear Cache and Reset Conversation"):
        # Clear the model cache
        st.cache_resource.clear()
        # Clear the chat history
        st.session_state.clear()
        # Rerun the app to re-initialize everything
        st.rerun()

# --- Page Title and Description ---
st.title("Medical Chatbot 🩺")
# ... (rest of the title/description is the same)

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
