# Technical Documentation: Medical Chatbot

This document provides a deep dive into the technical architecture, components, and workflow of the Medical Chatbot. It is intended for developers who want to understand, modify, or extend the project.

## 1. Project Architecture Overview

The chatbot is built on a **Retrieval Augmented Generation (RAG)** pipeline. This architecture allows a Large Language Model (LLM) to answer questions based on a specific knowledge base (in this case, a medical PDF) instead of relying solely on its pre-trained knowledge.

The process is entirely local and private, using open source models from Hugging Face and a local vector store.

The workflow is divided into two main stages:
1.  **Ingestion (Indexing)**: A one time process where the source PDF is read, processed, and converted into a searchable vector index using FAISS. This is handled by `ingest.py`.
2.  **Inference (Question Answering)**: The real-time process where a user's query is used to retrieve relevant information from the vector index, which is then fed to an LLM to generate a human-like answer. This is managed by `rag_pipeline.py` and served via the `app.py` Streamlit interface.

## 2. Core Components

The project is broken down into several key files, each with a specific responsibility.

### `ingest.py` - Data Processing and Indexing

This script is the starting point. Its sole purpose is to prepare the knowledge base.
*   **PDF Loading**: It uses `PyPDFLoader` to load the `Medical_book.pdf` from the `/data` directory.
*   **Text Splitting**: The loaded text is too large to be fed into a model at once. `RecursiveCharacterTextSplitter` breaks the text into smaller, manageable "chunks." This is a critical step, as the quality of these chunks directly impacts retrieval accuracy.
*   **Embedding Generation**: Each text chunk is converted into a numerical vector using the `HuggingFaceEmbeddings` model (`sentence-transformers/all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.
*   **Vector Store Creation**: The script uses **FAISS (Facebook AI Similarity Search)** to create a local vector store. It takes all the text chunks and their corresponding embeddings and builds a searchable index. This index is saved to the `vector_store/faiss_index` directory.

### `rag_pipeline.py` - The Core RAG Logic

This module contains the `RAGPipeline` class, which orchestrates the question-answering process.
*   **Initialization**: When `RAGPipeline` is created, it loads the pre-built FAISS index from disk and initializes the two core models:
    1.  **Embedding Model**: The same `sentence-transformers` model used during ingestion. This is required to convert the user's incoming question into a vector.
    2.  **Generative Model**: A `google/flan-t5-base` model wrapped in the `HuggingFacePipeline`. This model is responsible for generating the final answer.
*   **The `ask` Method**: This is the central function. When a user asks a question:
    1.  The FAISS index performs a **similarity search** to find the text chunks whose embeddings are most similar to the user's question embedding. These are the most relevant documents.
    2.  A **prompt** is dynamically constructed using a template. This prompt includes the retrieved text chunks (the "context") and the original user question.
    3.  This combined prompt is passed to the Flan-T5 model, which **synthesizes** an answer based *only* on the context provided.
    4.  The final dictionary contains the generated answer and the source documents used, which are displayed in the "Debugger View."

### `app.py` - The User Interface

This script creates the web-based user interface using Streamlit.
*   **UI and Styling**: It sets up the page layout, title, sidebar, and all custom CSS for a polished look and feel.
*   **Vector Store Check**: On startup, it checks if the FAISS index exists. If not, it instructs the user to run the ingestion script, preventing errors.
*   **Model Loading**: It uses `@st.cache_resource` to load the `RAGPipeline` once and keep it in memory, avoiding slow re-initialization on every user interaction.
*   **Chat State Management**: `st.session_state` is used to store and persist the conversation history, allowing the chat to feel continuous.
*   **User Interaction**: It captures user input, passes it to the `rag_pipeline.ask()` method, and displays the response in a chat-like format.

## 3. Models and Pipeline Optimization

### Models
*   **`sentence-transformers/all-MiniLM-L6-v2`** (Embedding Model): This is a lightweight and fast model designed to create high-quality sentence and text embeddings. Its small size makes it ideal for running locally on a CPU.
*   **`google/flan-t5-base`** (Generative Model): This is an instruction-tuned model, which makes it effective at following prompts and performing tasks like question-answering based on a given context. The "base" version provides a good balance between performance and resource requirements.

### Model Optimization and Configuration
*   **Context Size**: The `flan-t5-base` model has a limited context window (the amount of text it can consider at once). In `rag_pipeline.py`, the retriever is configured to fetch the top `k=5` most relevant chunks. This number is a trade-off: too few chunks might miss important context, while too many could exceed the model's context window or introduce noise.
*   **Chunking Strategy**: The `RecursiveCharacterTextSplitter` in `ingest.py` is configured with `chunk_size=500` and `chunk_overlap=50`. The overlap ensures that sentences or ideas are not awkwardly split between two chunks, which improves the chances of retrieving a complete context.
*   **Synthesis vs. Extraction**: The model doesn't just find and return text. It *synthesizes* an answer. This means it rewrites the information from the retrieved chunks into a coherent, new sentence that directly answers the user's question, which is a key strength of the RAG approach.

## 4. Application Walkthrough

The interface provides a simple and intuitive chat experience. The sidebar contains controls, while the main area hosts the conversation. The "Debugger View" is a crucial feature for understanding the model's reasoning, as it shows the exact text chunks the AI used to formulate its answer.

[1]

## 5. Scalability and Flexibility

The project is designed to be modular and adaptable.
*   **Using Your Own Data**: The entire chatbot can be repurposed for any subject by simply replacing the `Medical_book.pdf` file in the `/data` folder with another PDF. After replacing the file, delete the old `/vector_store` directory and run `python backend/ingest.py` again to create a new knowledge base. The app will then answer questions based on your new document.
*   **Changing Models**: The embedding and generative models can be easily swapped. To use a different model, change the `model_name` variable in `rag_pipeline.py` (and `ingest.py` for the embedding model) to another model identifier from the Hugging Face Hub. Note that more powerful models may require more RAM and a GPU for acceptable performance.
*   **Scaling Up**: For extremely large documents or high-traffic applications, the local FAISS index could become a bottleneck. A future step for scalability would be to replace the local index with a dedicated vector database server like **Pinecone**, **Weaviate**, or **ChromaDB**. This would offload the storage and search operations to a more robust and scalable service.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/94090686/bdd688bb-ee83-4352-81c2-f145d19d3a02/Streamlit.pdf)
[2](https://github.com/anandkaman/Med_Chatbot)
