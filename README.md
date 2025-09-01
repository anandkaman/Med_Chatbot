# Medical Chatbot with Local RAG Pipeline

This project is a private, locally-hosted medical question-answering chatbot. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer user queries based on the content of a provided medical PDF. All models and data remain on your local machine, ensuring 100% privacy.

### Features
*   **Fully Local & Private**: No reliance on external APIs like OpenAI or Pinecone.
*   **Open-Source Models**: Powered by Hugging Face models (`google/flan-t5-base` for generation and `sentence-transformers/all-MiniLM-L6-v2` for embeddings).
*   **Efficient Vector Search**: Uses FAISS (Facebook AI Similarity Search) for fast, local document retrieval.
*   **User-Friendly Interface**: Built with Streamlit, providing an interactive chat experience with a built-in debugger view.
*   **Self-Contained & Deployable**: Automatically creates the necessary vector store on first run, making it easy to deploy.

### Project Structure
```bash
medical-chatbot/
├── backend/
│ ├── rag_pipeline.py # Core RAG logic
│ └── ingest.py # PDF processing and vector store creation
├── data/
│ └── Medical_book.pdf # Source document
├── vector_store/ # Generated FAISS index (ignored by Git)
├── app.py # The Streamlit web application
├── README.md # This file
├── ABOUT.md # Project origin and development story
└── requirements.txt # Python dependencies
```

### Getting Started

1.  **Clone the Repository**
    ```
    git clone <your_repo_url>
    cd medical-chatbot
    ```

2.  **Set Up Virtual Environment and Install Dependencies**
    ```
    # Create a virtual environment
    python -m venv venv
    # Activate it (Windows)
    .\venv\Scripts\activate
    # Install required packages
    pip install -r requirements.txt
    ```

3.  **Add Your Data**
    Place your source PDF file inside the `data/` directory.

4.  **Run the Application**
    The app will automatically create the vector store on its first run.
    ```
    streamlit run app.py
    ```

