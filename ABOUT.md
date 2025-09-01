# About This Project

### The Idea
This project started with a simple but powerful goal: to build a medical AI assistant that was completely private, reliable, and free from external dependencies. In a world of API-first AI, we wanted to prove that a sophisticated, document-aware chatbot could run entirely on a local machine, using only open-source tools. The idea was to empower users with an intelligent tool to query their own documents without ever sending their data to the cloud.

### The Journey & The Challenges

Our path from a Jupyter Notebook to a full-fledged web application was a classic developer journey filled with debugging and refinement. The key challenges we tackled were:

1.  **Moving Beyond APIs**: Our first hurdle was replacing cloud-based services like Pinecone. We chose **FAISS** as our vector store, which brought a new challenge: managing the index files locally and making the application portable.

2.  **The Context Window Conundrum**: Early on, we constantly hit the dreaded `Token indices sequence length is longer than the specified maximum sequence length` error. Our LLM (`google/flan-t5-base`) has a 512-token limit, and our initial approach of "stuffing" documents was too naive.

3.  **Taming the `map_reduce` Chain**: Our solution was a **`map_reduce`** pipeline, but even that failed initially. The model returned garbled text or claimed it was "not sure" even when the context was correct. Through careful prompt engineering, we transformed our prompts from simple "summarize" commands to directive "extract relevant facts" commands. This was the turning point for getting accurate answers.

4.  **Solving the Blank Screen of Death**: When we first tried to deploy our Streamlit app, we were met with a blank black screen. We diagnosed this as a startup crash caused by the app trying to load a non-existent vector store in a fresh deployment environment. We solved this by building a **self-healing startup sequence** into `app.py` that detects if the index is missing and runs the ingestion process automatically.

### What We Built
The result is a robust, self-contained RAG application that is not just a proof-of-concept, but a deployable and user-friendly tool. It demonstrates a practical blueprint for creating private, powerful AI applications with the incredible ecosystem of open-source software.
