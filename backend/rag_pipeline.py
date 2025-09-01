# backend/rag_pipeline.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate

# Define paths and model names
VECTOR_STORE_PATH = 'vector_store/faiss_index'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.retriever = self._setup_retriever()
        self.llm = self._setup_llm()
        self.chain = self._setup_chain()
        print("RAG Pipeline initialized successfully.")

    def _setup_retriever(self):
        """Initializes the retriever from the local FAISS index."""
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def _setup_llm(self):
        """Initializes the local Large Language Model."""
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        return HuggingFacePipeline(pipeline=pipe)

    # In backend/rag_pipeline.py

# ... (rest of the class definition and other methods are the same)

    # In backend/rag_pipeline.py

    def _setup_chain(self):
        """Sets up the Map-Reduce RAG chain with more concise prompts."""
        
        # --- MORE CONCISE MAP PROMPT ---
        map_template = """
        Extract any sentences from the text below that mention treatments, medications, or remedies for the user's question. If none, say 'No relevant information found.'
        Text: '{text}'
        """
        map_prompt = PromptTemplate.from_template(map_template)
        
        # --- MORE CONCISE COMBINE PROMPT ---
        combine_template = (
            "You are a medical assistant. Answer the user's question using ONLY the extracted information below. "
            "If it says 'No relevant information found,' then you MUST say 'I am not sure based on the available information.' "
            "List any specific treatments or medications found.\n\n"
            "Information:\n{text}\n\nQuestion: {input}\n\nAnswer:"
        )
        combine_prompt = PromptTemplate.from_template(combine_template)

        return load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            combine_document_variable_name="text",
            verbose=False
        )

# ... (rest of the class is the same)

    # In backend/rag_pipeline.py

# ... (rest of the class definition is the same)

    def ask(self, question: str) -> dict:
        """Asks a question to the RAG pipeline and returns answer and debug info."""
        print(f"Retrieving documents for: {question}")
        retrieved_docs = self.retriever.invoke(question)
        
        if not retrieved_docs:
            return {
                "answer": "I could not find any relevant information for your question.",
                "retrieved_docs": []
            }
        
        print("Generating answer from retrieved documents...")
        response = self.chain.invoke({
            "input_documents": retrieved_docs,
            "input": question
        })
        
        return {
            "answer": response.get("output_text", "Sorry, I had trouble generating a response."),
            "retrieved_docs": retrieved_docs
        }
