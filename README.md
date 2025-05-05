# rag_initio

Simple Retrieval-Augmented Generation (RAG) with Langchain and Groq. Answers questions based on `sample_knowledge.txt`.

**Key Features:**

* Loads knowledge from `sample_knowledge.txt`.
* Chunks text and generates embeddings using Sentence Transformers.
* Stores embeddings in Chroma vector database.
* Retrieves top 3 relevant chunks for queries.
* Uses Groq Llama 3 model (`llama-3-70b-versatile`) via Langchain for answer generation.
* Employs RetrievalQA chain for question answering.
* Secure API key handling via `GROQ_API_KEY` environment variable.
* Prints query, answer, and source documents.

**Usage:**

1.  Clone the repository (if applicable).
2.  Install dependencies:
    ```bash
    pip install -U langchain langchain-community langchain-groq sentence-transformers chromadb
    ```
3.  Set your Groq API key:
    ```bash
    export GROQ_API_KEY="your_groq_api_key_here"
    ```
    (Replace with your actual key.)



 # e.g., python rag_example.py
 **Install Packages:**
```bash
pip install -U langchain langchain-community langchain-groq sentence-transformers chromadb
```
