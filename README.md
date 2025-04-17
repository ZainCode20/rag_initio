# rag_initio
Simple Retrieval-Augmented Generation (RAG) with Langchain and Groq
This Python script demonstrates a basic Retrieval-Augmented Generation (RAG) application using the Langchain framework and the Groq Llama 3 language model. It answers questions based on a small, provided knowledge base.

**Knowledge Loading:** 
Loads text-based knowledge from a ** sample_knowledge.txt** file.
Text Chunking: Splits the loaded text into smaller, manageable chunks for better retrieval.
Embedding Generation: Uses Sentence Transformers to create vector embeddings of the text chunks, capturing their semantic meaning.
Vector Storage: Stores the embeddings in a Chroma vector database for efficient similarity search.
Retrieval: Retrieves the top 3 most relevant text chunks from the Chroma database based on the user's query.
Language Model Integration: Leverages the Groq llama-3.3-70b-versatile chat model via Langchain's ChatGroq integration to generate answers.
**Question Answering:**
Uses a RetrievalQA chain to combine the retrieval and generation steps, answering user queries based on the retrieved context.
Environment Variable for API Key: Securely uses an environment variable (GROQ_API_KEY) to access the Groq API.
Output: Prints the original query, the generated answer, and the source documents used to formulate the answer.
**How to Use:**

Clone this repository (if applicable).
Install the required Python packages:
Bash

pip install -U langchain langchain-community langchain-groq sentence-transformers chromadb
Set your Groq API key as an environment variable:
Bash
## you can usw openai model or any other llm
export GROQ_API_KEY="your_groq_api_key_here"
(Replace "your_groq_api_key_here" with your actual Groq API key.)
Run the Python script:
Bash

python your_script_name.py
 # e.g., python rag_example.py
