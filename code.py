import os
from typing import List, Mapping, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.chat_models import ChatGroq
from langchain_groq import ChatGroq
# Import ChatGroq
# from langchain_groq import ChatGroq

# 1. Load Documents
with open("sample_knowledge.txt", "w") as f:
    f.write("""
    Faisalabad is a major industrial city in Pakistan.
    It is located in the Punjab province.
    The city is known for its textile industry and is often called the "Manchester of Pakistan".
    The clock tower in Faisalabad is a famous landmark.
    Agriculture also plays a significant role in the economy of the surrounding region.
    The current time in Faisalabad is Thursday, April 17, 2025 at 11:09 PM PKT.
    """)

loader = TextLoader("sample_knowledge.txt")
documents = loader.load()

# 2. Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 3. Create Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# 4. Create Vector Store
db = Chroma.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 5. Initialize Groq Chat Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=1024,
    api_key=os.environ["GROQ_API_KEY"], # Use the environment variable
)

# 6. Create Retrieval QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Query the RAG Application
query = "this city is manchester?"
result = qa({"query": query})

print("Query:", result["query"])
print("Answer:", result["result"])
if "source_documents" in result:
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(doc.page_content)

