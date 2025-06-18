from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Constants
PERSIST_DIR = "db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

# Load vectorstore and retriever
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Load local LLM
llm = Ollama(model=LLM_MODEL)

# Ask a question
query = "Who is finaceindia?"
docs = retriever.get_relevant_documents(query)

# Create the prompt
context = "\n\n".join(doc.page_content for doc in docs)
prompt = f"""Use the following context to answer the question:
{context}

Question: {query}
Answer:"""

# Call the model
response = llm.invoke(prompt)

# Display result
print("\n Answer:\n", response)
