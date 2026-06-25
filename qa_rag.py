import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import config
from config import PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL

# Load vectorstore and retriever
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Groq client (friendly exit instead of a traceback if the key is missing)
try:
    client = config.get_groq_client()
except RuntimeError as e:
    print(f"\n[config] {e}")
    sys.exit(1)

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
response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[{"role": "user", "content": prompt}]
).choices[0].message.content

# Display result
print("\n Answer:\n", response)
