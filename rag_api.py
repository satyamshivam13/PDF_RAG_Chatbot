from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from duckduckgo_search import DDGS

import hashlib
import fitz  # PyMuPDF
import json
import os
import time
import warnings

import config
from config import (
    PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Web Scraping
def search_duckduckgo(query, max_results=10):
    with DDGS() as ddgs:
        results = ddgs.text(
            keywords=query,
            region="wt-wt",
            safesearch="moderate",
            max_results=max_results
        )
        return list(results)


# Initialize components (none of these require GROQ_API_KEY, so import never crashes)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Lazily-created Groq client (built on first use so a missing key never crashes import)
_client = None


def get_client():
    global _client
    if _client is None:
        _client = config.get_groq_client()  # raises RuntimeError with a clear message
    return _client


@app.on_event("startup")
def validate_configuration():
    """Log a clear, non-fatal warning if the app starts without a Groq key."""
    if not config.is_configured():
        print(f"[startup] WARNING: {config.MISSING_KEY_MESSAGE}")
    else:
        print(f"[startup] Groq configured — model={LLM_MODEL}")


class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "configured": config.is_configured(), **config.config_status()}

@app.post("/ask-stream")
def ask_question_stream(query: Query):
    def generate():
        # Validate configuration before doing any work; stream a clear message if unset.
        try:
            client = get_client()
        except RuntimeError as e:
            yield str(e)
            return

        # Retrieve document chunks using correct method
        retrieved_docs = retriever.get_relevant_documents(query.question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs[:2])
        
        # If no documents are retrieved, provide a default message
        if not retrieved_docs:
            context = "[We are still learning. No relevant information found in our database.]"
        else:
            context = "\n\n".join(doc.page_content for doc in retrieved_docs[:2])

        # Retrieve previous chat
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        # Build prompt
        prompt = f"""You are a helpful financial expert representing Finace India, a company that assists with financial and compliance services in India. Always frame your answers in the context of how Finace India can help solve the user's problem.

Using the following document context and chat history, answer the user's question in a clear and helpful way. Stick to 3-4 sentences maximum.

Document Context:
{context}

Previous Conversation:
{chat_history}

User: {query.question}
Assistant:"""

        # Stream the response from the LLM
        response = ""
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                response += content
                yield content

        # Save context to memory
        memory.save_context({"input": query.question}, {"output": response})

    # Use event-stream for real-time streaming to browser
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_text = ""

    if file.filename.endswith(".pdf"):
        with fitz.open(stream=await file.read(), filetype="pdf") as doc:
            file_text = "\n".join(page.get_text() for page in doc)

    elif file.filename.endswith(".json"):
        content = await file.read()
        file_text = json.dumps(json.loads(content), indent=2)

    elif file.filename.endswith(".txt"):
        file_text = (await file.read()).decode("utf-8")

    else:
        return {"error": "Unsupported file format."}

    # Check if file is already processed
    file_hash = hashlib.md5(file_text.encode()).hexdigest()
    hash_marker = os.path.join(PERSIST_DIR, f"hash_{file_hash}.done")

    if os.path.exists(hash_marker):
        return {"message": " File already embedded. Skipping."}

    # Split and store
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_documents([Document(page_content=file_text)])
    vectorstore.add_documents(docs)
    vectorstore.persist()

    with open(hash_marker, "w") as f:
        f.write("done")

    return {"message": " File embedded and stored successfully!"}

@app.post("/search")
async def search(request: Request):
    body = await request.json()
    query = body.get("query")

    if not query:
        return {"results": []}

    try:
        results = search_duckduckgo(query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}