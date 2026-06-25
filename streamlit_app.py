import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import json
import fitz  # PyMuPDF
import hashlib
import time
import sys

# Bridge Streamlit Cloud secrets -> environment so the env-based config sees them.
# (Streamlit exposes secrets via st.secrets, not reliably via os.environ.)
try:
    for _k in ("GROQ_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml locally is fine; env vars / .env still work

import config
from config import CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_DIR, EMBEDDING_MODEL, LLM_MODEL, TOP_K

# Session state for modal
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

def toggle_modal():
    st.session_state.show_modal = not st.session_state.show_modal

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
sys.modules["torch.classes"] = None

# Validate configuration with a friendly UI warning instead of a crash/traceback.
try:
    client = config.get_groq_client()
except RuntimeError as e:
    st.error(str(e))
    st.stop()


# UI Layout
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Finace India - AI Assistant")
with col2:
    st.button("Upload File ", on_click=toggle_modal)

# File upload modal
if st.session_state.show_modal:
    with st.container():
        st.markdown("### Upload a file (.txt, .pdf, .json)")
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "json"])

        if uploaded_file is not None:
            file_text = ""
            if uploaded_file.type == "application/pdf":
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                    file_text = "\n".join(page.get_text() for page in pdf)
            elif uploaded_file.type == "application/json":
                file_text = json.dumps(json.load(uploaded_file), indent=2)
            elif uploaded_file.type == "text/plain":
                file_text = uploaded_file.read().decode("utf-8")

            if file_text:
                file_hash = hashlib.md5(file_text.encode()).hexdigest()
                hash_path = os.path.join(PERSIST_DIR, f"hash_{file_hash}.done")

                if not os.path.exists(hash_path):
                    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                    chunks = splitter.split_documents([Document(page_content=file_text)])
                    vectorstore.add_documents(chunks)
                    vectorstore.persist()
                    with open(hash_path, "w") as f:
                        f.write("done")
                    st.success(" File embedded and stored successfully!")
                else:
                    st.warning(" File already processed. Skipping.")

# Query input
query = st.text_input("Ask me anything about your docs:")

if query:
    with st.spinner("Thinking..."):
        start_time = time.time()
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""Use the following context to answer the question:
{context}

Question: {query}
Answer:"""

        # STREAMING STARTS HERE
        placeholder = st.empty()
        partial_response = ""

        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                partial_response += content
                placeholder.markdown(partial_response + "▌")

        placeholder.markdown(partial_response)  # Finalize the response

        end_time = time.time()
        st.info(f"Response time: {round(end_time - start_time, 2)} seconds")