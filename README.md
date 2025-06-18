# RAG-based AI Assistant

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot assistant, offering users real-time document querying, web-enhanced search, and file-based AI Q&A.

Built with:
- FastAPI for backend API
- Streamlit for the web UI
- LangChain for chaining LLMs with vector stores
- HuggingFace Embeddings + ChromaDB for document retrieval
- Ollama for local LLM inference (Mistral/phi)

---

## Features

-  Upload `.txt`, `.pdf`, or `.json` files for Q&A
-  Ask questions and get answers powered by documents
-  Optional DuckDuckGo web search integration
-  Chat memory with conversational context
-  Real-time LLM streaming (FastAPI/Streamlit)
-  Persistent vector store with duplicate file hashing

---

## Project Structure

```
├── rag_api.py          # FastAPI backend for streaming and upload
├── streamlit_app.py    # Streamlit-based UI interface
├── index.html          # Optional web UI (vanilla JS-based)
├── vectorstore.py      # CLI tool to pre-process and store documents
├── db/                 # ChromaDB persistent storage
├── demo.txt            # (Optional) Sample document
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/satyamshivam13/AI_Chatbot.git
cd AI_Chatbot
```

### 2. Create a Virtual Enviroment 

```bash
python -m venv venv
venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** You’ll need `ollama` installed and a model like `mistral` or `phi` downloaded locally.

### 3. Run Vectorstor

```bash
python vectorstore.py
```

### 4. Run API

```bash
uvicorn rag_api:app --reload --port 8000
```

### 5. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## File Upload & Search

- Upload documents via Streamlit UI or API (`/upload`)
- Ask questions via Streamlit or `index.html` chat UI
- Enable web search for fallback answers via DuckDuckGo

---

## License

This project is licensed under the MIT License - see [LICENSE](./LICENSE) for details.

---

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
