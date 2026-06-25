# RAG-based AI Assistant

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot assistant, offering users real-time document querying, web-enhanced search, and file-based AI Q&A.

Built with:
- FastAPI for backend API
- Streamlit for the web UI
- LangChain for chaining LLMs with vector stores
- HuggingFace Embeddings + ChromaDB for document retrieval
- Groq Cloud API for fast LLM inference (no local GPU required)

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
git clone https://github.com/satyamshivam13/PDF_RAG_Chatbot.git
cd PDF_RAG_Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key

This app calls the **Groq Cloud API** for LLM inference, so you need a free API key
from [console.groq.com/keys](https://console.groq.com/keys). Set it as an environment
variable before running (see the [LLM Provider](#llm-provider) section for per-OS steps).

### 5. (Optional) Pre-load the sample document

```bash
python vectorstore.py
```

### 6. Run the FastAPI backend

```bash
uvicorn rag_api:app --reload --port 8000
```

### 7. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## LLM Provider

This project uses the **Groq Cloud API** for LLM inference — there is **no Ollama, no local
GPU, and no model download** required.

- **Provider:** Groq Cloud (OpenAI-compatible chat completions)
- **Default model:** `llama-3.1-8b-instant`
- **Embeddings:** run locally via `sentence-transformers/all-MiniLM-L6-v2` (CPU, ~90 MB)
- **Why Groq:** very low latency (ideal for token streaming) and a generous free tier, so the
  app stays fully cloud-deployable without any local model hosting.

### Required environment variable

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key from [console.groq.com/keys](https://console.groq.com/keys) |
| `LLM_MODEL` | No | `llama-3.1-8b-instant` | Override the Groq chat model |
| `LLM_BASE_URL` | No | Groq default | Point at any OpenAI-compatible endpoint |

### Setting `GROQ_API_KEY`

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY = "your_api_key_here"
```

**macOS / Linux (bash/zsh):**
```bash
export GROQ_API_KEY="your_api_key_here"
```

Or create a `.env` file in the project root (it is git-ignored):
```env
GROQ_API_KEY=your_api_key_here
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
- [Groq](https://groq.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
