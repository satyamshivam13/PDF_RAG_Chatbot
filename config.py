"""Central, environment-driven configuration for the PDF RAG Chatbot.

Importing this module never raises and never requires any environment variable,
so the app can always start. Validation of GROQ_API_KEY happens lazily, only when
the LLM is actually needed, so callers can surface a friendly error instead of an
import-time crash / stack trace.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # load a local .env if present (no-op otherwise)
except Exception:
    # python-dotenv is optional at runtime; never let config import fail.
    pass

# ── LLM (Groq Cloud, OpenAI-compatible) ────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")  # optional OpenAI-compatible endpoint

# ── Retrieval / embeddings ─────────────────────────────────────────────
PERSIST_DIR = os.getenv("PERSIST_DIR", "db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Larger chunks + higher top-k give the LLM more grounding context than the original
# 200/50/k=2 defaults, which were too small for real PDFs. All are env-tunable.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))

# User-facing message reused by both the API and the UI.
MISSING_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
    "and set it as an environment variable (or add it to a .env file). "
    "See the README → LLM Provider section for per-OS instructions."
)


def _groq_key() -> str | None:
    """Read the key live so a secrets->env bridge (e.g. Streamlit Cloud) set after
    import is still respected."""
    return os.getenv("GROQ_API_KEY")


def is_configured() -> bool:
    """True if the minimum required configuration (the Groq key) is present."""
    return bool(_groq_key())


def config_status() -> dict:
    """Non-secret config snapshot for /health and debugging."""
    return {
        "groq_api_key_set": bool(_groq_key()),
        "llm_model": os.getenv("LLM_MODEL", LLM_MODEL),
        "llm_base_url": os.getenv("LLM_BASE_URL") or "(groq default)",
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "top_k": TOP_K,
    }


def get_groq_client():
    """Return a configured Groq client, or raise RuntimeError with a clear,
    message-only error (no secret, no stack-trace-worthy internals)."""
    key = _groq_key()
    if not key:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    from groq import Groq

    kwargs = {"api_key": key}
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return Groq(**kwargs)
