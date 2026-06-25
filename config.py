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
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# User-facing message reused by both the API and the UI.
MISSING_KEY_MESSAGE = (
    "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
    "and set it as an environment variable (or add it to a .env file). "
    "See the README → LLM Provider section for per-OS instructions."
)


def is_configured() -> bool:
    """True if the minimum required configuration (the Groq key) is present."""
    return bool(GROQ_API_KEY)


def config_status() -> dict:
    """Non-secret config snapshot for /health and debugging."""
    return {
        "groq_api_key_set": bool(GROQ_API_KEY),
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL or "(groq default)",
        "embedding_model": EMBEDDING_MODEL,
    }


def get_groq_client():
    """Return a configured Groq client, or raise RuntimeError with a clear,
    message-only error (no secret, no stack-trace-worthy internals)."""
    if not GROQ_API_KEY:
        raise RuntimeError(MISSING_KEY_MESSAGE)
    from groq import Groq

    kwargs = {"api_key": GROQ_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    return Groq(**kwargs)
