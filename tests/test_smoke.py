"""Lightweight smoke tests for the PDF RAG Chatbot.

These validate that the app's structure, configuration, and core data layer are
sound — without heavy integration (no LLM calls, no embedding-model downloads).
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ["config.py", "rag_api.py", "streamlit_app.py", "qa_rag.py", "vectorstore.py"]


def test_all_modules_byte_compile():
    """Every Python module parses/compiles (catches syntax & import-position errors)."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(ROOT / m) for m in MODULES]],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_imports_with_sane_defaults():
    import config

    assert config.LLM_MODEL  # has a non-empty default
    assert config.EMBEDDING_MODEL.startswith("sentence-transformers/")
    assert isinstance(config.CHUNK_SIZE, int) and config.CHUNK_SIZE > 0
    assert isinstance(config.CHUNK_OVERLAP, int) and config.CHUNK_OVERLAP >= 0


def test_environment_validation_without_key(monkeypatch):
    import config

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert config.is_configured() is False
    with pytest.raises(RuntimeError):
        config.get_groq_client()


def test_environment_validation_with_key(monkeypatch):
    import config

    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    assert config.is_configured() is True


def test_config_status_has_expected_shape_and_no_secret(monkeypatch):
    import config

    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
    status = config.config_status()
    for key in ("groq_api_key_set", "llm_model", "llm_base_url", "embedding_model"):
        assert key in status
    # The status snapshot must never leak the actual key value.
    assert "super-secret-value" not in str(status)
    assert status["groq_api_key_set"] is True


def test_llm_model_override(monkeypatch):
    import config

    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    assert config.config_status()["llm_model"] == "llama-3.3-70b-versatile"


def test_vector_database_initializes(tmp_path):
    """ChromaDB persistent client initializes and stores/counts a document."""
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.get_or_create_collection("smoke")
    collection.add(ids=["1"], documents=["hello world"])
    assert collection.count() == 1
