"""Centralized, env-overridable configuration for docintel-ai.

Every tunable that used to live as a hardcoded literal (chunk sizing, retriever
parameters, the LLM model/temperature, and the embedding model name) is defined
here in one place. Values are read from the environment with the ``DOCINTEL_``
prefix (also honoring ``backend/.env``), so nothing needs a code change to
retune:

    DOCINTEL_CHUNK_SIZE=1000 DOCINTEL_LLM_TEMPERATURE=0.1 uvicorn app.main:app

Import the shared ``settings`` singleton instead of hardcoding these values.

Note: router retrieval tunables (``HYBRID_RAG_TOP_K`` /
``HYBRID_RAG_SIMILARITY_THRESHOLD``) intentionally remain in ``chains/router.py``
— they are part of the routing/threshold logic, not general config hygiene.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ``backend/`` root. This file is ``backend/app/core/config.py`` so parents[2]
# resolves to ``backend/`` regardless of the process working directory — the
# ``.env`` and ``storage/`` paths below are anchored to it, not to the CWD.
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Text splitting (ingestion/text_splitter.py)
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retriever / MMR search (retrievers/retriever.py)
    retriever_k: int = 4  # final number of chunks returned to the LLM
    retriever_fetch_k: int = 10  # candidates fetched before MMR reranking
    retriever_lambda_mult: float = 0.7  # MMR relevance vs diversity (higher = more relevance)

    # LLM (models/llm_model.py)
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3

    # Embeddings (ingestion/embeddings.py + vector_store.py log sites)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # -- API server (FastAPI backend) --------------------------------------------
    # Root directory for uploaded PDFs and persisted FAISS indexes.
    storage_dir: str = str(BACKEND_DIR / "storage")
    # Comma-separated CORS origins allowed to call the API (frontend dev servers).
    cors_allow_origins: str = "http://localhost:5173"

    @property
    def cors_origins(self) -> list[str]:
        """CORS origins as a list (parsed from the comma-separated setting)."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def uploads_dir(self) -> Path:
        return Path(self.storage_dir) / "uploads"

    @property
    def vector_stores_dir(self) -> Path:
        return Path(self.storage_dir) / "vector_stores"


settings = Settings()
