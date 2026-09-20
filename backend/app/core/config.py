"""Centralized, env-overridable configuration for docintel-ai.

Every tunable that used to live as a hardcoded literal (chunk sizing, retriever
parameters, the LLM model/temperature, and the embedding model name) is defined
here in one place. Values are read from the environment with the ``DOCINTEL_``
prefix (also honoring ``backend/.env``), so nothing needs a code change to
retune:

    DOCINTEL_CHUNK_SIZE=1000 DOCINTEL_LLM_TEMPERATURE=0.1 uvicorn app.main:app

Import the shared ``settings`` singleton instead of hardcoding these values.

Note: the RAG-vs-LLM routing thresholds — the whole *evidence policy* (the
semantic ``HYBRID_RAG_SIMILARITY_THRESHOLD`` plus the lexical and cross-retriever
agreement knobs) — intentionally remain in ``chains/router.py`` as
``EvidencePolicy``: they are part of the routing *decision*, not general config
hygiene. The retrieval *structure* (candidate/final counts, RRF constant) lives
here as ``retrieval_candidates_k`` / ``final_context_k`` / ``rrf_k``.

The **agentic orchestration** knobs (``agentic_rag_enabled`` and friends) *do*
live here: they are structural switches for the orchestration layer
(``app/agents/``) — whether it runs at all, how many self-healing retrieval
attempts it may make, and the confidence guideline it uses — not per-signal
routing thresholds. See ``chains/chat_chain.py`` for where the switch is read.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
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

    # -- Hybrid retrieval (retrievers/hybrid_retriever.py) -----------------------
    # BM25 (keyword) + FAISS (vector) candidates fused with Reciprocal Rank
    # Fusion. Both retrievers fetch ``retrieval_candidates_k`` candidates; RRF
    # merges the two rankings and the top ``final_context_k`` chunks become the
    # RAG context. ``rrf_k`` is the RRF damping constant (higher = smaller gap
    # between ranks). The conventional bare env names (RETRIEVAL_CANDIDATES_K /
    # FINAL_CONTEXT_K / RRF_K) work alongside the DOCINTEL_-prefixed ones.
    retrieval_candidates_k: int = Field(
        default=20,
        validation_alias=AliasChoices("RETRIEVAL_CANDIDATES_K", "DOCINTEL_RETRIEVAL_CANDIDATES_K"),
    )
    final_context_k: int = Field(
        default=4,
        validation_alias=AliasChoices("FINAL_CONTEXT_K", "DOCINTEL_FINAL_CONTEXT_K"),
    )
    rrf_k: int = Field(
        default=60,
        validation_alias=AliasChoices("RRF_K", "DOCINTEL_RRF_K"),
    )

    # -- Agentic RAG orchestration (agents/orchestrator.py) ----------------------
    # A controlled orchestration layer *on top of* the hybrid retriever. It
    # understands the query, judges whether the retrieved evidence can actually
    # answer the ORIGINAL question, and self-heals by rewriting the query and
    # retrying (bounded) before answering or falling back. These are structural
    # switches, not per-signal routing thresholds (those stay in EvidencePolicy).
    #
    #   * enabled           — master switch. When False the chain uses the legacy
    #                         one-pass hybrid router (chains/router.py) unchanged.
    #   * max_attempts      — hard cap on retrieval attempts per query (>=1). The
    #                         self-healing loop rewrites + retries up to this many
    #                         times, so it can never loop forever.
    #   * evidence_threshold — a confidence *guideline* (0..1) the evidence
    #                         evaluator uses when blending its LLM judgement with
    #                         the hybrid signals. It is deliberately NOT a single
    #                         hard score gate replacing the old FAISS threshold.
    # Bare env names (AGENTIC_*) work alongside the DOCINTEL_-prefixed ones.
    agentic_rag_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("AGENTIC_RAG_ENABLED", "DOCINTEL_AGENTIC_RAG_ENABLED"),
    )
    agentic_max_retrieval_attempts: int = Field(
        default=3,
        validation_alias=AliasChoices(
            "AGENTIC_MAX_RETRIEVAL_ATTEMPTS", "DOCINTEL_AGENTIC_MAX_RETRIEVAL_ATTEMPTS"
        ),
    )
    agentic_evidence_threshold: float = Field(
        default=0.70,
        validation_alias=AliasChoices(
            "AGENTIC_EVIDENCE_THRESHOLD", "DOCINTEL_AGENTIC_EVIDENCE_THRESHOLD"
        ),
    )

    # -- LLM via Groq (models/llm_model.py) -------------------------------------
    # Groq exposes an OpenAI-compatible API, so the app still uses the same
    # ChatOpenAI abstraction with a different API key, base URL, and model.
    # GROQ_* names are preferred, while the old OPENROUTER_* names remain accepted
    # for compatibility during transition.
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "GROQ_API_KEY",
            "DOCINTEL_GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "DOCINTEL_OPENROUTER_API_KEY",
        ),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices(
            "GROQ_BASE_URL",
            "DOCINTEL_GROQ_BASE_URL",
            "OPENROUTER_BASE_URL",
            "DOCINTEL_OPENROUTER_BASE_URL",
        ),
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        validation_alias=AliasChoices(
            "GROQ_MODEL",
            "DOCINTEL_GROQ_MODEL",
            "OPENROUTER_MODEL",
            "DOCINTEL_LLM_MODEL",
        ),
    )
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
