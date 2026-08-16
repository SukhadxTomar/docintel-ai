"""Centralized, env-overridable configuration for docintel-ai.

Every tunable that used to live as a hardcoded literal (chunk sizing, retriever
parameters, the LLM model/temperature, and the embedding model name) is defined
here in one place. Values are read from the environment with the ``DOCINTEL_``
prefix (also honoring a local ``.env`` file), so nothing needs a code change to
retune:

    DOCINTEL_CHUNK_SIZE=1000 DOCINTEL_LLM_TEMPERATURE=0.1 streamlit run app.py

Import the shared ``settings`` singleton instead of hardcoding these values.

Note: router retrieval tunables (``HYBRID_RAG_TOP_K`` /
``HYBRID_RAG_SIMILARITY_THRESHOLD``) intentionally remain in ``chains/router.py``
— they are part of the routing/threshold logic, not general config hygiene.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
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


settings = Settings()
