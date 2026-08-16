from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.utils.logger import log

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the HuggingFace embedding model once, on first use, then reuse it."""
    global _embeddings

    if _embeddings is None:
        log.info("Loading Embedding Model...")
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        log.success("Embedding Model Loaded")
        log.kv("Embedding Model Name", settings.embedding_model)
        log.success("Load Success")

    return _embeddings
