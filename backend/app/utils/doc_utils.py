"""Shared, dependency-free helpers for documents and retrievers.

These were previously copy-pasted across app.py and the chain modules. They are
pure functions (no logging, no config, no I/O) so any module can import them
without creating an import cycle.
"""

from __future__ import annotations

import os
from typing import Any


def page_label(page: Any) -> str:
    """Human-friendly 1-based page label from 0-based PDF metadata."""
    if page is None:
        return "Unknown"

    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page)


def source_name(doc: Any) -> str:
    """Display name for a document: original upload name, else the file basename."""
    metadata = getattr(doc, "metadata", None) or {}
    source = metadata.get("source", "Unknown")
    return str(metadata.get("original_name") or os.path.basename(str(source)))


def source_names(docs: list[Any]) -> list[str]:
    """Distinct source names across docs, order-preserving."""
    names: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        name = source_name(doc)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def context_length(docs: list[Any]) -> int:
    """Total character length across all document contents."""
    return sum(len(getattr(doc, "page_content", "") or "") for doc in docs)


def vector_store_from_retriever(retriever: Any | None) -> Any | None:
    """The FAISS/vector store backing a retriever, if any."""
    if retriever is None:
        return None

    return getattr(retriever, "vectorstore", None) or getattr(retriever, "vector_store", None)


def indexed_chunks(retriever: Any | None) -> int | str:
    """Number of vectors indexed behind a retriever (0/'Unknown' when unavailable)."""
    vector_store = vector_store_from_retriever(retriever)
    if vector_store is None:
        return 0

    index = getattr(vector_store, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return int(index.ntotal)

    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        return len(ids)

    return "Unknown"
