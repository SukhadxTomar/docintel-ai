"""Builders for the project's main retrieval path: the hybrid BM25 + FAISS + RRF
retriever (see ``retrievers/hybrid_retriever.py``).

The earlier MMR ``as_retriever`` configuration has been removed — it was only
ever a score-less fallback that the router never actually used, and it is now
fully superseded by the hybrid retriever. Both builders below return a
:class:`HybridRetriever`, which exposes ``.vector_store`` so the existing
persistence / logging helpers keep working unchanged.
"""
from app.ingestion.vector_store import create_vector_store
from app.retrievers.hybrid_retriever import HybridRetriever, build_hybrid_retriever
from app.utils.logger import log


def _index_size(vector_store):
    index = getattr(vector_store, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return index.ntotal

    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        return len(ids)

    return "Unknown"


def _log_retriever(retriever: HybridRetriever, vector_store) -> None:
    log.section("Hybrid Retriever Build")
    log.kv("Vector Store Created", "YES" if vector_store is not None else "NO")
    log.kv("FAISS Index Size", _index_size(vector_store))
    log.kv("BM25 Chunks Indexed", retriever.num_chunks)
    log.kv("Candidates K (BM25 + FAISS each)", retriever.candidates_k)
    log.kv("Final Context K", retriever.final_k)
    log.kv("RRF K", retriever.rrf_k)
    log.success("Hybrid Retriever Created")


def retriever_from_vector_store(vector_store) -> HybridRetriever:
    """Build the hybrid retriever from an existing (e.g. persisted) FAISS store.

    The chunk ``Document`` objects are recovered from the FAISS docstore so the
    BM25 index is rebuilt over exactly the same chunks that were embedded — the
    load-persisted-index path in ``session/manager.py``.
    """
    retriever = build_hybrid_retriever(vector_store)
    _log_retriever(retriever, vector_store)
    return retriever


def create_retriever(chunks):
    """Build the hybrid retriever from freshly-split chunks.

    The same ``chunks`` feed both the FAISS embeddings/index and the BM25 index
    (one chunking pipeline, not two), so source/page metadata and citations line
    up across both retrievers.
    """
    vector_store = create_vector_store(chunks)
    retriever = build_hybrid_retriever(vector_store, documents=chunks)
    _log_retriever(retriever, vector_store)
    return retriever
