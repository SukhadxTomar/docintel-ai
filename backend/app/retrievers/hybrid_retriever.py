"""Hybrid retrieval: BM25 (keyword) + FAISS (vector) fused with Reciprocal Rank Fusion.

This is the project's main retrieval path. It replaces the earlier direct
FAISS-similarity / MMR retriever: every query fetches ``retrieval_candidates_k``
candidates from *both* a BM25 keyword index and the FAISS vector index, then
merges the two rankings with RRF (rank-based, so the two incompatible raw score
scales never get compared) and returns the top ``final_context_k`` unique chunks.

The retriever also surfaces the best FAISS *relevance* score (0..1) separately —
that value, not the RRF score, is what the router thresholds to decide RAG vs
general LLM (see ``chains/router.py``). RRF decides *which* chunks make the best
context; the FAISS relevance signal still answers *whether the document is
relevant to the question at all*.

Retrieval logic lives here; routing logic lives in the router. The router
consumes a :class:`HybridRetrievalResult` rather than reaching into FAISS/BM25.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.core.config import settings
from app.utils.doc_utils import context_length

_TOKEN_RE = re.compile(r"\w+")


def _bm25_preprocess(text: str) -> list[str]:
    """Lowercase + word-boundary tokenizer for BM25.

    The library default is a bare ``str.split()`` — case-sensitive and it keeps
    trailing punctuation, so "Refund." and "refund" become different terms.
    Folding case and splitting on word characters gives more reliable keyword /
    terminology matches, which is the whole reason BM25 is in the mix.
    """
    return _TOKEN_RE.findall(text.lower())


def _doc_key(doc: Any) -> str:
    """Stable identity for a chunk, used to dedupe the same chunk across lists.

    BM25 and FAISS return *different* ``Document`` instances for the same chunk,
    so object identity can't dedupe them. Source + page + content is stable
    across both (FAISS reconstructs identical ``page_content``/metadata from its
    docstore; BM25 holds the same chunk list), so it's a reliable fusion key.
    """
    metadata = getattr(doc, "metadata", None) or {}
    raw = "|".join(
        (
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            getattr(doc, "page_content", "") or "",
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# -- FAISS relevance scoring (routing signal) -----------------------------------
# Kept identical to the pre-hybrid router so the RAG-vs-LLM threshold behaviour is
# unchanged: prefer the vector store's 0..1 relevance score, fall back to turning
# a raw distance into a 0..1 confidence only if that's all the store exposes.
def _coerce_relevance_score(score: float) -> float:
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _normalize_distance_score(score: float) -> float:
    """Convert a distance-style vector score into a 0..1 confidence value."""
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return 1.0 / (1.0 + max(float(score), 0.0))


def _faiss_with_scores(
    vector_store: Any, query: str, k: int
) -> tuple[list[tuple[Any, float]], bool]:
    """Return ``([(doc, 0..1 score)], used_scores)`` for the top-k FAISS hits.

    ``used_scores`` is False only when the store exposes no scored search at all
    (so the router can't evaluate its threshold and falls back to the LLM).
    """
    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
        return [(doc, _coerce_relevance_score(float(score))) for doc, score in results], True

    if hasattr(vector_store, "similarity_search_with_score"):
        results = vector_store.similarity_search_with_score(query, k=k)
        return [(doc, _normalize_distance_score(float(score))) for doc, score in results], True

    return [], False


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Any]]], *, rrf_k: int
) -> list[tuple[Any, float]]:
    """Fuse ranked document lists by Reciprocal Rank Fusion.

    ``ranked_lists`` is a list of ranked lists; each ranked list is an ordered
    sequence of ``(doc_key, Document)`` tuples, best first (rank 1 = index 0).
    Deduplication is inherent — a chunk appearing in several lists has its
    contributions summed under one key:

        RRF_score(d) = Σ over lists  1 / (rrf_k + rank(d))

    Returns ``[(Document, rrf_score)]`` sorted by descending RRF score. Only the
    rank *position* feeds the score, never the lists' raw (incomparable) scores.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Any] = {}
    for ranked in ranked_lists:
        for rank, (key, doc) in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs.setdefault(key, doc)  # keep the first instance seen for this chunk
    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(docs[key], score) for key, score in fused]


def documents_from_vector_store(vector_store: Any) -> list[Document]:
    """Recover the chunk ``Document`` objects stored inside a FAISS index.

    Used on the load-persisted-index path so the BM25 index can be rebuilt over
    exactly the same chunks that were embedded, without persisting BM25 itself.
    """
    docstore = getattr(vector_store, "docstore", None)
    id_map = getattr(vector_store, "index_to_docstore_id", None)
    if docstore is None or id_map is None:
        return []

    ids = id_map.values() if isinstance(id_map, dict) else id_map
    documents: list[Document] = []
    for doc_id in ids:
        doc = docstore.search(doc_id)
        if isinstance(doc, Document):
            documents.append(doc)
    return documents


@dataclass
class HybridRetrievalResult:
    """Everything one hybrid retrieval produced — for the router and for logging."""

    documents: list[Any]  # final top-``final_k`` chunks (RRF order) → RAG context
    faiss_best_score: float | None  # routing signal: max FAISS relevance (0..1)
    used_scores: bool  # False when FAISS exposed no scores at all
    faiss_ranked: list[tuple[Any, float]] = field(default_factory=list)  # (doc, relevance)
    bm25_ranked: list[Any] = field(default_factory=list)  # docs, best first
    fused: list[tuple[Any, float]] = field(default_factory=list)  # (doc, rrf_score)
    candidates_k: int = 0
    final_k: int = 0
    rrf_k: int = 0
    retrieval_time_ms: float | None = None
    context_length: int = 0


class HybridRetriever:
    """BM25 + FAISS retrieval fused with RRF over one shared set of chunks.

    Build it once per document set (BM25 is indexed at construction, not per
    query). ``vector_store`` is exposed as an attribute so the existing
    ``doc_utils.vector_store_from_retriever`` / persistence helpers keep working
    unchanged.
    """

    def __init__(
        self,
        vector_store: Any,
        *,
        documents: list[Any] | None = None,
        candidates_k: int | None = None,
        final_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.candidates_k = candidates_k or settings.retrieval_candidates_k
        self.final_k = final_k or settings.final_context_k
        self.rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

        docs = documents if documents is not None else documents_from_vector_store(vector_store)
        self._documents: list[Any] = list(docs)

        self._bm25: BM25Retriever | None = None
        if self._documents:
            self._bm25 = BM25Retriever.from_documents(
                self._documents, preprocess_func=_bm25_preprocess
            )
            self._bm25.k = self.candidates_k

    @property
    def num_chunks(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str) -> HybridRetrievalResult:
        """Retrieve BM25 + FAISS candidates and fuse them into a ranked context."""
        started_at = perf_counter()

        # FAISS vector candidates (carry the 0..1 relevance used for routing).
        faiss_scored, used_scores = _faiss_with_scores(
            self.vector_store, query, self.candidates_k
        )
        faiss_best = max((score for _, score in faiss_scored), default=None)

        # BM25 keyword candidates (ranked Documents; RRF only needs their order).
        bm25_docs: list[Any] = []
        if self._bm25 is not None:
            bm25_docs = list(self._bm25.invoke(query))

        # Fuse the two rankings, dedupe, and take the final context window.
        faiss_ranked = [(_doc_key(doc), doc) for doc, _ in faiss_scored]
        bm25_ranked = [(_doc_key(doc), doc) for doc in bm25_docs]
        fused = reciprocal_rank_fusion([faiss_ranked, bm25_ranked], rrf_k=self.rrf_k)
        final_documents = [doc for doc, _ in fused[: self.final_k]]

        return HybridRetrievalResult(
            documents=final_documents,
            faiss_best_score=faiss_best,
            used_scores=used_scores,
            faiss_ranked=faiss_scored,
            bm25_ranked=bm25_docs,
            fused=fused,
            candidates_k=self.candidates_k,
            final_k=self.final_k,
            rrf_k=self.rrf_k,
            retrieval_time_ms=(perf_counter() - started_at) * 1000,
            context_length=context_length(final_documents),
        )


def build_hybrid_retriever(
    vector_store: Any, documents: list[Any] | None = None
) -> HybridRetriever:
    """Factory for a :class:`HybridRetriever` using the configured k / RRF values."""
    return HybridRetriever(vector_store, documents=documents)
