"""Hybrid retrieval: BM25 (keyword) + FAISS (vector) fused with Reciprocal Rank Fusion.

This is the project's main retrieval path. It replaces the earlier direct
FAISS-similarity / MMR retriever: every query fetches ``retrieval_candidates_k``
candidates from *both* a BM25 keyword index and the FAISS vector index, then
merges the two rankings with RRF (rank-based, so the two incompatible raw score
scales never get compared) and returns the top ``final_context_k`` unique chunks.

Beyond the fused context, the retriever surfaces the per-chunk retrieval facts the
router needs to make a *hybrid evidence* routing decision (RAG vs. general LLM):
for every candidate it records the FAISS rank + 0..1 relevance, the BM25 rank, the
fused RRF rank/score, and whether *both* retrievers found it (cross-retriever
agreement). These land in :class:`HybridRetrievalResult.evidence` as
:class:`ChunkEvidence` rows. See ``chains/router.py`` for the policy that weighs
those signals.

RRF still only decides *which* chunks are the best context, and is never treated
as a probability. *Whether* the document is relevant enough to answer from is a
separate call the router makes from the combined lexical + semantic + agreement
evidence — no longer from the FAISS relevance score alone.

Retrieval logic lives here; routing logic lives in the router. The router consumes
a :class:`HybridRetrievalResult` rather than reaching into FAISS/BM25, and one user
query performs exactly one hybrid retrieval.
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


def tokenize(text: str) -> list[str]:
    """Lowercase + word-boundary tokenizer, shared by BM25 and the router.

    The library default is a bare ``str.split()`` — case-sensitive and it keeps
    trailing punctuation, so "Refund." and "refund" become different terms.
    Folding case and splitting on word characters gives more reliable keyword /
    terminology matches, which is the whole reason BM25 is in the mix.

    The router reuses this exact tokenizer for its lexical-evidence check, so
    "does this chunk actually contain the query's distinctive terms?" is measured
    with the same tokenization BM25 indexed and matched with — there is no second,
    divergent tokenizer to drift out of sync.
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


def chunk_key(doc: Any) -> str:
    """Public, stable chunk identity — the same value RRF and the evidence map key on.

    Exposed so the router can relate a fused/final ``Document`` back to its
    :class:`ChunkEvidence` row (e.g. to log cross-retriever agreement) without
    re-deriving keys or reaching into retrieval internals.
    """
    return _doc_key(doc)


# -- FAISS relevance scoring (semantic evidence signal) -------------------------
# The 0..1 relevance score is the router's *semantic* evidence signal — one of the
# three signals it weighs (semantic / lexical / agreement). Scoring itself is
# unchanged from the pre-hybrid router: prefer the vector store's native 0..1
# relevance score, and only fall back to turning a raw distance into a 0..1
# confidence if that is all the store exposes.
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


@dataclass(frozen=True)
class ChunkEvidence:
    """Per-chunk retrieval facts, so the router can weigh evidence without
    recomputing keys or re-querying either retriever.

    Pure retrieval bookkeeping — no thresholds, no routing policy. Ranks are
    1-based; a ``None`` rank/score means the chunk did not appear in that
    retriever's candidate list. ``document`` is kept so the router can read
    ``page_content`` for its lexical-overlap check (that content is never logged).
    """

    key: str
    document: Any
    faiss_rank: int | None = None
    faiss_relevance: float | None = None  # 0..1 vector relevance (None if unscored)
    bm25_rank: int | None = None
    rrf_rank: int | None = None
    rrf_score: float | None = None

    @property
    def in_both(self) -> bool:
        """True when BOTH retrievers surfaced this chunk (cross-retriever agreement)."""
        return self.faiss_rank is not None and self.bm25_rank is not None


def _build_evidence(
    faiss_keyed: list[tuple[str, Any, float]],
    bm25_keyed: list[tuple[str, Any]],
    fused: list[tuple[Any, float]],
) -> dict[str, ChunkEvidence]:
    """Fold each retriever's ranks/scores into one map keyed by stable chunk id.

    ``fused`` already spans the full candidate union (RRF sees every chunk from
    both input lists), so it drives the chunk set and supplies the canonical
    ``Document`` instance per key. This is retrieval bookkeeping only; the router
    applies the policy.
    """
    faiss_rank: dict[str, int] = {}
    faiss_rel: dict[str, float] = {}
    for rank, (key, _doc, score) in enumerate(faiss_keyed, start=1):
        faiss_rank.setdefault(key, rank)
        faiss_rel.setdefault(key, score)

    bm25_rank: dict[str, int] = {}
    for rank, (key, _doc) in enumerate(bm25_keyed, start=1):
        bm25_rank.setdefault(key, rank)

    evidence: dict[str, ChunkEvidence] = {}
    for rrf_rank, (doc, rrf_score) in enumerate(fused, start=1):
        key = _doc_key(doc)
        if key in evidence:
            continue
        evidence[key] = ChunkEvidence(
            key=key,
            document=doc,
            faiss_rank=faiss_rank.get(key),
            faiss_relevance=faiss_rel.get(key),
            bm25_rank=bm25_rank.get(key),
            rrf_rank=rrf_rank,
            rrf_score=rrf_score,
        )
    return evidence


@dataclass
class HybridRetrievalResult:
    """Everything one hybrid retrieval produced — for the router and for logging."""

    documents: list[Any]  # final top-``final_k`` chunks (RRF order) → RAG context
    faiss_best_score: float | None  # max FAISS relevance (0..1); the semantic signal
    used_scores: bool  # False when FAISS exposed no scores at all
    faiss_ranked: list[tuple[Any, float]] = field(default_factory=list)  # (doc, relevance)
    bm25_ranked: list[Any] = field(default_factory=list)  # docs, best first
    fused: list[tuple[Any, float]] = field(default_factory=list)  # (doc, rrf_score)
    # Per-chunk evidence the router consumes for its hybrid routing decision, and
    # the keys of the final chunks (aligned with ``documents``). Together these let
    # the router evaluate semantic + lexical + agreement signals off a single
    # retrieval pass — it never re-runs BM25/FAISS or recomputes chunk keys.
    evidence: dict[str, ChunkEvidence] = field(default_factory=dict)
    final_keys: list[str] = field(default_factory=list)
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
                self._documents, preprocess_func=tokenize
            )
            self._bm25.k = self.candidates_k

    @property
    def num_chunks(self) -> int:
        return len(self._documents)

    def retrieve(self, query: str) -> HybridRetrievalResult:
        """Retrieve BM25 + FAISS candidates and fuse them into a ranked context."""
        started_at = perf_counter()

        # FAISS vector candidates (carry the 0..1 relevance used as the semantic signal).
        faiss_scored, used_scores = _faiss_with_scores(
            self.vector_store, query, self.candidates_k
        )
        faiss_best = max((score for _, score in faiss_scored), default=None)

        # BM25 keyword candidates (ranked Documents; RRF only needs their order).
        bm25_docs: list[Any] = []
        if self._bm25 is not None:
            bm25_docs = list(self._bm25.invoke(query))

        # Key every candidate exactly once; RRF, the final context, and the
        # per-chunk evidence map below all reuse these keyed lists.
        faiss_keyed = [(_doc_key(doc), doc, score) for doc, score in faiss_scored]
        bm25_keyed = [(_doc_key(doc), doc) for doc in bm25_docs]

        # Fuse the two rankings, dedupe, and take the final context window.
        fused = reciprocal_rank_fusion(
            [[(key, doc) for key, doc, _ in faiss_keyed], bm25_keyed], rrf_k=self.rrf_k
        )
        final_documents = [doc for doc, _ in fused[: self.final_k]]

        return HybridRetrievalResult(
            documents=final_documents,
            faiss_best_score=faiss_best,
            used_scores=used_scores,
            faiss_ranked=faiss_scored,
            bm25_ranked=bm25_docs,
            fused=fused,
            evidence=_build_evidence(faiss_keyed, bm25_keyed, fused),
            final_keys=[_doc_key(doc) for doc in final_documents],
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
