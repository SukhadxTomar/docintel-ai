"""Query router: decide RAG vs. general LLM, and hand RAG the fused context.

The router no longer talks to FAISS/BM25 directly — it consumes a
:class:`~app.retrievers.hybrid_retriever.HybridRetrievalResult` produced by the
hybrid retriever (BM25 + FAISS fused with RRF). Two distinct signals come out of
that result:

* **Routing signal** — the best FAISS *relevance* score (0..1). This is compared
  against ``HYBRID_RAG_SIMILARITY_THRESHOLD`` (0.35) to decide whether the
  document is relevant to the question at all. It is deliberately NOT the RRF
  score: RRF scores are tiny rank-fusion values (~1/61) on an arbitrary scale, so
  the 0.35 threshold would be meaningless against them.
* **Context ranking** — the RRF-fused top-``final_context_k`` chunks
  (``result.documents``). When the router chooses RAG, these are the chunks the
  RAG chain actually answers from.

So RRF decides *which* chunks are the best context; the FAISS relevance signal
decides *whether* to use them. Everything else (RAG chain, LLM fallback, graceful
degradation when scores are unavailable) is unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from app.retrievers.hybrid_retriever import HybridRetrievalResult
from app.utils.doc_utils import page_label, source_name, source_names
from app.utils.logger import log

RouteName = Literal["llm", "rag"]

# The routing threshold stays here (it is a routing decision, not general config
# hygiene). The retrieval structure — candidate/final counts and the RRF constant
# — lives in ``core/config.py``.
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("HYBRID_RAG_SIMILARITY_THRESHOLD", "0.35")
)

# How many candidate rows to print per ranked list (FAISS / BM25 / RRF). Purely a
# logging cap; retrieval always fuses the full candidate set.
_MAX_LOGGED = 20


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str
    docs: list[Any] = field(default_factory=list)  # RRF-fused chunks used for RAG
    best_score: float | None = None  # FAISS best relevance (the routing signal)
    used_scores: bool = False
    threshold: float | None = None
    retrieval_time_ms: float | None = None
    total_routing_time_ms: float | None = None
    retrieved_docs: list[Any] = field(default_factory=list)  # final fused chunks
    context_length: int = 0
    result: HybridRetrievalResult | None = None  # full hybrid result, for logging


def _score_label(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


def _time_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _candidate_row(rank: int, doc: Any, score: float | None, *, score_fmt: str) -> str:
    """One content-free candidate line: rank | source | page [| score].

    Deliberately logs only source + page + rank + score — never ``page_content`` —
    so full document text never lands in the logs.
    """
    metadata = getattr(doc, "metadata", None) or {}
    page = page_label(metadata.get("page"))
    base = f"{rank:>2}. {source_name(doc)} | Page {page}"
    if score is None:
        return base
    return f"{base} | {score_fmt}: {score:.4f}" if score_fmt else base


def _log_faiss(result: HybridRetrievalResult) -> None:
    log.info(f"FAISS Vector Candidates (top {result.candidates_k}, by relevance)")
    if not result.faiss_ranked:
        log.list_item("None")
        return
    for rank, (doc, score) in enumerate(result.faiss_ranked[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, score, score_fmt="relevance"))


def _log_bm25(result: HybridRetrievalResult) -> None:
    log.info(f"BM25 Keyword Candidates (top {result.candidates_k}, by rank)")
    if not result.bm25_ranked:
        log.list_item("None")
        return
    for rank, doc in enumerate(result.bm25_ranked[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, None, score_fmt=""))


def _log_fused(result: HybridRetrievalResult) -> None:
    log.info("RRF Fused Ranking (both lists merged by rank)")
    if not result.fused:
        log.list_item("None")
        return
    for rank, (doc, score) in enumerate(result.fused[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, score, score_fmt="rrf"))


def _log_final(documents: list[Any]) -> None:
    log.info("Final Context (top-k after fusion)")
    if not documents:
        log.list_item("None")
        return
    for rank, doc in enumerate(documents, start=1):
        log.list_item(_candidate_row(rank, doc, None, score_fmt=""))


def _log_router_decision(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
) -> None:
    log.section("NEW USER QUERY")
    log.kv("Query", question)
    log.kv("Retriever Exists", "Yes" if retriever_available else "No")

    result = decision.result
    if retriever_available and result is not None:
        log.divider()
        _log_faiss(result)
        log.divider()
        _log_bm25(result)
        log.divider()
        _log_fused(result)
        log.divider()
        _log_final(decision.retrieved_docs)

        log.divider()
        log.info("Routing")
        log.kv("Candidates K (per retriever)", result.candidates_k)
        log.kv("Final Context K", result.final_k)
        log.kv("RRF K", result.rrf_k)
        log.kv("FAISS Best Relevance (routing signal)", _score_label(decision.best_score))
        log.kv("Threshold", "N/A" if decision.threshold is None else f"{decision.threshold:.4f}")
        log.kv("Context Length", decision.context_length)
        log.kv("Retrieval Time", _time_label(decision.retrieval_time_ms))

    log.divider()
    log.kv("Router Decision", "RAG" if decision.route == "rag" else "GENERAL LLM")
    log.kv("Reason", decision.reason)
    log.kv(
        "Source",
        ", ".join(source_names(decision.docs)) if decision.route == "rag" else "General AI Knowledge",
    )
    log.kv("Routing Time", _time_label(decision.total_routing_time_ms))


def _finish(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
    started_at: float,
) -> RouteDecision:
    final_decision = RouteDecision(
        route=decision.route,
        reason=decision.reason,
        docs=decision.docs,
        best_score=decision.best_score,
        used_scores=decision.used_scores,
        threshold=decision.threshold,
        retrieval_time_ms=decision.retrieval_time_ms,
        total_routing_time_ms=(perf_counter() - started_at) * 1000,
        retrieved_docs=decision.retrieved_docs,
        context_length=decision.context_length,
        result=decision.result,
    )
    _log_router_decision(question, retriever_available, final_decision)
    return final_decision


def route_query(
    retriever: Any | None,
    question: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RouteDecision:
    """Route by hybrid retrieval quality and reuse the fused chunks for RAG.

    The retriever runs BM25 + FAISS and fuses them with RRF; this function only
    reads the result. RAG is chosen when the FAISS best relevance score clears the
    threshold, and the RRF-fused top-k chunks become the RAG context.
    """
    started_at = perf_counter()

    if retriever is None:
        return _finish(
            question,
            False,
            RouteDecision(
                "llm",
                "No document retriever available.",
                threshold=similarity_threshold,
                retrieval_time_ms=0.0,
            ),
            started_at,
        )

    try:
        result = retriever.retrieve(question)
        best_score = result.faiss_best_score

        # Routing signal = FAISS relevance, NOT the RRF score. RRF only ordered the
        # context; whether the document is relevant at all is a vector-space call.
        if result.used_scores and best_score is not None and best_score >= similarity_threshold:
            return _finish(
                question,
                True,
                RouteDecision(
                    "rag",
                    f"Best FAISS relevance ({best_score:.4f}) met the threshold ({similarity_threshold:.4f}); "
                    f"answering from the top {result.final_k} RRF-fused chunks.",
                    docs=result.documents,
                    best_score=best_score,
                    used_scores=True,
                    threshold=similarity_threshold,
                    retrieval_time_ms=result.retrieval_time_ms,
                    retrieved_docs=result.documents,
                    context_length=result.context_length,
                    result=result,
                ),
                started_at,
            )

        if not result.used_scores:
            reason = "FAISS relevance scores unavailable, so the routing threshold could not be evaluated."
        elif best_score is None:
            reason = "Hybrid retrieval returned no candidates."
        else:
            reason = (
                f"Best FAISS relevance ({best_score:.4f}) fell below the threshold "
                f"({similarity_threshold:.4f}); treated as unrelated to the documents."
            )

        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                reason,
                docs=[],
                best_score=best_score,
                used_scores=result.used_scores,
                threshold=similarity_threshold,
                retrieval_time_ms=result.retrieval_time_ms,
                retrieved_docs=result.documents,
                context_length=0,
                result=result,
            ),
            started_at,
        )

    except Exception as exc:
        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                f"Retrieval failed, so routing fell back to the normal LLM chain: {exc}",
                threshold=similarity_threshold,
                retrieval_time_ms=None,
            ),
            started_at,
        )
