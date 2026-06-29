from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from utils.logger import log

RouteName = Literal["llm", "rag"]

DEFAULT_TOP_K = int(os.getenv("HYBRID_RAG_TOP_K", "4"))
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("HYBRID_RAG_SIMILARITY_THRESHOLD", "0.35")
)


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str
    docs: list[Any] = field(default_factory=list)
    scores: list[float | None] = field(default_factory=list)
    best_score: float | None = None
    used_scores: bool = False
    threshold: float | None = None
    retrieval_time_ms: float | None = None
    total_routing_time_ms: float | None = None
    retrieved_docs: list[Any] = field(default_factory=list)
    retrieved_scores: list[float | None] = field(default_factory=list)
    context_length: int = 0


def _top_k(retriever: Any) -> int:
    search_kwargs = getattr(retriever, "search_kwargs", None) or {}
    return int(search_kwargs.get("k") or DEFAULT_TOP_K)


def _normalize_distance_score(score: float) -> float:
    """Convert distance-style vector scores into a 0..1 confidence value."""
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return 1.0 / (1.0 + max(float(score), 0.0))


def _coerce_relevance_score(score: float) -> float:
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(float(score), 1.0))


def _vector_store(retriever: Any) -> Any | None:
    return getattr(retriever, "vectorstore", None) or getattr(retriever, "vector_store", None)


def _retrieve_with_scores(
    vector_store: Any,
    question: str,
    k: int,
) -> list[tuple[Any, float]] | None:
    """Return docs with comparable 0..1 confidence scores when possible."""
    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)
        return [(doc, _coerce_relevance_score(float(score))) for doc, score in results]

    if hasattr(vector_store, "similarity_search_with_score"):
        results = vector_store.similarity_search_with_score(question, k=k)
        return [(doc, _normalize_distance_score(float(score))) for doc, score in results]

    return None


def _retrieve_without_scores(retriever: Any, question: str) -> list[Any]:
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(question))

    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(question))

    return []


def _page_label(page: Any) -> str:
    if page is None:
        return "Unknown"

    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page)


def _source_name(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    source = metadata.get("source", "Unknown")
    return str(metadata.get("original_name") or os.path.basename(str(source)))


def _source_names(docs: list[Any]) -> list[str]:
    names = []
    seen = set()
    for doc in docs:
        name = _source_name(doc)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _score_label(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


def _top_score_labels(scores: list[float | None], limit: int = 3) -> list[str]:
    return [_score_label(score) for score in scores[:limit]]


def _time_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _context_length(docs: list[Any]) -> int:
    return sum(len(getattr(doc, "page_content", "") or "") for doc in docs)


def _log_router_decision(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
) -> None:
    log.section("NEW USER QUERY")
    log.kv("Question", question)
    log.kv("Retriever Exists", "Yes" if retriever_available else "No")
    log.kv("Router Decision", "RAG" if decision.route == "rag" else "GENERAL LLM")
    log.kv("Reason", decision.reason)

    if not retriever_available:
        log.kv("Source", "General AI Knowledge")
        log.kv("Routing Time", _time_label(decision.total_routing_time_ms))
        return

    log.divider()
    log.info("Retriever")
    log.kv("Retrieved Chunks", len(decision.retrieved_docs))
    log.kv("Retrieved PDF Names", ", ".join(_source_names(decision.retrieved_docs)) or "None")
    log.kv("Chunks Selected For RAG", len(decision.docs))
    log.kv("Similarity Scores", "N/A" if not decision.used_scores else [_score_label(score) for score in decision.retrieved_scores])
    log.kv("Top 3 Scores", "N/A" if not decision.used_scores else _top_score_labels(decision.retrieved_scores))
    log.kv("Threshold", "N/A" if decision.threshold is None else f"{decision.threshold:.4f}")
    log.kv("Best Similarity Score", _score_label(decision.best_score))
    log.kv("Retrieval Time", _time_label(decision.retrieval_time_ms))

    if decision.retrieved_docs:
        log.info("Documents")
        for index, doc in enumerate(decision.retrieved_docs, start=1):
            metadata = getattr(doc, "metadata", None) or {}
            page = _page_label(metadata.get("page"))
            score = decision.retrieved_scores[index - 1] if index - 1 < len(decision.retrieved_scores) else None
            preview = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")[:180]
            log.list_item(f"{index}. {_source_name(doc)} | Page {page} | Score: {_score_label(score)}")
            if preview:
                log.list_item(f"   Preview: {preview}")
    else:
        log.info("Documents: None")

    if not decision.used_scores:
        log.info("Similarity scores unavailable; router cannot apply the vector-score threshold.")

    log.divider()
    log.info("Prompt")
    log.kv("Context Length", decision.context_length)
    log.kv("Question Length", len(question))

    log.divider()
    log.info("LLM Response")
    log.kv("Source", ", ".join(_source_names(decision.docs)) if decision.route == "rag" else "General AI Knowledge")
    log.kv("Routing Time", _time_label(decision.total_routing_time_ms))
    log.kv("Response Time", _time_label(decision.total_routing_time_ms))


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
        scores=decision.scores,
        best_score=decision.best_score,
        used_scores=decision.used_scores,
        threshold=decision.threshold,
        retrieval_time_ms=decision.retrieval_time_ms,
        total_routing_time_ms=(perf_counter() - started_at) * 1000,
        retrieved_docs=decision.retrieved_docs,
        retrieved_scores=decision.retrieved_scores,
        context_length=decision.context_length,
    )
    _log_router_decision(question, retriever_available, final_decision)
    return final_decision


def route_query(
    retriever: Any | None,
    question: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RouteDecision:
    """Route by retrieval quality and reuse the same chunks for RAG."""
    started_at = perf_counter()

    if retriever is None:
        log.section("Router Retrieval Debug")
        log.kv("Question", question)
        log.kv("Attempting Retrieval", "NO")
        log.kv("Retrieval Skipped Reason", "Retriever is None.")
        return _finish(
            question,
            False,
            RouteDecision(
                "llm",
                "No document retriever available.",
                retrieval_time_ms=0.0,
            ),
            started_at,
        )

    retrieval_started_at = perf_counter()

    try:
        k = _top_k(retriever)
        vector_store = _vector_store(retriever)
        log.section("Router Retrieval Debug")
        log.kv("Question", question)
        log.kv("Attempting Retrieval", "YES")
        log.kv("Retriever Type", type(retriever).__name__)
        log.kv("Vector Store Exists", "YES" if vector_store is not None else "NO")

        if vector_store is not None:
            scored_results = _retrieve_with_scores(vector_store, question, k)

            if scored_results is not None:
                retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
                docs = [doc for doc, _ in scored_results]
                scores = [score for _, score in scored_results]
                best_score = max(scores, default=None)
                log.kv("Retrieved Chunks", len(docs))
                log.kv("Best Similarity Score", _score_label(best_score))
                log.kv("Threshold", f"{similarity_threshold:.4f}")
                log.kv("Top 3 Scores", _top_score_labels(scores))
                log.kv("Source PDFs", ", ".join(_source_names(docs)) or "None")

                if best_score is not None and best_score >= similarity_threshold:
                    log.kv("Final Route", "RAG")
                    log.kv("Reason", f"Best score {best_score:.4f} >= threshold {similarity_threshold:.4f}.")
                    return _finish(
                        question,
                        True,
                        RouteDecision(
                            "rag",
                            f"Best vector similarity score ({best_score:.4f}) met the threshold ({similarity_threshold:.4f}).",
                            docs=docs,
                            scores=scores,
                            best_score=best_score,
                            used_scores=True,
                            threshold=similarity_threshold,
                            retrieval_time_ms=retrieval_time_ms,
                            retrieved_docs=docs,
                            retrieved_scores=scores,
                            context_length=_context_length(docs),
                        ),
                        started_at,
                    )

                log.kv("Final Route", "LLM")
                log.kv(
                    "Reason",
                    "No retrieved chunk met the vector similarity threshold."
                    if best_score is not None
                    else "Retriever returned no scored chunks.",
                )
                return _finish(
                    question,
                    True,
                    RouteDecision(
                        "llm",
                        (
                            "No retrieved chunk met the vector similarity threshold."
                            if best_score is not None
                            else "Retriever returned no scored chunks."
                        ),
                        docs=[],
                        scores=[],
                        best_score=best_score,
                        used_scores=True,
                        threshold=similarity_threshold,
                        retrieval_time_ms=retrieval_time_ms,
                        retrieved_docs=docs,
                        retrieved_scores=scores,
                        context_length=0,
                    ),
                    started_at,
                )

        if vector_store is None:
            log.kv("Retrieval Skipped Reason", "Retriever has no vectorstore/vector_store attribute.")
        else:
            log.kv("Retrieval Skipped Reason", "Vector store does not expose scored similarity search.")

        docs = _retrieve_without_scores(retriever, question)
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
        log.kv("Retrieved Chunks", len(docs))
        log.kv("Best Similarity Score", "N/A")
        log.kv("Threshold", f"{similarity_threshold:.4f}")
        log.kv("Top 3 Scores", "N/A")
        log.kv("Source PDFs", ", ".join(_source_names(docs)) or "None")
        log.kv("Final Route", "LLM")
        log.kv("Reason", "Similarity scores unavailable, so the vector-score threshold could not be evaluated.")
        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                "Similarity scores unavailable, so the vector-score threshold could not be evaluated.",
                docs=[],
                scores=[],
                best_score=None,
                used_scores=False,
                threshold=similarity_threshold,
                retrieval_time_ms=retrieval_time_ms,
                retrieved_docs=docs,
                retrieved_scores=[None] * len(docs),
                context_length=0,
            ),
            started_at,
        )

    except Exception as exc:
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
        log.kv("Retrieved Chunks", 0)
        log.kv("Best Similarity Score", "N/A")
        log.kv("Threshold", f"{similarity_threshold:.4f}")
        log.kv("Top 3 Scores", "N/A")
        log.kv("Source PDFs", "None")
        log.kv("Final Route", "LLM")
        log.kv("Reason", f"Retrieval failed: {exc}")
        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                f"Retrieval failed, so routing fell back to the normal LLM chain: {exc}",
                retrieval_time_ms=retrieval_time_ms,
            ),
            started_at,
        )

