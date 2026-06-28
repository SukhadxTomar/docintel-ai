from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

from utils.logger import log

RouteName = Literal["llm", "rag"]

DEFAULT_TOP_K = int(os.getenv("HYBRID_RAG_TOP_K", "4"))
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("HYBRID_RAG_SIMILARITY_THRESHOLD", "0.55")
)
MIN_SCORELESS_CONFIDENCE = float(
    os.getenv("HYBRID_RAG_SCORELESS_CONFIDENCE", "0.28")
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "tell", "explain", "describe", "give",
}


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


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _has_meaningful_content(docs: list[Any]) -> bool:
    return any(len(getattr(doc, "page_content", "").strip()) >= 80 for doc in docs)


def _chunk_confidence(question: str, doc: Any) -> float:
    query_tokens = _tokens(question)
    if not query_tokens:
        return 0.0

    content = getattr(doc, "page_content", "") or ""
    content_tokens = _tokens(content)
    if not content_tokens:
        return 0.0

    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    density = len(query_tokens & content_tokens) / max(len(content_tokens), 1)
    coverage = min(len(content.strip()) / 900, 1.0)
    metadata = getattr(doc, "metadata", None) or {}
    metadata_bonus = 0.06 if metadata.get("source") else 0.0
    page_bonus = 0.03 if metadata.get("page") is not None else 0.0

    return min((overlap * 0.74) + (density * 0.10) + (coverage * 0.07) + metadata_bonus + page_bonus, 1.0)


def _scoreless_confidence(question: str, docs: list[Any]) -> float:
    return max((_chunk_confidence(question, doc) for doc in docs), default=0.0)


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
    if hasattr(vector_store, "similarity_search_with_score"):
        results = vector_store.similarity_search_with_score(question, k=k)
        return [(doc, _normalize_distance_score(float(score))) for doc, score in results]

    if hasattr(vector_store, "similarity_search_with_relevance_scores"):
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)
        return [(doc, _coerce_relevance_score(float(score))) for doc, score in results]

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
    log.kv("Chunks Retrieved", len(decision.retrieved_docs))
    log.kv("Chunks Selected For RAG", len(decision.docs))
    log.kv("Similarity Scores", "N/A" if not decision.used_scores else [_score_label(score) for score in decision.retrieved_scores])
    log.kv("Threshold", "N/A" if decision.threshold is None else f"{decision.threshold:.4f}")
    log.kv("Best Score", _score_label(decision.best_score))
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
        log.info("Similarity scores unavailable; scoreless routing used conservative text/metadata confidence.")

    log.divider()
    log.info("Prompt")
    log.kv("Context Length", decision.context_length)
    log.kv("Question Length", len(question))

    log.divider()
    log.info("LLM Response")
    log.kv("Source", ", ".join(_source_names(decision.docs)) if decision.route == "rag" else "General AI Knowledge")
    log.kv("Routing Time", _time_label(decision.total_routing_time_ms))


def _with_total_time(decision: RouteDecision, started_at: float) -> RouteDecision:
    return RouteDecision(
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


def _finish(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
    started_at: float,
) -> RouteDecision:
    final_decision = _with_total_time(decision, started_at)
    _log_router_decision(question, retriever_available, final_decision)
    return final_decision


def _qualified_scored_results(
    question: str,
    scored_results: list[tuple[Any, float]],
    threshold: float,
) -> list[tuple[Any, float]]:
    qualified = []
    for doc, score in scored_results:
        lexical_confidence = _chunk_confidence(question, doc)
        # Vector score is primary. Text/metadata confidence prevents weak top-k
        # results from becoming RAG, while still allowing semantically strong hits.
        routing_confidence = max(score, (score * 0.70) + (lexical_confidence * 0.30))
        if routing_confidence >= threshold:
            qualified.append((doc, routing_confidence))
    return qualified


def route_query(
    retriever: Any | None,
    question: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RouteDecision:
    """Route by retrieval quality and reuse the same chunks for RAG."""
    started_at = perf_counter()

    if retriever is None:
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

        if vector_store is not None:
            scored_results = _retrieve_with_scores(vector_store, question, k)

            if scored_results is not None:
                retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
                docs = [doc for doc, _ in scored_results]
                scores = [score for _, score in scored_results]
                best_score = max(scores, default=None)
                qualified_results = _qualified_scored_results(question, scored_results, similarity_threshold)

                if qualified_results:
                    rag_docs = [doc for doc, _ in qualified_results]
                    rag_scores = [score for _, score in qualified_results]
                    return _finish(
                        question,
                        True,
                        RouteDecision(
                            "rag",
                            "Retrieved document chunks met the routing confidence threshold.",
                            docs=rag_docs,
                            scores=rag_scores,
                            best_score=max(rag_scores, default=best_score),
                            used_scores=True,
                            threshold=similarity_threshold,
                            retrieval_time_ms=retrieval_time_ms,
                            retrieved_docs=docs,
                            retrieved_scores=scores,
                            context_length=_context_length(rag_docs),
                        ),
                        started_at,
                    )

                return _finish(
                    question,
                    True,
                    RouteDecision(
                        "llm",
                        "Retrieved chunks were below the routing confidence threshold.",
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

        docs = _retrieve_without_scores(retriever, question)
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
        confidence = _scoreless_confidence(question, docs)
        has_confident_chunks = _has_meaningful_content(docs) and confidence >= MIN_SCORELESS_CONFIDENCE

        if has_confident_chunks:
            return _finish(
                question,
                True,
                RouteDecision(
                    "rag",
                    f"Similarity scores unavailable, but scoreless retrieval confidence was strong ({confidence:.4f}).",
                    docs=docs,
                    scores=[None] * len(docs),
                    best_score=None,
                    used_scores=False,
                    threshold=MIN_SCORELESS_CONFIDENCE,
                    retrieval_time_ms=retrieval_time_ms,
                    retrieved_docs=docs,
                    retrieved_scores=[None] * len(docs),
                    context_length=_context_length(docs),
                ),
                started_at,
            )

        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                f"Similarity scores unavailable and scoreless retrieval confidence was weak ({confidence:.4f}).",
                docs=[],
                scores=[],
                best_score=None,
                used_scores=False,
                threshold=MIN_SCORELESS_CONFIDENCE,
                retrieval_time_ms=retrieval_time_ms,
                retrieved_docs=docs,
                retrieved_scores=[None] * len(docs),
                context_length=0,
            ),
            started_at,
        )

    except Exception as exc:
        retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
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

