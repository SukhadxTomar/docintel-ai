from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

RouteName = Literal["llm", "rag"]

DEFAULT_TOP_K = int(os.getenv("HYBRID_RAG_TOP_K", "4"))
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("HYBRID_RAG_SIMILARITY_THRESHOLD", "0.65")
)
MIN_SCORELESS_CONFIDENCE = float(
    os.getenv("HYBRID_RAG_SCORELESS_CONFIDENCE", "0.28")
)
DEBUG_ROUTER = os.getenv("DEBUG_ROUTER", "False").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
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


def _scoreless_confidence(question: str, docs: list[Any]) -> float:
    """Estimate relevance when the vector store cannot provide scores.

    This is a conservative fallback only. Normal FAISS routing should use vector
    similarity scores; this combines query/content overlap, chunk substance, and
    source metadata to avoid treating arbitrary top-k retrieval as relevant.
    """
    query_tokens = _tokens(question)
    if not query_tokens or not docs:
        return 0.0

    best_confidence = 0.0

    for doc in docs:
        content = getattr(doc, "page_content", "") or ""
        content_tokens = _tokens(content)
        if not content_tokens:
            continue

        overlap = len(query_tokens & content_tokens) / len(query_tokens)
        coverage = min(len(content.strip()) / 700, 1.0)
        metadata = getattr(doc, "metadata", None) or {}
        metadata_bonus = 0.08 if metadata.get("source") else 0.0
        page_bonus = 0.04 if metadata.get("page") is not None else 0.0

        confidence = min((overlap * 0.78) + (coverage * 0.10) + metadata_bonus + page_bonus, 1.0)
        best_confidence = max(best_confidence, confidence)

    return best_confidence


def _normalize_distance_score(score: float) -> float:
    """Convert distance-style vector scores into a 0..1 relevance score."""
    if math.isnan(score):
        return 0.0
    return 1.0 / (1.0 + max(float(score), 0.0))


def _vector_store(retriever: Any) -> Any | None:
    return getattr(retriever, "vectorstore", None) or getattr(retriever, "vector_store", None)


def _retrieve_with_relevance_scores(
    vector_store: Any,
    question: str,
    k: int,
) -> list[tuple[Any, float]] | None:
    if not hasattr(vector_store, "similarity_search_with_relevance_scores"):
        return None

    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    return [(doc, float(score)) for doc, score in results]


def _retrieve_with_raw_scores(
    vector_store: Any,
    question: str,
    k: int,
) -> list[tuple[Any, float]] | None:
    if not hasattr(vector_store, "similarity_search_with_score"):
        return None

    results = vector_store.similarity_search_with_score(question, k=k)
    return [(doc, _normalize_distance_score(float(score))) for doc, score in results]


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


def _score_label(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


def _time_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _log_router_decision(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
) -> None:
    if not DEBUG_ROUTER:
        return

    separator = "=" * 60
    lines = [
        separator,
        "ROUTER DECISION",
        separator,
        "Question:",
        question,
        "",
        "Retriever Available:",
        "Yes" if retriever_available else "No",
    ]

    if retriever_available:
        lines.extend(["", f"Retrieved Chunks: {len(decision.docs)}"])

        if decision.docs:
            for index, doc in enumerate(decision.docs, start=1):
                metadata = getattr(doc, "metadata", None) or {}
                page = _page_label(metadata.get("page"))
                score = decision.scores[index - 1] if index - 1 < len(decision.scores) else None
                lines.append(
                    f"{index}. {_source_name(doc)} | Page {page} | Score: {_score_label(score)}"
                )
        else:
            lines.append("None")

        if not decision.used_scores:
            lines.extend(
                [
                    "",
                    "Similarity Scores:",
                    "N/A",
                    "",
                    "Scoring:",
                    "Similarity scores unavailable; routing used conservative scoreless confidence.",
                ]
            )

        lines.extend(
            [
                "",
                "Threshold:",
                "N/A" if decision.threshold is None else f"{decision.threshold:.4f}",
                "",
                "Best Score:",
                _score_label(decision.best_score),
                "",
                "Retrieval Time:",
                _time_label(decision.retrieval_time_ms),
            ]
        )

    lines.extend(
        [
            "",
            "Total Routing Time:",
            _time_label(decision.total_routing_time_ms),
            "",
            "Decision:",
            decision.route.upper(),
            "",
            "Reason:",
            decision.reason,
            "",
            separator,
        ]
    )

    print("\n".join(lines), flush=True)


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
    scored_results: list[tuple[Any, float]],
    threshold: float,
) -> list[tuple[Any, float]]:
    return [(doc, score) for doc, score in scored_results if score >= threshold]


def route_query(
    retriever: Any | None,
    question: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RouteDecision:
    """Route by retrieval quality, not by user wording.

    The router owns the single retrieval pass. Its returned docs are reused by
    the RAG chain so logging never causes a second FAISS/vector-store search.
    """
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
            scored_results = _retrieve_with_relevance_scores(vector_store, question, k)
            if scored_results is None:
                scored_results = _retrieve_with_raw_scores(vector_store, question, k)

            if scored_results is not None:
                retrieval_time_ms = (perf_counter() - retrieval_started_at) * 1000
                best_score = max((score for _, score in scored_results), default=None)
                qualified_results = _qualified_scored_results(scored_results, similarity_threshold)

                if qualified_results:
                    docs = [doc for doc, _ in qualified_results]
                    scores = [score for _, score in qualified_results]
                    return _finish(
                        question,
                        True,
                        RouteDecision(
                            "rag",
                            "Relevant document chunks exceeded the routing threshold.",
                            docs=docs,
                            scores=scores,
                            best_score=best_score,
                            used_scores=True,
                            threshold=similarity_threshold,
                            retrieval_time_ms=retrieval_time_ms,
                        ),
                        started_at,
                    )

                docs = [doc for doc, _ in scored_results]
                scores = [score for _, score in scored_results]
                return _finish(
                    question,
                    True,
                    RouteDecision(
                        "llm",
                        "No sufficiently relevant chunks found.",
                        docs=docs,
                        scores=scores,
                        best_score=best_score,
                        used_scores=True,
                        threshold=similarity_threshold,
                        retrieval_time_ms=retrieval_time_ms,
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
                ),
                started_at,
            )

        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                f"Similarity scores unavailable and scoreless retrieval confidence was weak ({confidence:.4f}).",
                docs=docs,
                scores=[None] * len(docs),
                best_score=None,
                used_scores=False,
                threshold=MIN_SCORELESS_CONFIDENCE,
                retrieval_time_ms=retrieval_time_ms,
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



