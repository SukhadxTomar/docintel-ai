from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

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
    best_score: float | None = None
    used_scores: bool = False


def _top_k(retriever: Any) -> int:
    search_kwargs = getattr(retriever, "search_kwargs", None) or {}
    return int(search_kwargs.get("k") or DEFAULT_TOP_K)


def _has_meaningful_content(docs: list[Any]) -> bool:
    return any(getattr(doc, "page_content", "").strip() for doc in docs)


def _normalize_distance_score(score: float) -> float:
    """Convert distance-style vector scores into a 0..1 relevance score."""
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


def route_query(
    retriever: Any | None,
    question: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> RouteDecision:
    """Route by retrieval quality, not by user wording.

    If a vector store can return relevance scores, the best score controls whether
    RAG is used. Otherwise, any meaningful retrieved chunks are treated as enough
    evidence to use RAG.
    """
    if retriever is None:
        return RouteDecision("llm", "No retriever is available.")

    k = _top_k(retriever)
    vector_store = _vector_store(retriever)

    if vector_store is not None:
        scored_results = _retrieve_with_relevance_scores(vector_store, question, k)
        if scored_results is None:
            scored_results = _retrieve_with_raw_scores(vector_store, question, k)

        if scored_results is not None:
            docs = [doc for doc, _ in scored_results]
            best_score = max((score for _, score in scored_results), default=None)

            if best_score is not None and best_score >= similarity_threshold:
                return RouteDecision(
                    "rag",
                    "Retrieved chunks passed the similarity threshold.",
                    docs=docs,
                    best_score=best_score,
                    used_scores=True,
                )

            return RouteDecision(
                "llm",
                "Retrieved chunks did not pass the similarity threshold.",
                docs=docs,
                best_score=best_score,
                used_scores=True,
            )

    docs = _retrieve_without_scores(retriever, question)
    if _has_meaningful_content(docs):
        return RouteDecision(
            "rag",
            "Retriever returned meaningful chunks without scores.",
            docs=docs,
            used_scores=False,
        )

    return RouteDecision(
        "llm",
        "Retriever returned no meaningful chunks.",
        docs=docs,
        used_scores=False,
    )
