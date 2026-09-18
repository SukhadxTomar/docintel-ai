from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from app.agents.orchestrator import AgenticOrchestrator
from app.core.config import settings
from app.utils.doc_utils import (
    context_length,
    indexed_chunks,
    page_label,
    source_name,
    vector_store_from_retriever,
)
from app.utils.logger import log

from .llm_chain import create_llm_chain, stream_llm_response
from .rag_chain import create_rag_chain, stream_rag_response
from .router import RouteDecision, route_query


def _get_context_preview(docs: list[Any]) -> str:
    """Return a short preview of the retrieved context for logging."""
    context = "\n\n".join(
        getattr(doc, "page_content", "") or ""
        for doc in docs
    )
    return context.replace("\n", " ")[:300]


def _get_sources(decision: RouteDecision | None) -> list[dict[str, str]]:
    """Build a unique list of document sources used by the RAG response."""
    if decision is None or not decision.docs:
        return []

    sources: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for doc in decision.docs:
        metadata = getattr(doc, "metadata", None) or {}

        name = source_name(doc)
        page = page_label(metadata.get("page"))
        source_key = (name, page)

        if source_key in seen:
            continue

        seen.add(source_key)
        sources.append({
            "name": name,
            "page": page,
        })

    return sources


@dataclass
class HybridChatChain:
    """Route user questions between general LLM chat and PDF-based RAG."""

    retriever: Any | None = None
    last_decision: RouteDecision | None = None

    def __post_init__(self) -> None:
        self.llm_chain = create_llm_chain()
        self.rag_chain = create_rag_chain()
        self._orchestrator: AgenticOrchestrator | None = None

    def _get_orchestrator(self) -> AgenticOrchestrator | None:
        """Create the agentic orchestrator lazily when retrieval is available."""
        if self.retriever is None:
            return None

        if self._orchestrator is None:
            self._orchestrator = AgenticOrchestrator(self.retriever)

        return self._orchestrator

    def _route(
        self,
        question: str,
        chat_history: str,
    ) -> RouteDecision | None:
        """
        Decide whether the question should use RAG or the general LLM.

        Agentic routing is used when enabled. If the agentic layer fails,
        the legacy router is used as a fallback. If routing itself fails,
        the caller falls back to the general LLM.
        """
        if settings.agentic_rag_enabled:
            orchestrator = self._get_orchestrator()

            if orchestrator is not None:
                try:
                    return orchestrator.run(question, chat_history)
                except Exception as exc:
                    log.error(
                        "Agentic orchestration failed; "
                        f"falling back to legacy router: {exc}"
                    )

        try:
            return route_query(self.retriever, question)
        except Exception as exc:
            log.error(
                f"Router failed; falling back to general LLM: {exc}"
            )
            return None

    def stream(
        self,
        inputs: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """
        Stream the response for a single user question.

        Events:
            {"type": "token", "text": "..."}
            {"type": "final", "mode": "rag"|"llm", "sources": [...]}

        The final event contains the routing information and sources,
        so callers do not need to rely on shared instance state.
        """
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", "")

        request_id = log.get_request_id() or log.new_request_id()
        started_at = perf_counter()

        response_chunks: list[str] = []
        actual_route = "llm"

        log.section("Before Router")
        log.kv("Question", question)
        log.kv("Retriever is None", self.retriever is None)
        log.kv(
            "Vector Store Exists",
            "YES"
            if vector_store_from_retriever(self.retriever) is not None
            else "NO",
        )
        log.kv("Indexed Chunks", indexed_chunks(self.retriever))

        decision = self._route(question, chat_history)
        self.last_decision = decision

        try:
            if (
                decision is not None
                and decision.route == "rag"
                and decision.docs
            ):
                try:
                    actual_route = "rag"

                    log.section("Before RAG Chain")
                    log.kv("Route Selected", decision.route.upper())
                    log.kv(
                        "Documents Passed To RAG",
                        len(decision.docs),
                    )
                    log.kv(
                        "Context Length",
                        context_length(decision.docs),
                    )
                    log.kv(
                        "Context Preview",
                        _get_context_preview(decision.docs),
                    )

                    for chunk in stream_rag_response(
                        self.rag_chain,
                        decision.docs,
                        question,
                        chat_history,
                    ):
                        response_chunks.append(chunk)
                        yield {
                            "type": "token",
                            "text": chunk,
                        }

                    yield {
                        "type": "final",
                        "mode": "rag",
                        "sources": _get_sources(decision),
                    }
                    return

                except Exception as exc:
                    actual_route = "llm"
                    log.error(
                        f"RAG chain failed; falling back to LLM: {exc}"
                    )

            else:
                log.section("Before RAG Chain")

                selected_route = (
                    decision.route.upper()
                    if decision is not None
                    else "UNKNOWN"
                )

                log.kv("Route Selected", selected_route)
                log.kv(
                    "Documents Passed To RAG",
                    0 if decision is None else len(decision.docs),
                )
                log.kv("Context Length", 0)

                if decision is None:
                    reason = "Router failed before returning a decision."
                else:
                    reason = "Router did not select RAG."

                log.kv("Why No RAG Documents", reason)

            log.section("Before LLM Chain")
            log.kv("Route Selected", "LLM")
            log.kv(
                "Was Retrieval Attempted",
                "NO" if self.retriever is None else "YES",
            )

            if decision is None:
                log.kv(
                    "Why RAG Rejected",
                    "Router failed before returning a decision.",
                )

                if self.retriever is None:
                    log.kv(
                        "Why Retrieval Was Not Attempted",
                        "HybridChatChain.retriever is None.",
                    )
            else:
                if decision.route != "rag":
                    rejection_reason = decision.reason
                else:
                    rejection_reason = (
                        "RAG was selected, but the RAG chain failed "
                        "before completion."
                    )

                log.kv("Why RAG Rejected", rejection_reason)
                log.kv(
                    "Retrieved Chunks",
                    len(decision.retrieved_docs),
                )
                log.kv(
                    "Best Similarity Score",
                    (
                        decision.best_score
                        if decision.best_score is not None
                        else "N/A"
                    ),
                )
                log.kv(
                    "Threshold",
                    (
                        decision.threshold
                        if decision.threshold is not None
                        else "N/A"
                    ),
                )

            for chunk in stream_llm_response(
                self.llm_chain,
                question,
                chat_history,
            ):
                response_chunks.append(chunk)
                yield {
                    "type": "token",
                    "text": chunk,
                }

            yield {
                "type": "final",
                "mode": "llm",
                "sources": [],
            }

        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            response_text = "".join(response_chunks)

            log.section("Chat Chain")
            log.kv("Request ID", request_id)
            log.kv("Question", question)
            log.kv("Route", actual_route.upper())
            log.kv(
                "Documents Used",
                (
                    len(decision.docs)
                    if decision is not None and actual_route == "rag"
                    else 0
                ),
            )
            log.kv("Response Length", len(response_text))
            log.kv("Response Time", f"{elapsed_ms:.2f} ms")

    def invoke(self, inputs: dict[str, Any]) -> str:
        """Run the chain and return the complete response as a string."""
        return "".join(
            event["text"]
            for event in self.stream(inputs)
            if event.get("type") == "token"
        )


def create_chat_chain(
    chunks: list[Any] | None = None,
):
    """Create the hybrid chat chain and its optional retriever."""
    retriever = None

    if chunks:
        from app.retrievers.retriever import create_retriever

        retriever = create_retriever(chunks)

    chain = HybridChatChain(retriever=retriever)

    return chain, retriever