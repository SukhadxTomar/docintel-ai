from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

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


def _context_preview(docs: list[Any]) -> str:
    context = "\n\n".join(getattr(doc, "page_content", "") or "" for doc in docs)
    return context.replace("\n", " ")[:300]


def _sources_from_decision(decision: RouteDecision | None) -> list[dict[str, str]]:
    """Distinct ``{name, page}`` sources for the RAG documents that answered the query."""
    if decision is None or not getattr(decision, "docs", None):
        return []

    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for doc in decision.docs:
        metadata = getattr(doc, "metadata", None) or {}
        entry = (source_name(doc), page_label(metadata.get("page")))
        if entry not in seen:
            seen.add(entry)
            sources.append({"name": entry[0], "page": entry[1]})
    return sources


@dataclass
class HybridChatChain:
    """Routes each question to either general LLM chat or PDF RAG."""

    retriever: Any | None = None
    last_decision: RouteDecision | None = None

    def __post_init__(self) -> None:
        self.llm_chain = create_llm_chain()
        self.rag_chain = create_rag_chain()

    def stream(self, inputs: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yield streaming events for one question.

        Each yielded value is a dict:
          - ``{"type": "token", "text": <str>}``  one per generated text delta
          - ``{"type": "final", "mode": "rag"|"llm", "sources": [{"name","page"}...]}``
            emitted exactly once, after the answer finishes.

        The routing decision travels *in* the stream (the ``final`` event) rather
        than via shared instance state, so a single chain instance can be streamed
        concurrently without callers racing on ``last_decision``. The attribute is
        still set for logging continuity but must not be read across requests.
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
        log.kv("Vector Store Exists", "YES" if vector_store_from_retriever(self.retriever) is not None else "NO")
        log.kv("Indexed Chunks", indexed_chunks(self.retriever))

        try:
            decision = route_query(self.retriever, question)
        except Exception as exc:
            log.error(f"Router failed; falling back to LLM: {exc}")
            decision = None

        self.last_decision = decision

        try:
            if decision is not None and decision.route == "rag" and decision.docs:
                try:
                    actual_route = "rag"
                    log.section("Before RAG Chain")
                    log.kv("Route Selected", decision.route.upper())
                    log.kv("Documents Passed To RAG", len(decision.docs))
                    log.kv("Context Length", context_length(decision.docs))
                    log.kv("Context Preview", _context_preview(decision.docs))
                    for chunk in stream_rag_response(
                        self.rag_chain,
                        decision.docs,
                        question,
                        chat_history,
                    ):
                        response_chunks.append(chunk)
                        yield {"type": "token", "text": chunk}
                    yield {"type": "final", "mode": "rag", "sources": _sources_from_decision(decision)}
                    return
                except Exception as exc:
                    actual_route = "llm"
                    log.error(f"RAG chain failed; falling back to LLM: {exc}")
            else:
                log.section("Before RAG Chain")
                log.kv("Route Selected", getattr(decision, "route", "unknown").upper() if decision is not None else "UNKNOWN")
                log.kv("Documents Passed To RAG", 0 if decision is None else len(decision.docs))
                log.kv("Context Length", 0)
                log.kv(
                    "Why No RAG Documents",
                    "Router did not select RAG."
                    if decision is not None
                    else "Router failed before returning a decision.",
                )

            log.section("Before LLM Chain")
            log.kv("Route Selected", "LLM")
            log.kv("Was Retrieval Attempted", "NO" if self.retriever is None else "YES")
            if decision is None:
                log.kv("Why RAG Rejected", "Router failed before returning a decision.")
                if self.retriever is None:
                    log.kv("Why Retrieval Was Not Attempted", "HybridChatChain.retriever is None.")
            else:
                log.kv("Why RAG Rejected", decision.reason if decision.route != "rag" else "RAG selected but RAG chain failed before completion.")
                log.kv("Retrieved Chunks", len(decision.retrieved_docs))
                log.kv("Best Similarity Score", decision.best_score if decision.best_score is not None else "N/A")
                log.kv("Threshold", decision.threshold if decision.threshold is not None else "N/A")
            for chunk in stream_llm_response(self.llm_chain, question, chat_history):
                response_chunks.append(chunk)
                yield {"type": "token", "text": chunk}
            yield {"type": "final", "mode": "llm", "sources": []}

        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            response_text = "".join(response_chunks)
            log.section("Chat Chain")
            log.kv("Request ID", request_id)
            log.kv("Question", question)
            log.kv("Route", actual_route.upper())
            log.kv("Documents Used", len(decision.docs) if decision is not None and actual_route == "rag" else 0)
            log.kv("Response Length", len(response_text))
            log.kv("Response Time", f"{elapsed_ms:.2f} ms")

    def invoke(self, inputs: dict[str, Any]) -> str:
        return "".join(
            event["text"]
            for event in self.stream(inputs)
            if event.get("type") == "token"
        )


def create_chat_chain(chunks: list[Any] | None = None):
    if chunks:
        from app.retrievers.retriever import create_retriever

        retriever = create_retriever(chunks)
    else:
        retriever = None

    chain = HybridChatChain(retriever=retriever)
    return chain, retriever
