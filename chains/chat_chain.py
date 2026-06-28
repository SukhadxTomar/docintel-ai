from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

from utils.logger import log

from .llm_chain import create_llm_chain, stream_llm_response
from .rag_chain import create_rag_chain, stream_rag_response
from .router import RouteDecision, route_query


@dataclass
class HybridChatChain:
    """Routes each question to either general LLM chat or PDF RAG."""

    retriever: Any | None = None
    last_decision: RouteDecision | None = None

    def __post_init__(self) -> None:
        self.llm_chain = create_llm_chain()
        self.rag_chain = create_rag_chain()

    def stream(self, inputs: dict[str, Any]) -> Iterator[str]:
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", "")
        started_at = perf_counter()
        response_chunks: list[str] = []
        actual_route = "llm"

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
                    for chunk in stream_rag_response(
                        self.rag_chain,
                        decision.docs,
                        question,
                        chat_history,
                    ):
                        response_chunks.append(chunk)
                        yield chunk
                    return
                except Exception as exc:
                    actual_route = "llm"
                    log.error(f"RAG chain failed; falling back to LLM: {exc}")

            for chunk in stream_llm_response(self.llm_chain, question, chat_history):
                response_chunks.append(chunk)
                yield chunk

        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            response_text = "".join(response_chunks)
            log.section("Chat Chain")
            log.kv("Question", question)
            log.kv("Route", actual_route.upper())
            log.kv("Documents Used", len(decision.docs) if decision is not None and actual_route == "rag" else 0)
            log.kv("Response Length", len(response_text))
            log.kv("Response Time", f"{elapsed_ms:.2f} ms")

    def invoke(self, inputs: dict[str, Any]) -> str:
        return "".join(self.stream(inputs))


def create_chat_chain(chunks: list[Any] | None = None):
    if chunks:
        from retrievers.retriever import create_retriever

        retriever = create_retriever(chunks)
    else:
        retriever = None

    chain = HybridChatChain(retriever=retriever)
    return chain, retriever
