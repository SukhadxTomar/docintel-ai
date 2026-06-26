from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from retrievers.retriever import create_retriever

from .llm_chain import create_llm_chain, stream_llm_response
from .rag_chain import create_rag_chain, stream_rag_response
from .router import route_query


@dataclass
class HybridChatChain:
    """Routes each question to either general LLM chat or PDF RAG."""

    retriever: Any | None = None

    def __post_init__(self) -> None:
        self.llm_chain = create_llm_chain()
        self.rag_chain = create_rag_chain()

    def stream(self, inputs: dict[str, Any]) -> Iterator[str]:
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", "")

        # The router performs retrieval once and returns the chunks to reuse.
        try:
            decision = route_query(self.retriever, question)
        except Exception:
            decision = None

        if decision is not None and decision.route == "rag":
            try:
                yield from stream_rag_response(
                    self.rag_chain,
                    decision.docs,
                    question,
                    chat_history,
                )
                return
            except Exception:
                pass

        yield from stream_llm_response(self.llm_chain, question, chat_history)

    def invoke(self, inputs: dict[str, Any]) -> str:
        return "".join(self.stream(inputs))


def create_chat_chain(chunks: list[Any] | None = None):
    retriever = create_retriever(chunks) if chunks else None
    chain = HybridChatChain(retriever=retriever)
    return chain, retriever
