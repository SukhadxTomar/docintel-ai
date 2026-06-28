from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Iterable, Iterator

from langchain_core.output_parsers import StrOutputParser

from models.llm_model import load_llm
from prompts.chatbot_prompt import chat_prompt
from utils.logger import log

from .llm_chain import chunk_to_text


class MissingRetrieverError(RuntimeError):
    pass


def create_rag_chain():
    return chat_prompt | load_llm() | StrOutputParser()


def _page_label(page: Any) -> str:
    if page is None:
        return "Unknown"

    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(page)


def format_docs(docs: Iterable[Any]) -> str:
    formatted_docs = []

    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown")
        source_name = metadata.get("original_name") or os.path.basename(str(source))
        page = _page_label(metadata.get("page"))

        formatted_docs.append(
            f"Source: {source_name}, Page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


def stream_rag_response(
    chain: Any,
    docs: list[Any],
    question: str,
    chat_history: str,
) -> Iterator[str]:
    if not docs:
        return

    started_at = perf_counter()
    first_token_latency_ms = None
    generated_chars = 0

    log.info("Entering RAG Chain")
    log.kv("Files Used", len({getattr(doc.metadata, "get", lambda *_: None)("source") for doc in docs}))
    log.kv("Pages Used", len({getattr(doc.metadata, "get", lambda *_: None)("page") for doc in docs}))

    payload = {
        "question": question,
        "chat_history": chat_history,
        "context": format_docs(docs),
    }

    context_preview = payload["context"].replace("\n", " ")[:300]
    log.kv("Context Size", len(payload["context"]))
    log.kv("Context Preview", context_preview)
    log.success("Prompt Created")
    log.info("Streaming Started")

    for chunk in chain.stream(payload):
        text = chunk_to_text(chunk)
        if text:
            if first_token_latency_ms is None:
                first_token_latency_ms = (perf_counter() - started_at) * 1000
            generated_chars += len(text)
            yield text

    log.kv("First Token Latency", f"{first_token_latency_ms:.2f} ms" if first_token_latency_ms is not None else "N/A")
    log.success("Streaming Finished")
    log.kv("Characters Generated", generated_chars)

