from __future__ import annotations

import os
from typing import Any, Iterable, Iterator

from langchain_core.output_parsers import StrOutputParser

from models.llm_model import load_llm
from prompts.chatbot_prompt import chat_prompt

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

    payload = {
        "question": question,
        "chat_history": chat_history,
        "context": format_docs(docs),
    }

    for chunk in chain.stream(payload):
        text = chunk_to_text(chunk)
        if text:
            yield text
