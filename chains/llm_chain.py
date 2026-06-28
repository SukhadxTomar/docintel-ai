from __future__ import annotations

from time import perf_counter
from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.llm_model import load_llm
from utils.logger import log


GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful, professional AI assistant.
Answer general questions directly and clearly.
Use the conversation history only when it is relevant.

Conversation history:
{chat_history}
            """.strip(),
        ),
        ("human", "{question}"),
    ]
)


def chunk_to_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(chunk, dict):
        for key in ("answer", "output", "text", "content"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value

    return str(chunk)


def create_llm_chain():
    return GENERAL_PROMPT | load_llm() | StrOutputParser()


def stream_llm_response(chain: Any, question: str, chat_history: str) -> Iterator[str]:
    started_at = perf_counter()
    first_token_latency_ms = None
    generated_chars = 0

    log.info("Entering LLM Chain")
    log.kv("Conversation History Length", len(chat_history.splitlines()))

    payload = {"question": question, "chat_history": chat_history}
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
