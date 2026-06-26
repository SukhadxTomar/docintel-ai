from __future__ import annotations

from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models.llm_model import load_llm


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
    payload = {"question": question, "chat_history": chat_history}

    for chunk in chain.stream(payload):
        text = chunk_to_text(chunk)
        if text:
            yield text
