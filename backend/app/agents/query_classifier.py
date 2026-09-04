"""Query understanding: does this question need the uploaded documents at all?

The first decision the orchestrator makes. The goal is narrow and deliberately
conservative: skip retrieval only for questions that are *clearly* general — a
greeting, small talk, or general knowledge plainly unrelated to any uploaded
document — and retrieve for everything else. Getting this wrong toward "general"
would silently drop a real document question, so the classifier is **biased
toward retrieval**: anything uncertain, and any failure, resolves to
``"document"``.

Two cheap layers, in order:

1. A dependency-free heuristic fast-path for obvious greetings / small talk, so
   "hi", "thanks", "who are you" never cost an LLM call.
2. One short, deterministic LLM classification for everything else, instructed to
   prefer ``DOCUMENT`` whenever unsure.

This is intentionally *not* a complex classifier (see the task's guidance) — it is
a lightweight gate that avoids obviously-wasted retrieval without risking real
document questions.
"""
from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.logger import log

QueryType = Literal["document", "general"]

# Tokens that, on their own, make a message pure greeting/acknowledgement filler.
_GREETING_TOKENS: frozenset[str] = frozenset(
    {
        "hi", "h111", "hello", "helo", "hey", "heya", "hiya", "yo", "greetings",
        "sup", "hola", "thanks", "thank", "thankyou", "ty", "thx", "cheers",
        "bye", "goodbye", "cya", "ok", "okay", "k", "cool", "nice", "great",
        "awesome", "please", "there", "morning", "afternoon", "evening",
    }
)

# Whole-message small-talk phrases that are clearly not about any document.
_SMALLTALK_PHRASES: frozenset[str] = frozenset(
    {
        "how are you", "how are you doing", "how is it going", "hows it going",
        "who are you", "what are you", "what can you do", "what do you do",
        "help", "good morning", "good afternoon", "good evening", "good night",
        "nice to meet you", "thank you", "thanks a lot", "thank you very much",
        "test", "testing",
    }
)

_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a routing classifier for an assistant that answers questions from a
user's uploaded documents. Decide whether answering the user's message requires
looking inside those uploaded documents.

Reply with EXACTLY one word, nothing else:
- DOCUMENT — the message asks about specific facts, figures, terms, policies, or
  content that could plausibly live in an uploaded document.
- GENERAL — the message is clearly general chit-chat or general world knowledge
  unrelated to any uploaded document (greetings, common definitions, math, trivia).

When you are unsure, reply DOCUMENT.
            """.strip(),
        ),
        ("human", "Message: {question}\nAnswer:"),
    ]
)


def _normalize(question: str) -> str:
    """Lowercase, strip punctuation to bare words + single spaces (for matching)."""
    lowered = question.strip().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", lowered)).strip()


def is_smalltalk(question: str) -> bool:
    """True for obvious greetings / small talk that never need retrieval.

    Conservative on purpose: fires only when the *entire* message is greeting
    tokens or an exact small-talk phrase, so a real question that merely opens
    with "hi" still goes to the LLM classifier (and, when unsure, to retrieval).
    """
    normalized = _normalize(question)
    if not normalized:
        return True  # empty / punctuation-only → nothing to retrieve for
    if normalized in _SMALLTALK_PHRASES:
        return True
    tokens = normalized.split()
    return bool(tokens) and all(token in _GREETING_TOKENS for token in tokens)


def classify_query(
    question: str, *, has_documents: bool, llm: Any | None = None
) -> QueryType:
    """Classify whether ``question`` needs document retrieval.

    Returns ``"general"`` only when there are no documents, the message is
    obvious small talk, or the LLM is confident it is general knowledge.
    Everything else — including any ambiguity or any failure — returns
    ``"document"`` so a real question is never denied retrieval.
    """
    if not has_documents:
        return "general"
    if is_smalltalk(question):
        return "general"

    try:
        if llm is None:
            from app.models.llm_model import load_agent_llm

            llm = load_agent_llm()
        chain = _CLASSIFY_PROMPT | llm | StrOutputParser()
        answer = (chain.invoke({"question": question}) or "").strip().upper()
        # Bias to retrieval: only an explicit, unambiguous GENERAL skips it.
        if "GENERAL" in answer and "DOCUMENT" not in answer:
            return "general"
        return "document"
    except Exception as exc:  # noqa: BLE001 — never let classification break a turn
        log.warning(f"[AGENT] Query classification failed; defaulting to document retrieval: {exc}")
        return "document"
