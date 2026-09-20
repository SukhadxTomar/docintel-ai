"""LLM client, backed by Groq.

Groq exposes an OpenAI-compatible API, so we keep the existing
``ChatOpenAI`` abstraction and point it at Groq with the Groq API key and
Groq base URL. To use a different model, change the configured ``llm_model``
value (see app.core.config); no application logic change is needed.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings

_llm: ChatOpenAI | None = None  # cached, shared client (created on first load_llm() call)
_agent_llm: ChatOpenAI | None = None  # cached client for the agent's reasoning calls


def load_llm() -> ChatOpenAI:
    """Return a shared Groq chat client, created once and reused."""
    global _llm

    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env "
                "(get a key at https://console.groq.com/keys)."
            )

        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            streaming=True,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

    return _llm


def load_agent_llm() -> ChatOpenAI:
    """Return a shared Groq client tuned for the agent's *reasoning* calls.

    The agentic layer (``app/agents/``) makes short, deterministic decisions —
    classify a query, judge evidence sufficiency, rewrite a query — that want
    exact, repeatable output rather than the creative, streamed prose the answer
    model produces. So this is a *separate* cached client at ``temperature=0`` and
    non-streaming (these calls are ``invoke``d for a single structured result, not
    streamed to the user). It reuses the same Groq model/key/base as
    :func:`load_llm`; answer generation keeps using ``load_llm`` unchanged.
    """
    global _agent_llm

    if _agent_llm is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env "
                "(get a key at https://console.groq.com/keys)."
            )

        _agent_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            streaming=False,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
        )

    return _agent_llm
