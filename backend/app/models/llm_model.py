"""LLM client, backed by OpenRouter.

OpenRouter exposes an OpenAI-compatible API, so we drive it with LangChain's
``ChatOpenAI`` — pointing ``base_url`` at OpenRouter and passing the OpenRouter
key. To use a different model, change ``OPENROUTER_MODEL`` (see app.core.config);
no code change is needed.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings

_llm: ChatOpenAI | None = None  # cached, shared client (created on first load_llm() call)
_agent_llm: ChatOpenAI | None = None  # cached client for the agent's reasoning calls


def load_llm() -> ChatOpenAI:
    """Return a shared OpenRouter chat client, created once and reused."""
    global _llm

    if _llm is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env "
                "(get a key at https://openrouter.ai/keys)."
            )

        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            streaming=True,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            # Optional attribution headers OpenRouter uses for its rankings.
            default_headers={
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "DocIntel-AI",
            },
        )

    return _llm


def load_agent_llm() -> ChatOpenAI:
    """Return a shared OpenRouter client tuned for the agent's *reasoning* calls.

    The agentic layer (``app/agents/``) makes short, deterministic decisions —
    classify a query, judge evidence sufficiency, rewrite a query — that want
    exact, repeatable output rather than the creative, streamed prose the answer
    model produces. So this is a *separate* cached client at ``temperature=0`` and
    non-streaming (these calls are ``invoke``d for a single structured result, not
    streamed to the user). It reuses the same OpenRouter model/key/base as
    :func:`load_llm`; answer generation keeps using ``load_llm`` unchanged.
    """
    global _agent_llm

    if _agent_llm is None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to backend/.env "
                "(get a key at https://openrouter.ai/keys)."
            )

        _agent_llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            streaming=False,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "DocIntel-AI",
            },
        )

    return _agent_llm
