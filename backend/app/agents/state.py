"""Lightweight per-request state for the agentic orchestration layer.

One :class:`AgentState` is created per user query and threaded through the
orchestrator's classify → retrieve → evaluate → rewrite → retry loop. It is a
plain mutable dataclass (no behaviour, no I/O) — just the running record of what
the agent has tried and decided, so the loop stays readable and every decision
can be logged without recomputing anything.

The one invariant that matters: **``original_query`` is the exact text the human
typed and never changes.** Query rewriting only ever reassigns ``current_query``
(and appends to ``rewritten_queries``). This is what lets the agent retrieve with
an improved query while the final answer still answers what the user actually
asked — see ``orchestrator.py`` and the RAG answer step, which always use
``original_query``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable per-request record threaded through the orchestration loop."""

    # The human's exact question. Frozen by convention — never reassigned.
    original_query: str
    # The query used for the *current* retrieval attempt. Starts equal to
    # ``original_query``; only query rewriting changes it.
    current_query: str
    # 1-based index of the retrieval attempt in progress (0 before the first).
    attempt_number: int = 0
    # One entry per retrieval attempt (the HybridRetrievalResult), in order.
    retrieval_results: list[Any] = field(default_factory=list)
    # The most recent evidence verdict (an EvidenceEvaluation). Concise metadata
    # only — never raw chunk text or chain-of-thought.
    evidence_assessment: Any | None = None
    # Every rewritten query the agent generated, in order (excludes the original).
    rewritten_queries: list[str] = field(default_factory=list)
    # The chunks chosen as the answer context (supporting evidence first).
    final_documents: list[Any] = field(default_factory=list)
    # The resolved route for this query: "rag" or "llm".
    route: str = "llm"
