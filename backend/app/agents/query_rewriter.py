"""Query rewriting: one improved retrieval query per self-healing retry.

When the evidence evaluator judges a retrieval insufficient, the orchestrator asks
for **one** rewritten query and tries again (bounded by
``AGENTIC_MAX_RETRIEVAL_ATTEMPTS``). Rewriting is how the agent compensates for
BM25's exact-token matching *without changing BM25 itself* — e.g. a query for
"deduction" never lexically matches a chunk that says "deductions", so a rewrite
that adds the plural (and synonyms, expanded abbreviations, named entities) gives
the next retrieval a real chance.

Two layers:

1. One low-temperature LLM call that reads the original question, the current
   query, and what the evaluator said was missing, and returns a single improved
   query.
2. A dependency-free **morphological fallback** when the LLM is unavailable or
   returns nothing usable: it augments the current query with singular/plural
   variants of its distinctive terms — directly targeting the exact-token gap.

Crucially, the rewriter returns ``""`` when it cannot produce anything *new*
(empty, or a duplicate of the original / current / an already-tried query). The
orchestrator treats ``""`` as "stop retrying", which is what guarantees the loop
never repeats a retrieval and never spins forever. ``original_query`` is never
touched — only the next ``current_query`` is produced here.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.logger import log

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Short/common words not worth generating morphological variants for in the fallback.
_SKIP_VARIANT = frozenset(
    {"the", "and", "for", "with", "what", "which", "does", "did", "are", "was", "how"}
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You rewrite a search query to retrieve better passages from a hybrid keyword +
vector search over a user's uploaded documents. The previous query did not
retrieve enough evidence to answer the user's question.

Produce ONE improved search query that is more likely to match the relevant
passages. Useful techniques:
- add synonyms and domain-specific terminology,
- expand or contract abbreviations (both forms if helpful),
- include singular AND plural forms of key terms,
- name specific entities, figures, or section titles implied by the question,
- keep distinctive keywords; drop filler.

Output ONLY the rewritten query text on a single line — no quotes, no label, no
explanation.
            """.strip(),
        ),
        (
            "human",
            "User's original question:\n{original}\n\n"
            "Current search query (insufficient):\n{current}\n\n"
            "What was missing:\n{missing}\n\n"
            "Queries already tried (do NOT repeat any of these):\n{tried}\n\n"
            "Rewritten query:",
        ),
    ]
)


def _normalize(text: str) -> str:
    """Lowercased, whitespace-collapsed form for duplicate detection."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_new(candidate: str, seen: list[str]) -> bool:
    """True if ``candidate`` is non-empty and not a duplicate of anything in ``seen``."""
    norm = _normalize(candidate)
    return bool(norm) and norm not in {_normalize(s) for s in seen}


def _morphological_variants(token: str) -> set[str]:
    """Naive singular/plural variants of one lowercase token (BM25 token gap)."""
    variants: set[str] = set()
    if len(token) <= 2 or token in _SKIP_VARIANT:
        return variants
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 3:
        variants.add(token[:-2])
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    if token.endswith(("s", "x", "z")) or token.endswith(("ch", "sh")):
        variants.add(token + "es")
    else:
        variants.add(token + "s")
    if token.endswith("y") and len(token) > 2 and token[-2] not in "aeiou":
        variants.add(token[:-1] + "ies")
    variants.discard(token)
    return variants


def _heuristic_rewrite(current_query: str) -> str:
    """Augment the current query with morphological variants of its terms.

    Adds only variants not already present, so BM25 gets another exact-token shot
    (deduction → deductions). Returns ``""`` if no new variant can be produced.
    """
    tokens = _WORD_RE.findall(current_query.lower())
    if not tokens:
        return ""
    present = set(tokens)
    additions: list[str] = []
    for token in tokens:
        for variant in _morphological_variants(token):
            if variant not in present and variant not in additions:
                additions.append(variant)
    if not additions:
        return ""
    return f"{current_query.strip()} {' '.join(additions)}".strip()


def rewrite_query(state: Any, evidence: Any, *, llm: Any | None = None) -> str:
    """Produce ONE improved query for the next retrieval attempt.

    Reads ``state.original_query`` / ``state.current_query`` / ``state.rewritten_queries``
    and ``evidence.missing_information``. Returns ``""`` when it cannot produce a
    genuinely new query, signalling the orchestrator to stop retrying.
    ``state.original_query`` is never modified here.
    """
    original = getattr(state, "original_query", "") or ""
    current = getattr(state, "current_query", "") or original
    tried = [original, current, *getattr(state, "rewritten_queries", [])]
    missing = getattr(evidence, "missing_information", "") or "(not specified)"

    try:
        if llm is None:
            from app.models.llm_model import load_agent_llm

            llm = load_agent_llm()
        chain = _REWRITE_PROMPT | llm | StrOutputParser()
        raw = chain.invoke(
            {
                "original": original,
                "current": current,
                "missing": missing,
                "tried": "\n".join(f"- {q}" for q in dict.fromkeys(tried) if q) or "(none)",
            }
        )
        # Take the first non-empty line and strip stray quoting/labels.
        candidate = ""
        for line in (raw or "").splitlines():
            line = line.strip().strip('"').strip("'").strip()
            if line.lower().startswith(("rewritten query:", "query:")):
                line = line.split(":", 1)[1].strip()
            if line:
                candidate = line
                break
        if _is_new(candidate, tried):
            return candidate
    except Exception as exc:  # noqa: BLE001 — rewriting must never break a turn
        log.warning(f"[AGENT] Query rewrite via LLM unavailable; using heuristic variant: {exc}")

    # Fallback (LLM failed, or produced empty/duplicate): morphological expansion.
    heuristic = _heuristic_rewrite(current)
    return heuristic if _is_new(heuristic, tried) else ""
