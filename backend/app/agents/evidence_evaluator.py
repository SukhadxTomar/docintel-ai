"""Evidence evaluation: can the retrieved chunks actually answer the ORIGINAL query?

This is the component that *replaces the FAISS hard threshold* as the RAG-vs-retry
decision. Instead of trusting a single similarity number, it asks the real
question — **"Can the user's original question be answered accurately using only
these retrieved excerpts?"** — by having a low-temperature LLM read the actual
chunk text, then blending that judgement with the deterministic hybrid signals
(``EvidenceAssessment`` from ``chains/router.py``: FAISS best, semantic/lexical/
agreement strength) as a safety net.

Design (see the approved plan):

* The **LLM judge is primary** — it reads content, so it can tell "same topic" from
  "answer actually present". It returns strict JSON:
  ``{sufficient, confidence, reason, missing_information, supporting_chunks}``.
* The **signals are the blend / fallback**, never a hard gate:
  - *Rescue* — a strong lexical match (the distinctive query terms are literally
    present in a top chunk) or two-plus corroborating signals can rescue a
    false-negative "insufficient" from the LLM. This is Example A: FAISS 0.28 but
    the requested metric is present → sufficient.
  - *Caution* — a low-confidence "sufficient" with **no** supporting signal is
    downgraded to a retry (``evidence_threshold`` is this guideline, not a score
    gate). A lone strong FAISS score can NOT force sufficiency on its own — that is
    Example B: FAISS 0.48 on the wrong topic → insufficient → rewrite.
  - On any LLM/JSON failure the verdict is derived deterministically from the
    signals, so evaluation never breaks a turn.

Only the concise verdict is ever logged — never chunk text and never
chain-of-thought (the prompt explicitly forbids step-by-step output).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.doc_utils import page_label, source_name
from app.utils.logger import log

# Per-chunk text handed to the judge is truncated — enough to decide sufficiency
# without shipping whole pages to the model (and it is never logged either way).
_MAX_CHUNK_CHARS = 600
# When strong deterministic signals rescue a hedged LLM "insufficient", floor the
# confidence here so the caution rule below can't immediately re-downgrade it.
_RESCUE_CONFIDENCE = 0.65


@dataclass(frozen=True)
class EvidenceEvaluation:
    """The blended verdict on whether the evidence can answer the original query.

    Concise decision metadata only — safe to log in full. ``supporting_chunk_indices``
    are 0-based indices into ``result.documents`` (the RRF top-k), used by the
    orchestrator to put the most relevant chunks first in the answer context.
    """

    sufficient: bool
    confidence: float  # 0..1, blended (LLM judgement tempered by the hybrid signals)
    reason: str  # one concise sentence; never chain-of-thought
    missing_information: str = ""  # what's absent (feeds the rewriter); "" if sufficient
    supporting_chunk_indices: list[int] = field(default_factory=list)
    source: str = "llm"  # "llm" (judge parsed) or "signals" (deterministic fallback)


_EVALUATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an evidence evaluator for a document question-answering system. You are
given the user's ORIGINAL question and a set of numbered excerpts retrieved from
their uploaded documents. Decide whether these excerpts contain enough information
to answer the ORIGINAL question accurately and specifically.

Judge ONLY from the excerpts — do not use outside knowledge. The decisive question
is: "Can the original question be answered accurately using ONLY these excerpts?"
Being about the same general topic is NOT enough; the specific answer must be
present.

Respond with a SINGLE JSON object and nothing else, in exactly this shape:
{{
  "sufficient": true or false,
  "confidence": a number from 0 to 1,
  "reason": "one short sentence",
  "missing_information": "if not sufficient, the specific info that is missing; otherwise an empty string",
  "supporting_chunks": [the 1-based numbers of the excerpts that directly help answer the question]
}}

Rules:
- "sufficient" is true only if the excerpts actually contain the answer.
- Keep "reason" to a single sentence. Do NOT include step-by-step reasoning.
- Output only the JSON object — no prose before or after it.
            """.strip(),
        ),
        (
            "human",
            "ORIGINAL QUESTION:\n{question}\n\n"
            "RETRIEVED EXCERPTS:\n{excerpts}\n\n"
            "RETRIEVAL SIGNALS (diagnostic only, do not over-rely on them):\n{signals}\n\n"
            "Return only the JSON object.",
        ),
    ]
)


def _clamp01(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _format_excerpts(documents: list[Any]) -> str:
    """Numbered, truncated chunk text for the judge (kept out of the logs)."""
    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        header = f"[{index}] {source_name(doc)} (page {page_label(metadata.get('page'))})"
        content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        if len(content) > _MAX_CHUNK_CHARS:
            content = content[:_MAX_CHUNK_CHARS].rstrip() + " …"
        lines.append(f"{header}\n{content}")
    return "\n\n".join(lines) if lines else "(no excerpts retrieved)"


def _format_signals(signals: Any) -> str:
    """One diagnostic line summarising the hybrid signals for the judge."""
    faiss_best = getattr(signals, "faiss_best", None)
    faiss_txt = "N/A" if faiss_best is None else f"{faiss_best:.3f}"
    return (
        f"faiss_best_relevance={faiss_txt}, "
        f"semantic_strong={bool(getattr(signals, 'semantic_strong', False))}, "
        f"lexical_strong={bool(getattr(signals, 'lexical_strong', False))}, "
        f"agreement_strong={bool(getattr(signals, 'agreement_strong', False))}"
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a single JSON object from a model response.

    Tolerant on purpose (OpenRouter models vary): try the whole string, then the
    first ``{`` … last ``}`` slice. Returns ``None`` if nothing parses.
    """
    if not text:
        return None
    for candidate in (text, _first_brace_block(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _first_brace_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _coerce_supporting(raw: Any, n_docs: int) -> list[int]:
    """Convert the model's 1-based ``supporting_chunks`` to valid 0-based indices."""
    if not isinstance(raw, (list, tuple)):
        return []
    seen: list[int] = []
    for item in raw:
        try:
            idx = int(item) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_docs and idx not in seen:
            seen.append(idx)
    return seen


def _signal_seed(result: Any, signals: Any) -> tuple[bool, float, str, str, list[int]]:
    """Deterministic verdict from the hybrid signals alone (LLM-free fallback)."""
    strong = sum(
        bool(getattr(signals, name, False))
        for name in ("semantic_strong", "lexical_strong", "agreement_strong")
    )
    sufficient = getattr(signals, "route", "llm") == "rag"
    confidence = min(0.5 + 0.15 * strong, 0.9) if sufficient else 0.3

    # Prefer chunks both retrievers surfaced; otherwise fall back to the top chunk.
    evidence = getattr(result, "evidence", {}) or {}
    final_keys = getattr(result, "final_keys", []) or []
    supporting = [
        i
        for i, key in enumerate(final_keys)
        if key in evidence and getattr(evidence[key], "in_both", False)
    ]
    if not supporting and getattr(result, "documents", None):
        supporting = [0]

    reason = (
        "Signal-based fallback: hybrid signals indicate relevant evidence."
        if sufficient
        else "Signal-based fallback: hybrid signals indicate weak or unrelated evidence."
    )
    missing = "" if sufficient else "more specific evidence that directly answers the question"
    return sufficient, confidence, reason, missing, supporting


def _blend(
    llm_sufficient: bool, llm_confidence: float, signals: Any, threshold: float
) -> tuple[bool, float]:
    """Reconcile the LLM verdict with the deterministic signals.

    Neither side is a hard gate. Strong signals can rescue a false "insufficient";
    a shaky "sufficient" with no signal support is downgraded to a retry. A lone
    strong FAISS score can never force sufficiency by itself.
    """
    strong = sum(
        bool(getattr(signals, name, False))
        for name in ("semantic_strong", "lexical_strong", "agreement_strong")
    )
    lexical_strong = bool(getattr(signals, "lexical_strong", False))

    sufficient = bool(llm_sufficient)
    confidence = _clamp01(llm_confidence, 0.5)

    # Rescue: literal keyword presence, or two+ corroborating signals, outweighs a
    # hedged LLM "no" (Example A: FAISS 0.28 + metric literally present → RAG).
    if lexical_strong or strong >= 2:
        sufficient = True
        confidence = max(confidence, _RESCUE_CONFIDENCE)

    # Caution: a low-confidence "yes" with NO supporting signal earns one retry
    # (Example B stays insufficient because the LLM already said no there).
    if sufficient and confidence < threshold and strong == 0:
        sufficient = False

    return sufficient, confidence


def evaluate_evidence(
    original_query: str,
    result: Any,
    signals: Any,
    *,
    llm: Any | None = None,
    threshold: float = 0.70,
) -> EvidenceEvaluation:
    """Judge whether ``result`` can answer ``original_query``; blend with signals.

    ``result`` is a ``HybridRetrievalResult``; ``signals`` is the diagnostic
    ``EvidenceAssessment`` from ``assess_evidence``. Returns a concise, log-safe
    :class:`EvidenceEvaluation`.
    """
    documents = list(getattr(result, "documents", []) or [])

    # No candidates at all → nothing to answer from; skip the LLM call entirely.
    if not documents:
        return EvidenceEvaluation(
            sufficient=False,
            confidence=0.0,
            reason="No document chunks were retrieved for this query.",
            missing_information="any relevant document content",
            supporting_chunk_indices=[],
            source="signals",
        )

    try:
        if llm is None:
            from app.models.llm_model import load_agent_llm

            llm = load_agent_llm()
        chain = _EVALUATE_PROMPT | llm | StrOutputParser()
        raw = chain.invoke(
            {
                "question": original_query,
                "excerpts": _format_excerpts(documents),
                "signals": _format_signals(signals),
            }
        )
        parsed = _extract_json(raw or "")
        if parsed is None:
            raise ValueError("evidence judge returned no parseable JSON object")

        llm_sufficient = bool(parsed.get("sufficient", False))
        llm_confidence = _clamp01(parsed.get("confidence"), 0.5)
        reason = str(parsed.get("reason", "")).strip()[:300] or "Evidence evaluated."
        missing = str(parsed.get("missing_information", "")).strip()[:300]
        supporting = _coerce_supporting(parsed.get("supporting_chunks"), len(documents))
        source = "llm"
    except Exception as exc:  # noqa: BLE001 — evaluation must never break a turn
        log.warning(f"[AGENT] Evidence judge unavailable; using signal-based fallback: {exc}")
        llm_sufficient, llm_confidence, reason, missing, supporting = _signal_seed(result, signals)
        source = "signals"

    sufficient, confidence = _blend(llm_sufficient, llm_confidence, signals, threshold)

    # If the blend flips a "yes" to "no", make sure the rewriter has a hint to act on.
    if not sufficient and not missing:
        missing = "more specific evidence that directly answers the question"

    return EvidenceEvaluation(
        sufficient=sufficient,
        confidence=confidence,
        reason=reason,
        missing_information="" if sufficient else missing,
        supporting_chunk_indices=supporting,
        source=source,
    )
