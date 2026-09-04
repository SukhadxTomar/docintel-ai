"""Query router: decide RAG vs. general LLM from *hybrid evidence*, and hand RAG
the fused context.

The router consumes a
:class:`~app.retrievers.hybrid_retriever.HybridRetrievalResult` produced by one
hybrid retrieval (BM25 + FAISS fused with RRF). It never re-runs a retriever.

Two responsibilities stay firmly separate (see the retriever module docstring):

* **RRF is the ranking mechanism** — it decides *which* chunks are the best
  context (``result.documents``, the top ``final_context_k`` fused chunks). Its
  score is a tiny rank-fusion value (~1/61) on an arbitrary scale and is never
  treated as a probability or thresholded.
* **Hybrid evidence is the routing mechanism** — it decides *whether* the document
  is relevant enough to answer from at all. This is what changed: routing no
  longer rests on the FAISS relevance score alone.

**Where this fits under the agentic layer.** When ``AGENTIC_RAG_ENABLED=true`` (the
default), the agentic orchestrator (``app/agents/``) owns the RAG-vs-LLM decision:
it judges whether the retrieved evidence can answer the *original* question and
self-heals by rewriting + retrying. There, the ``EvidencePolicy`` signals below —
including the FAISS ``faiss_strong`` (0.35) and ``faiss_moderate`` (0.30) thresholds
— are **advisory diagnostics** fed to the evidence evaluator; they can no longer
force or block a route on their own. ``route_query`` and the policy's own ``route``
verdict decide a route *only* on the legacy ``AGENTIC_RAG_ENABLED=false`` path
(unchanged, below). Nothing about FAISS scoring is removed — it is demoted from
final gate to one diagnostic signal.

Routing policy (evaluated per query; RAG if ANY branch fires, else LLM)
------------------------------------------------------------------------

* **Semantic** — the best FAISS relevance clears ``faiss_strong`` (0.35, the
  original threshold, unchanged). A strong vector match ⇒ RAG.

* **Lexical** — a *genuinely strong* keyword match, not just a high BM25 rank. A
  chunk BM25-ranked within ``bm25_strong_rank`` must actually contain the query's
  distinctive terms: at least ``lexical_min_terms`` of them AND at least
  ``lexical_min_ratio`` of all distinctive query terms. "Distinctive" = query
  tokens that survive a stopword list and the ``min_term_len`` length floor. This
  is what fixes the reported bug — a strong exact match now supports RAG even when
  the vector score sits just under 0.35 — while the informativeness guard stops a
  BM25 rank-1 match on generic/common words from triggering RAG on its own.

* **Agreement** — cross-retriever corroboration: the *same* chunk appears in the
  top ``agreement_rank`` of BOTH BM25 and FAISS, and the vector model still gives
  it at least ``faiss_moderate`` (0.30) relevance. Two independent "this is
  relevant" votes on one chunk => RAG, even if neither signal is strong alone. The
  moderate-relevance floor is the false-positive guard: FAISS always returns
  nearest neighbours, so agreement without near-relevant semantic support (e.g. a
  sports question hitting a "2018" chunk in a tax PDF) does NOT count.

Everything else (RAG chain, LLM fallback, graceful degradation when scores are
unavailable, SSE streaming, citations) is unchanged. Thresholds are configurable
via env (see :class:`EvidencePolicy`); the log for every decision spells out each
signal and the human-readable reason.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal

from app.retrievers.hybrid_retriever import (
    ChunkEvidence,
    HybridRetrievalResult,
    chunk_key,
    tokenize,
)
from app.utils.doc_utils import page_label, source_name, source_names
from app.utils.logger import log

RouteName = Literal["llm", "rag"]

# How many candidate rows to print per ranked list (FAISS / BM25 / RRF). Purely a
# logging cap; retrieval always fuses the full candidate set.
_MAX_LOGGED = 20

# Common English function/question words dropped when deciding which query terms
# are "distinctive" enough to count as lexical evidence. Deliberately small and
# dependency-free — it only needs to strip the words that make BM25 rank-1 matches
# on filler look like real evidence ("who won the world cup" should not match a tax
# PDF on "the"/"won"). Domain terms are never in here.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but", "if",
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
        "did", "done", "has", "have", "had", "what", "who", "whom", "whose",
        "which", "when", "where", "why", "how", "this", "that", "these", "those",
        "it", "its", "as", "at", "by", "with", "from", "into", "about", "i", "you",
        "he", "she", "we", "they", "me", "my", "your", "our", "their", "his",
        "her", "them", "us", "can", "could", "would", "should", "will", "shall",
        "may", "might", "must", "not", "no", "nor", "so", "than", "then", "there",
        "here", "such", "get", "got", "please", "tell", "give", "show", "explain",
        "s", "t",
    }
)


@dataclass(frozen=True)
class EvidencePolicy:
    """Configurable thresholds for the hybrid routing decision.

    Each knob has a plain-English meaning and an env override. The field defaults
    below are the documented defaults; :meth:`from_env` reads the environment on
    top of them, so ``route_query`` picks up env overrides at call time (and tests
    can build a policy directly). These are routing-decision parameters, so they
    live here with the router — the retrieval *structure* (candidate/final counts,
    RRF constant) stays in ``core/config.py``.
    """

    # Semantic: best FAISS relevance (0..1) at/above this => strong vector match.
    # HYBRID_RAG_SIMILARITY_THRESHOLD (unchanged name/default).
    faiss_strong: float = 0.35
    # Agreement: minimum FAISS relevance an agreeing chunk needs, so cross-retriever
    # agreement can only promote a genuine near-miss (just under the strong bar) and
    # never fires on a low-relevance nearest neighbour. Kept close to faiss_strong
    # on purpose — the *lexical* branch already covers strong-keyword / low-vector
    # matches, so agreement is the narrower "both retrievers nearly agree" promoter.
    faiss_moderate: float = 0.30
    # Lexical: only BM25 candidates ranked at/above this (rank <= N) count as
    # strong-lexical evidence. Rank 1 is strongest; small N keeps it to top hits.
    bm25_strong_rank: int = 3
    # Agreement: a chunk must sit in the top-N of BOTH retrievers to corroborate.
    agreement_rank: int = 10
    # Lexical: fraction of the query's distinctive terms that must appear in the
    # matched chunk for a *strong* lexical match (0.5 = at least half).
    lexical_min_ratio: float = 0.5
    # Lexical: absolute floor on distinctive terms present — at least this many must
    # overlap regardless of ratio, so a single incidental term never qualifies.
    lexical_min_terms: int = 1
    # Query tokens shorter than this are treated as non-distinctive (dropped
    # alongside stopwords) when measuring lexical overlap.
    min_term_len: int = 3

    @classmethod
    def from_env(cls) -> "EvidencePolicy":
        """Build a policy from env vars, falling back to the field defaults.

        Read at call time, so a deployment can retune routing without a code
        change. ``HYBRID_RAG_SIMILARITY_THRESHOLD`` keeps its original name.
        """
        d = cls()

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            try:
                return float(raw) if raw is not None else default
            except ValueError:
                return default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            try:
                return int(raw) if raw is not None else default
            except ValueError:
                return default

        return cls(
            faiss_strong=_float("HYBRID_RAG_SIMILARITY_THRESHOLD", d.faiss_strong),
            faiss_moderate=_float("HYBRID_RAG_FAISS_MODERATE_FLOOR", d.faiss_moderate),
            bm25_strong_rank=_int("HYBRID_RAG_BM25_STRONG_RANK", d.bm25_strong_rank),
            agreement_rank=_int("HYBRID_RAG_AGREEMENT_RANK", d.agreement_rank),
            lexical_min_ratio=_float("HYBRID_RAG_LEXICAL_MIN_RATIO", d.lexical_min_ratio),
            lexical_min_terms=_int("HYBRID_RAG_LEXICAL_MIN_TERMS", d.lexical_min_terms),
            min_term_len=_int("HYBRID_RAG_MIN_TERM_LEN", d.min_term_len),
        )


@dataclass(frozen=True)
class EvidenceAssessment:
    """The routing verdict plus every signal that fed it (for the decision log)."""

    route: RouteName
    reason: str
    used_scores: bool
    has_candidates: bool
    final_k: int
    # Semantic
    faiss_best: float | None
    semantic_strong: bool
    # Lexical
    lexical_strong: bool
    lexical_query_terms: int
    lexical_best_overlap: int
    lexical_best_ratio: float
    lexical_chunk: ChunkEvidence | None
    # Agreement
    agreement_strong: bool
    agreement_chunk: ChunkEvidence | None
    # Count of final chunks corroborated by both retrievers (log-only colour).
    final_agreement_count: int = 0


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str
    docs: list[Any] = field(default_factory=list)  # RRF-fused chunks used for RAG
    best_score: float | None = None  # FAISS best relevance (the semantic signal)
    used_scores: bool = False
    threshold: float | None = None  # the semantic bar (faiss_strong), for logging
    retrieval_time_ms: float | None = None
    total_routing_time_ms: float | None = None
    retrieved_docs: list[Any] = field(default_factory=list)  # final fused chunks
    context_length: int = 0
    result: HybridRetrievalResult | None = None  # full hybrid result, for logging
    assessment: EvidenceAssessment | None = None  # the evidence breakdown


# -- Evidence evaluation (the routing mechanism) --------------------------------
def _content_terms(query: str, policy: EvidencePolicy) -> set[str]:
    """Distinctive query terms: tokens past the stopword list and length floor."""
    return {
        term
        for term in tokenize(query)
        if len(term) >= policy.min_term_len and term not in _STOPWORDS
    }


def _chunk_overlap(doc: Any, query_terms: set[str]) -> int:
    """How many distinct distinctive query terms actually appear in the chunk.

    Reads ``page_content`` to compare terms — that text is used only to count
    overlap and is never logged.
    """
    if not query_terms:
        return 0
    chunk_terms = set(tokenize(getattr(doc, "page_content", "") or ""))
    return len(query_terms & chunk_terms)


def assess_evidence(
    result: HybridRetrievalResult, question: str, policy: EvidencePolicy
) -> EvidenceAssessment:
    """Weigh semantic + lexical + agreement evidence into a RAG/LLM verdict.

    Pure decision logic over the already-retrieved facts; it neither re-runs a
    retriever nor mutates ``result``.

    Under the agentic layer (the default) this verdict is **advisory**: the
    evidence evaluator in ``app/agents/`` blends these signals with an LLM
    judgement of the actual chunk content and makes the real RAG/LLM decision. The
    ``route`` returned here is authoritative only on the legacy
    ``AGENTIC_RAG_ENABLED=false`` path.
    """
    faiss_best = result.faiss_best_score
    used_scores = result.used_scores
    evidence = result.evidence
    has_candidates = bool(result.fused)

    # -- Semantic: a strong FAISS relevance is sufficient on its own. -------------
    semantic_strong = bool(
        used_scores and faiss_best is not None and faiss_best >= policy.faiss_strong
    )

    # -- Lexical: strongest informative BM25 hit within the top ranks. ------------
    query_terms = _content_terms(question, policy)
    n_terms = len(query_terms)
    lexical_chunk: ChunkEvidence | None = None
    lexical_best_overlap = 0
    lexical_best_ratio = 0.0
    for ev in evidence.values():
        if ev.bm25_rank is None or ev.bm25_rank > policy.bm25_strong_rank:
            continue
        overlap = _chunk_overlap(ev.document, query_terms)
        ratio = (overlap / n_terms) if n_terms else 0.0
        # Prefer more overlapping terms; break ties on the higher coverage ratio.
        if (overlap, ratio) > (lexical_best_overlap, lexical_best_ratio):
            lexical_best_overlap, lexical_best_ratio, lexical_chunk = overlap, ratio, ev
    lexical_strong = bool(
        n_terms
        and lexical_chunk is not None
        and lexical_best_overlap >= policy.lexical_min_terms
        and lexical_best_ratio >= policy.lexical_min_ratio
    )

    # -- Agreement: same chunk near the top of both, with moderate vector support.
    agreement_chunk: ChunkEvidence | None = None
    for ev in evidence.values():
        if not ev.in_both:
            continue
        if ev.faiss_rank > policy.agreement_rank or ev.bm25_rank > policy.agreement_rank:
            continue
        if ev.faiss_relevance is None or ev.faiss_relevance < policy.faiss_moderate:
            continue
        # Keep the best-fused corroborating chunk (lowest RRF rank).
        if agreement_chunk is None or (ev.rrf_rank or 10**9) < (agreement_chunk.rrf_rank or 10**9):
            agreement_chunk = ev
    agreement_strong = agreement_chunk is not None

    final_agreement_count = sum(
        1
        for key in result.final_keys
        if key in evidence and evidence[key].in_both
    )

    route: RouteName = (
        "rag"
        if has_candidates and (semantic_strong or lexical_strong or agreement_strong)
        else "llm"
    )
    reason = _compose_reason(
        route=route,
        policy=policy,
        used_scores=used_scores,
        has_candidates=has_candidates,
        final_k=result.final_k,
        faiss_best=faiss_best,
        semantic_strong=semantic_strong,
        lexical_strong=lexical_strong,
        lexical_chunk=lexical_chunk,
        lexical_best_overlap=lexical_best_overlap,
        lexical_best_ratio=lexical_best_ratio,
        n_terms=n_terms,
        agreement_strong=agreement_strong,
        agreement_chunk=agreement_chunk,
    )

    return EvidenceAssessment(
        route=route,
        reason=reason,
        used_scores=used_scores,
        has_candidates=has_candidates,
        final_k=result.final_k,
        faiss_best=faiss_best,
        semantic_strong=semantic_strong,
        lexical_strong=lexical_strong,
        lexical_query_terms=n_terms,
        lexical_best_overlap=lexical_best_overlap,
        lexical_best_ratio=lexical_best_ratio,
        lexical_chunk=lexical_chunk,
        agreement_strong=agreement_strong,
        agreement_chunk=agreement_chunk,
        final_agreement_count=final_agreement_count,
    )


def _compose_reason(
    *,
    route: RouteName,
    policy: EvidencePolicy,
    used_scores: bool,
    has_candidates: bool,
    final_k: int,
    faiss_best: float | None,
    semantic_strong: bool,
    lexical_strong: bool,
    lexical_chunk: ChunkEvidence | None,
    lexical_best_overlap: int,
    lexical_best_ratio: float,
    n_terms: int,
    agreement_strong: bool,
    agreement_chunk: ChunkEvidence | None,
) -> str:
    """Human-readable explanation naming exactly which signal(s) drove the route."""
    best = "N/A" if faiss_best is None else f"{faiss_best:.4f}"

    if not has_candidates:
        return "Hybrid retrieval returned no candidates; nothing to answer from."

    if route == "rag":
        fired: list[str] = []
        if semantic_strong:
            fired.append(
                f"strong semantic match (FAISS best relevance {best} >= {policy.faiss_strong:.2f})"
            )
        if lexical_strong and lexical_chunk is not None:
            rel = lexical_chunk.faiss_relevance
            tail = (
                f", while its FAISS relevance {rel:.4f} was below the {policy.faiss_strong:.2f} semantic bar"
                if rel is not None and rel < policy.faiss_strong
                else ""
            )
            fired.append(
                f"strong lexical match (BM25 rank {lexical_chunk.bm25_rank}: "
                f"{lexical_best_overlap}/{n_terms} distinctive query terms present, "
                f"ratio {lexical_best_ratio:.2f} >= {policy.lexical_min_ratio:.2f}{tail})"
            )
        if agreement_strong and agreement_chunk is not None:
            fired.append(
                f"cross-retriever agreement (one chunk in the top-{policy.agreement_rank} of both "
                f"BM25 rank {agreement_chunk.bm25_rank} and FAISS rank {agreement_chunk.faiss_rank}, "
                f"vector relevance {agreement_chunk.faiss_relevance:.4f} >= {policy.faiss_moderate:.2f})"
            )
        return (
            "RAG - " + "; ".join(fired)
            + f". Answering from the top {final_k} RRF-fused chunks."
        )

    # route == "llm"
    parts: list[str] = []
    if not used_scores:
        parts.append(
            "FAISS relevance scores were unavailable (semantic and agreement signals could not be evaluated)"
        )
    else:
        parts.append(f"weak semantic match (FAISS best relevance {best} < {policy.faiss_strong:.2f})")
    parts.append(
        f"no strong lexical match (best informative chunk covered {lexical_best_overlap}/{n_terms} "
        f"distinctive query terms, need ratio >= {policy.lexical_min_ratio:.2f})"
    )
    parts.append("no corroborated cross-retriever agreement")
    return "LLM - " + "; ".join(parts) + ". Treated as unrelated to the uploaded documents."


# -- Logging --------------------------------------------------------------------
def _score_label(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


def _time_label(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


def _candidate_row(rank: int, doc: Any, score: float | None, *, score_fmt: str) -> str:
    """One content-free candidate line: rank | source | page [| score].

    Deliberately logs only source + page + rank + score — never ``page_content`` —
    so full document text never lands in the logs.
    """
    metadata = getattr(doc, "metadata", None) or {}
    page = page_label(metadata.get("page"))
    base = f"{rank:>2}. {source_name(doc)} | Page {page}"
    if score is None:
        return base
    return f"{base} | {score_fmt}: {score:.4f}" if score_fmt else base


def _agreement_suffix(result: HybridRetrievalResult, doc: Any) -> str:
    """Marks a row whose chunk both retrievers found (cross-retriever agreement)."""
    ev = result.evidence.get(chunk_key(doc))
    return "  [BM25+FAISS]" if ev is not None and ev.in_both else ""


def _log_faiss(result: HybridRetrievalResult) -> None:
    log.info(f"FAISS Vector Candidates (top {result.candidates_k}, by relevance)")
    if not result.faiss_ranked:
        log.list_item("None")
        return
    for rank, (doc, score) in enumerate(result.faiss_ranked[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, score, score_fmt="relevance"))


def _log_bm25(result: HybridRetrievalResult) -> None:
    log.info(f"BM25 Keyword Candidates (top {result.candidates_k}, by rank)")
    if not result.bm25_ranked:
        log.list_item("None")
        return
    for rank, doc in enumerate(result.bm25_ranked[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, None, score_fmt=""))


def _log_fused(result: HybridRetrievalResult) -> None:
    log.info("RRF Fused Ranking (both lists merged by rank; [BM25+FAISS] = found by both)")
    if not result.fused:
        log.list_item("None")
        return
    for rank, (doc, score) in enumerate(result.fused[:_MAX_LOGGED], start=1):
        log.list_item(_candidate_row(rank, doc, score, score_fmt="rrf") + _agreement_suffix(result, doc))


def _log_final(result: HybridRetrievalResult, documents: list[Any]) -> None:
    log.info("Final Context (top-k after fusion)")
    if not documents:
        log.list_item("None")
        return
    for rank, doc in enumerate(documents, start=1):
        log.list_item(_candidate_row(rank, doc, None, score_fmt="") + _agreement_suffix(result, doc))


def log_retrieval_tables(result: HybridRetrievalResult, documents: list[Any]) -> None:
    """Public: log the FAISS / BM25 / RRF / final-context candidate tables.

    Wraps the private ``_log_*`` table helpers (with dividers between them) so
    callers outside this module — namely the agentic orchestrator — can reuse the
    exact candidate-table diagnostics without importing privates. Content-free by
    construction (source + page + rank + score only, never ``page_content``) and
    purely diagnostic: it logs the retrieval evidence, never the route decision.
    """
    _log_faiss(result)
    log.divider()
    _log_bm25(result)
    log.divider()
    _log_fused(result)
    log.divider()
    _log_final(result, documents)


def _log_evidence(assessment: EvidenceAssessment, policy: EvidencePolicy) -> None:
    """The three routing signals and their strong/weak verdicts (the routing view)."""
    log.info("Hybrid Evidence (routing signals)")

    semantic = "STRONG" if assessment.semantic_strong else "weak"
    log.list_item(
        f"Semantic (FAISS): best relevance {_score_label(assessment.faiss_best)} | "
        f"bar {policy.faiss_strong:.2f} | {semantic}"
    )

    lexical = "STRONG" if assessment.lexical_strong else "weak"
    if assessment.lexical_chunk is not None:
        lex_detail = (
            f"best informative chunk BM25 rank {assessment.lexical_chunk.bm25_rank} | "
            f"{assessment.lexical_best_overlap}/{assessment.lexical_query_terms} distinctive terms "
            f"(ratio {assessment.lexical_best_ratio:.2f}, need {policy.lexical_min_ratio:.2f})"
        )
    else:
        lex_detail = (
            f"no BM25 chunk within rank {policy.bm25_strong_rank} "
            f"({assessment.lexical_query_terms} distinctive query terms)"
        )
    log.list_item(f"Lexical (BM25): {lex_detail} | {lexical}")

    agreement = "STRONG" if assessment.agreement_strong else "none"
    if assessment.agreement_chunk is not None:
        ac = assessment.agreement_chunk
        agr_detail = (
            f"chunk in top-{policy.agreement_rank} of both — BM25 rank {ac.bm25_rank}, "
            f"FAISS rank {ac.faiss_rank}, relevance {_score_label(ac.faiss_relevance)} "
            f"(floor {policy.faiss_moderate:.2f})"
        )
    else:
        agr_detail = "no chunk in the top ranks of both retrievers with moderate vector support"
    log.list_item(f"Agreement (RRF): {agr_detail} | {agreement}")
    log.list_item(
        f"Final chunks corroborated by both retrievers: "
        f"{assessment.final_agreement_count} of {assessment.final_k}"
    )


def _log_router_decision(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
    policy: EvidencePolicy,
) -> None:
    log.section("NEW USER QUERY")
    log.kv("Query", question)
    log.kv("Retriever Exists", "Yes" if retriever_available else "No")

    result = decision.result
    if retriever_available and result is not None:
        log.divider()
        _log_faiss(result)
        log.divider()
        _log_bm25(result)
        log.divider()
        _log_fused(result)
        log.divider()
        _log_final(result, decision.retrieved_docs)

        log.divider()
        log.info("Routing")
        log.kv("Candidates K (per retriever)", result.candidates_k)
        log.kv("Final Context K", result.final_k)
        log.kv("RRF K", result.rrf_k)
        log.kv("FAISS Best Relevance", _score_label(decision.best_score))
        log.kv("Semantic Bar (faiss_strong)", f"{policy.faiss_strong:.4f}")
        log.kv("Context Length", decision.context_length)
        log.kv("Retrieval Time", _time_label(decision.retrieval_time_ms))

        if decision.assessment is not None:
            log.divider()
            _log_evidence(decision.assessment, policy)

    log.divider()
    log.kv("Router Decision", "RAG" if decision.route == "rag" else "GENERAL LLM")
    log.kv("Reason", decision.reason)
    log.kv(
        "Source",
        ", ".join(source_names(decision.docs)) if decision.route == "rag" else "General AI Knowledge",
    )
    log.kv("Routing Time", _time_label(decision.total_routing_time_ms))


def _finish(
    question: str,
    retriever_available: bool,
    decision: RouteDecision,
    started_at: float,
    policy: EvidencePolicy,
) -> RouteDecision:
    final_decision = replace(
        decision, total_routing_time_ms=(perf_counter() - started_at) * 1000
    )
    _log_router_decision(question, retriever_available, final_decision, policy)
    return final_decision


def route_query(
    retriever: Any | None,
    question: str,
    *,
    policy: EvidencePolicy | None = None,
    similarity_threshold: float | None = None,
) -> RouteDecision:
    """Route by hybrid retrieval evidence and reuse the fused chunks for RAG.

    One hybrid retrieval runs (BM25 + FAISS + RRF); this function only reads the
    result. RAG is chosen when the semantic, lexical, or agreement branch of the
    :class:`EvidencePolicy` fires (see the module docstring), and the RRF-fused
    top-k chunks become the RAG context.

    ``policy`` overrides the env-configured thresholds wholesale;
    ``similarity_threshold`` is a backward-compatible shortcut that overrides only
    the semantic bar (``faiss_strong``).
    """
    started_at = perf_counter()

    if policy is None:
        policy = EvidencePolicy.from_env()
    if similarity_threshold is not None:
        policy = replace(policy, faiss_strong=similarity_threshold)

    if retriever is None:
        return _finish(
            question,
            False,
            RouteDecision(
                "llm",
                "No document retriever available.",
                threshold=policy.faiss_strong,
                retrieval_time_ms=0.0,
            ),
            started_at,
            policy,
        )

    try:
        result = retriever.retrieve(question)
        assessment = assess_evidence(result, question, policy)

        is_rag = assessment.route == "rag"
        return _finish(
            question,
            True,
            RouteDecision(
                route=assessment.route,
                reason=assessment.reason,
                docs=result.documents if is_rag else [],
                best_score=result.faiss_best_score,
                used_scores=result.used_scores,
                threshold=policy.faiss_strong,
                retrieval_time_ms=result.retrieval_time_ms,
                retrieved_docs=result.documents,
                context_length=result.context_length if is_rag else 0,
                result=result,
                assessment=assessment,
            ),
            started_at,
            policy,
        )

    except Exception as exc:
        return _finish(
            question,
            True,
            RouteDecision(
                "llm",
                f"Retrieval failed, so routing fell back to the normal LLM chain: {exc}",
                threshold=policy.faiss_strong,
                retrieval_time_ms=None,
            ),
            started_at,
            policy,
        )
