"""Agentic orchestration: the controlled loop that decides RAG vs. LLM.

This module is the layer that **replaces the FAISS hard-threshold routing**. It
sits on top of the *unchanged* hybrid retriever (BM25 + FAISS + RRF) and drives a
bounded, self-healing loop:

    classify → (general? → LLM)
             → retrieve → assess signals → judge evidence
                 → sufficient? → RAG (supporting chunks first)
                 → else rewrite the query and retry (bounded)
             → exhausted → controlled, grounded fallback

Key properties (see the approved plan and the task spec):

* **The retriever is consumed through one clean seam** — :meth:`search_documents`
  calls ``retriever.retrieve(query)`` and nothing else. The agent never touches
  FAISS/BM25/RRF internals, so hybrid retrieval is preserved exactly.
* **``original_query`` is immutable.** The first attempt retrieves with the user's
  exact question; only rewrites change ``current_query``. Evidence is always judged
  against the *original* question, and the downstream RAG chain answers the
  original question (``chat_chain`` passes it), so a rewritten retrieval never
  changes what gets answered.
* **FAISS is a signal, not a gate.** The FAISS score (and the whole
  ``EvidencePolicy``) feeds the evidence evaluator as one advisory input; the
  RAG/LLM decision comes from the evaluator's content-aware judgement blended with
  those signals — never from a single similarity threshold.
* **One retrieval per attempt** feeds signals + evaluator + answer context (no
  duplicate retrieval), and the loop is hard-capped by
  ``AGENTIC_MAX_RETRIEVAL_ATTEMPTS`` (never infinite).
* **Returns a populated ``RouteDecision``**, the exact type the legacy router
  returns, so SSE streaming, sources/citations, sessions, and the frontend are all
  unchanged. Latency is the only visible difference (agent work finishes before the
  first token).
* **Controlled fallback, no hallucination.** If evidence stays insufficient after
  every attempt but chunks exist, the route is RAG over the best chunks so the
  existing strict RAG prompt reports "I could not find this information in the
  uploaded documents" rather than inventing one; only a truly empty retrieval falls
  through to the general LLM.

Logging is concise ``[AGENT]`` decision lines plus the router's content-free
candidate tables. It never logs page content, secrets, or chain-of-thought.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from app.agents.evidence_evaluator import evaluate_evidence
from app.agents.query_classifier import classify_query
from app.agents.query_rewriter import rewrite_query
from app.agents.state import AgentState
from app.chains.router import (
    EvidencePolicy,
    RouteDecision,
    assess_evidence,
    log_retrieval_tables,
)
from app.core.config import settings
from app.utils.logger import log


def _fmt_score(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.4f}"


class AgenticOrchestrator:
    """Runs the self-healing retrieval loop and returns a ``RouteDecision``.

    Build once per retriever (cheap; holds no per-query state). ``run`` is
    stateless across calls — all per-query state lives in a fresh
    :class:`AgentState` — so one instance can serve concurrent requests.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        agent_llm: Any | None = None,
        max_attempts: int | None = None,
        evidence_threshold: float | None = None,
    ) -> None:
        self.retriever = retriever
        # May be None: the sub-components lazily load the cached agent LLM and each
        # degrades gracefully if it is unavailable, so construction never needs a key.
        self.agent_llm = agent_llm
        self.max_attempts = max(
            1,
            int(
                max_attempts
                if max_attempts is not None
                else settings.agentic_max_retrieval_attempts
            ),
        )
        self.evidence_threshold = float(
            evidence_threshold
            if evidence_threshold is not None
            else settings.agentic_evidence_threshold
        )

    # -- Retrieval seam ----------------------------------------------------------
    def search_documents(self, query: str) -> Any:
        """The agent's only retrieval entry point: one hybrid retrieval for ``query``.

        A thin wrapper over ``retriever.retrieve`` so the agent consumes the hybrid
        retriever as a black box (BM25 + FAISS + RRF unchanged) and never
        manipulates its internals. Returns a ``HybridRetrievalResult``.
        """
        return self.retriever.retrieve(query)

    def _has_documents(self) -> bool:
        num = getattr(self.retriever, "num_chunks", None)
        if isinstance(num, int):
            return num > 0
        return True  # unknown → assume documents exist (bias to retrieval)

    # -- Main loop ---------------------------------------------------------------
    def run(self, question: str, chat_history: str = "") -> RouteDecision:
        """Classify, retrieve, evaluate, and self-heal into a ``RouteDecision``.

        ``chat_history`` is accepted for interface parity with ``route_query``'s
        caller; retrieval and evidence judgement operate on the query itself.
        """
        started_at = perf_counter()
        state = AgentState(original_query=question, current_query=question)
        policy = EvidencePolicy.from_env()

        has_documents = self._has_documents()

        log.section("AGENTIC ORCHESTRATION")
        log.kv("Original Query", question)

        # -- Query understanding: does this even need the documents? -------------
        query_type = classify_query(question, has_documents=has_documents, llm=self.agent_llm)
        log.info(f"[AGENT] Query type: {query_type} (documents indexed: {has_documents})")

        if query_type == "general":
            log.info("[AGENT] Final route: GENERAL LLM (no retrieval needed)")
            return self._llm_decision(
                state,
                reason="Query understood as general (not document-specific); "
                "answered by the general LLM without retrieval.",
                started_at=started_at,
            )

        # -- Self-healing retrieval loop (bounded) -------------------------------
        last_result: Any | None = None
        last_signals: Any | None = None
        last_evaluation: Any | None = None

        for attempt in range(1, self.max_attempts + 1):
            state.attempt_number = attempt
            log.divider()
            log.info(f"[AGENT] Retrieval attempt {attempt}/{self.max_attempts}")
            log.kv("Search Query", state.current_query)

            try:
                result = self.search_documents(state.current_query)
            except Exception as exc:  # noqa: BLE001 — degrade to fallback, never crash the turn
                log.error(f"[AGENT] Retrieval failed on attempt {attempt}: {exc}")
                break

            state.retrieval_results.append(result)
            last_result = result

            # Content-free candidate diagnostics, reused verbatim from the router.
            log_retrieval_tables(result, result.documents)

            # Signals are computed against the ORIGINAL query and are advisory only —
            # they inform the evaluator; they do not decide the route.
            signals = assess_evidence(result, state.original_query, policy)
            last_signals = signals
            self._log_signals(signals)

            evaluation = evaluate_evidence(
                state.original_query,
                result,
                signals,
                llm=self.agent_llm,
                threshold=self.evidence_threshold,
            )
            state.evidence_assessment = evaluation
            last_evaluation = evaluation
            self._log_evaluation(evaluation)

            if evaluation.sufficient:
                docs = self._prioritize(result.documents, evaluation.supporting_chunk_indices)
                state.final_documents = docs
                state.route = "rag"
                log.info(
                    f"[AGENT] Final route: RAG (attempt {attempt}, "
                    f"confidence {evaluation.confidence:.2f})"
                )
                return self._rag_decision(
                    state,
                    result,
                    signals,
                    reason=(
                        f"Sufficient evidence on attempt {attempt} "
                        f"(confidence {evaluation.confidence:.2f}, {evaluation.source}): "
                        f"{evaluation.reason}"
                    ),
                    started_at=started_at,
                )

            # Insufficient → rewrite and retry, unless this was the last attempt.
            if attempt < self.max_attempts:
                new_query = rewrite_query(state, evaluation, llm=self.agent_llm)
                if not new_query:
                    log.info(
                        "[AGENT] No new query to try (rewrite empty/duplicate); stopping retries."
                    )
                    break
                state.rewritten_queries.append(new_query)
                state.current_query = new_query
                log.info(
                    f"[AGENT] Rewriting query to improve retrieval "
                    f"(missing: {evaluation.missing_information or 'n/a'})"
                )
                log.kv("Next Query", new_query)

        # -- Exhausted / stopped: controlled, grounded fallback ------------------
        return self._fallback_decision(
            state, last_result, last_signals, last_evaluation, started_at=started_at
        )

    # -- Supporting-chunk ordering ----------------------------------------------
    @staticmethod
    def _prioritize(documents: list[Any], supporting_indices: list[int]) -> list[Any]:
        """Put the evaluator's supporting chunks first, keeping the full top-k set.

        Preserves RRF order within each group. All final chunks are retained (the
        answer context is still the top ``final_context_k``); only their order
        changes so the strongest evidence leads.
        """
        if not supporting_indices:
            return list(documents)
        support = [i for i in supporting_indices if 0 <= i < len(documents)]
        if not support:
            return list(documents)
        chosen = set(support)
        return [documents[i] for i in support] + [
            doc for i, doc in enumerate(documents) if i not in chosen
        ]

    # -- RouteDecision builders --------------------------------------------------
    def _rag_decision(
        self,
        state: AgentState,
        result: Any,
        signals: Any,
        *,
        reason: str,
        started_at: float,
    ) -> RouteDecision:
        return RouteDecision(
            route="rag",
            reason=reason,
            docs=state.final_documents,
            best_score=result.faiss_best_score,
            used_scores=result.used_scores,
            # In agentic mode this is the confidence guideline, not a FAISS gate.
            threshold=self.evidence_threshold,
            retrieval_time_ms=result.retrieval_time_ms,
            total_routing_time_ms=(perf_counter() - started_at) * 1000,
            retrieved_docs=result.documents,
            context_length=result.context_length,
            result=result,
            assessment=signals,
        )

    def _llm_decision(
        self,
        state: AgentState,
        *,
        reason: str,
        started_at: float,
        result: Any | None = None,
    ) -> RouteDecision:
        return RouteDecision(
            route="llm",
            reason=reason,
            docs=[],
            best_score=getattr(result, "faiss_best_score", None),
            used_scores=getattr(result, "used_scores", False),
            threshold=self.evidence_threshold,
            retrieval_time_ms=getattr(result, "retrieval_time_ms", 0.0),
            total_routing_time_ms=(perf_counter() - started_at) * 1000,
            retrieved_docs=getattr(result, "documents", []) or [],
            context_length=0,
            result=result,
            assessment=None,
        )

    def _fallback_decision(
        self,
        state: AgentState,
        result: Any | None,
        signals: Any | None,
        evaluation: Any | None,
        *,
        started_at: float,
    ) -> RouteDecision:
        """Insufficient after all attempts: prefer a grounded RAG answer over the LLM.

        With chunks in hand we route RAG so the strict RAG prompt answers only from
        context (and otherwise says it could not find the information) — no
        hallucination. Only a genuinely empty retrieval falls through to the LLM.
        """
        if result is not None and getattr(result, "documents", None):
            supporting = evaluation.supporting_chunk_indices if evaluation is not None else []
            state.final_documents = self._prioritize(result.documents, supporting)
            state.route = "rag"
            log.divider()
            log.info("[AGENT] Final route: RAG (controlled fallback — grounded, no hallucination)")
            return self._rag_decision(
                state,
                result,
                signals,
                reason=(
                    f"Evidence judged insufficient after {state.attempt_number} attempt(s); "
                    "answering strictly from the best retrieved chunks so the assistant reports "
                    "missing information rather than inventing an answer."
                ),
                started_at=started_at,
            )

        state.route = "llm"
        log.divider()
        log.info("[AGENT] Final route: GENERAL LLM (no document evidence retrieved)")
        return self._llm_decision(
            state,
            reason=(
                "No usable document evidence was retrieved; answered by the general LLM."
                if result is not None
                else "Retrieval was unavailable; answered by the general LLM."
            ),
            started_at=started_at,
            result=result,
        )

    # -- Concise logging ---------------------------------------------------------
    @staticmethod
    def _log_signals(signals: Any) -> None:
        log.info(
            f"[AGENT] Signals (advisory): faiss_best={_fmt_score(getattr(signals, 'faiss_best', None))} "
            f"semantic={'Y' if getattr(signals, 'semantic_strong', False) else 'n'} "
            f"lexical={'Y' if getattr(signals, 'lexical_strong', False) else 'n'} "
            f"agreement={'Y' if getattr(signals, 'agreement_strong', False) else 'n'}"
        )

    @staticmethod
    def _log_evaluation(evaluation: Any) -> None:
        log.info(
            f"[AGENT] Evidence: sufficient={evaluation.sufficient} "
            f"confidence={evaluation.confidence:.2f} source={evaluation.source} :: {evaluation.reason}"
        )
