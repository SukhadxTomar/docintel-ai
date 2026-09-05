# DocIntel-AI

A PDF chatbot that actually knows when it's talking about your documents and when it's just talking. Upload a PDF, ask it questions, and it'll dig through the document for grounded answers — but if you ask it something unrelated ("what's the capital of France?"), it won't awkwardly force-fit your PDF into the answer. It just switches to general chat mode.

That switching is the whole point of this project. Most RAG chatbots either always retrieve (even for small talk) or never know when retrieval would actually help. DocIntel-AI makes that decision per-question, automatically, by weighing several retrieval signals together — vector similarity, keyword-match strength, and whether both retrievers agree — rather than trusting a single score.

## What it does

- Upload one or more PDFs and chat with them through a clean React interface
- Hybrid retrieval: each query is run through **both** a BM25 keyword search and a FAISS vector search, and the two rankings are fused with Reciprocal Rank Fusion (RRF) — so exact-term matches and semantic matches both count
- Evidence-based routing: every question is weighed against your documents — semantic (vector) relevance, keyword-match strength, and cross-retriever agreement — before deciding whether to answer from the PDF or from general knowledge
- Agentic self-healing retrieval: the RAG-vs-LLM decision no longer rests on a fixed FAISS similarity threshold — an orchestration layer judges whether the retrieved chunks can actually answer your *original* question, and if not, rewrites the query and retries (bounded) before falling back, instead of guessing or hallucinating
- Token-by-token streaming so responses appear live, the way ChatGPT-style interfaces do
- Source attribution: every answer tells you whether it came from your documents or general AI knowledge — and for document answers, the file and page it came from
- Console logging throughout the pipeline, so if something goes wrong (or you're just curious), you can see exactly what BM25, FAISS, and RRF produced and what the router decided and why

## How it works, in one paragraph

When you upload a PDF, it gets split into chunks, embedded, and stored in a FAISS vector index; the same chunks are also indexed for BM25 keyword search. When you ask a question, both retrievers fetch their top candidates and Reciprocal Rank Fusion merges the two rankings into a single best-of-both list — the top few fused chunks become the candidate context. Routing is a separate decision, and it no longer rests on the vector score alone — in fact it no longer rests on any fixed similarity threshold. An **agentic orchestration layer** takes over: it first decides whether your question even needs the documents, then — for document questions — retrieves once and asks a focused question of the evidence, *"can the original question actually be answered from these chunks?"*, judging the real chunk content rather than trusting a score. The three hybrid signals — **semantic** (best FAISS relevance), **lexical** (do the top BM25 chunks actually contain the question's distinctive keywords?), and **agreement** (did both retrievers independently rank the same chunk near the top?) — are folded in as advisory evidence, not as gates. If the evidence is sufficient, your question and the supporting chunks go to the RAG chain, which answers using only that retrieved context. If it isn't, the agent **rewrites the query and retries** (bounded by a configurable cap) — this is what recovers exact-token misses like *deduction* vs. *deductions* — and if evidence is still missing after every attempt it answers strictly from the best chunks (reporting that it couldn't find the information rather than inventing it), or, when nothing relevant was retrieved at all, falls through to a general LLM chain. Either way, the answer streams back token by token.

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | React + Vite (TypeScript) |
| Backend / API | FastAPI (token streaming over SSE) |
| LLM | Any model on [OpenRouter](https://openrouter.ai/models) (via `langchain-openai`) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (runs locally, no API key) |
| Vector store | FAISS |
| Orchestration | LangChain (LCEL) |
| Retrieval strategy | Hybrid — BM25 (keyword) + FAISS (vector), fused with Reciprocal Rank Fusion |

## Project structure

```
.
├── backend/                        # FastAPI service
│   ├── app/
│   │   ├── main.py                 # App + CORS + startup lifespan (creates storage dirs) + /api/health
│   │   ├── api/
│   │   │   ├── routes/             # sessions, documents (upload), chat (SSE stream)
│   │   │   ├── schemas.py          # Pydantic request/response models
│   │   │   └── deps.py             # shared route deps (session lookup, 404 if missing)
│   │   ├── session/manager.py      # in-memory session registry: history, PDF metadata, per-session chain
│   │   ├── chains/                 # chat_chain (router + both chains), router, rag_chain, llm_chain
│   │   ├── agents/                 # Agentic RAG: orchestrator, query_classifier, evidence_evaluator, query_rewriter, state
│   │   ├── models/llm_model.py     # OpenRouter chat models: load_llm (streaming answers) + load_agent_llm (temp-0 agent)
│   │   ├── ingestion/              # pdf_loader, text_splitter, embeddings, vector_store
│   │   ├── retrievers/             # hybrid_retriever (BM25 + FAISS + RRF), retriever (builders)
│   │   ├── prompts/chatbot_prompt.py  # the strict RAG system prompt (history + context + question)
│   │   ├── core/config.py          # Env-overridable settings (OpenRouter, chunking, retrieval k/RRF, agentic knobs)
│   │   └── utils/                  # colorized logging + doc helpers
│   ├── requirements.txt
│   ├── .env.example                # copy to .env, then add your OpenRouter key + model
│   └── .env                        # your local secrets (git-ignored)
└── frontend/                       # React + Vite (TypeScript)
    ├── index.html                  # applies the saved theme before first paint
    ├── src/
    │   ├── App.tsx                 # two-pane layout (sidebar + conversation)
    │   ├── components/             # Sidebar, UploadPanel, StatusPanel, ChatHeader (theme toggle), Composer, MessageList, MessageItem, SourceBadge…
    │   ├── hooks/                  # useChat (session/streaming), useTheme (light/dark)
    │   ├── api/                    # typed REST client + SSE stream reader
    │   └── styles/                 # tokens.css (design tokens + light/dark) + global.css
    └── package.json
```

## Running it locally

The app is two processes: a **FastAPI backend** (the RAG pipeline + LLM) and a
**React frontend** (the chat UI). Run the backend first, then the frontend.

**Prerequisites:** Python 3.10+ and Node.js 18+.

### 1. Backend (FastAPI)

```bash
cd backend

# create + activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell/cmd)
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Then set up your OpenRouter credentials — this is the **only** key the app needs
(embeddings run locally, so there's nothing else to configure):

```bash
cp .env.example .env           # Windows: copy .env.example .env
```

Open `backend/.env` and fill in:

```env
OPENROUTER_API_KEY=sk-or-...            # from https://openrouter.ai/keys
OPENROUTER_MODEL=openai/gpt-4o-mini     # any id from https://openrouter.ai/models
```

Start the API (from the `backend/` folder):

```bash
uvicorn app.main:app --reload
```

It serves on **http://localhost:8000** (health check: http://localhost:8000/api/health).

> First run note: the local embedding model (`BAAI/bge-small-en-v1.5`) downloads
> automatically the first time you upload a PDF, so that upload takes a bit longer.

### 2. Frontend (React + Vite)

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The frontend calls the backend at
`http://localhost:8000` by default (CORS is already configured for it). To point
it elsewhere, set `VITE_API_BASE` in `frontend/.env` (see `frontend/.env.example`).

### 3. Use it

Upload a PDF from the sidebar, wait for it to process, and start chatting. You can
also chat without uploading anything — it behaves as a general assistant until you
give it documents to ground itself in.

## Appearance (light / dark mode)

The UI ships with both light and dark themes. Use the sun/moon toggle in the top-right
of the chat header to switch. Your choice is saved in the browser (`localStorage`), and
on first visit the app follows your operating system's light/dark preference.

## Configuration

Everything is configured through environment variables in `backend/.env` — no code
changes needed.

**LLM (OpenRouter)** — the only credentials the app needs:

- `OPENROUTER_API_KEY` — your OpenRouter API key (**required**)
- `OPENROUTER_MODEL` — the model id, e.g. `openai/gpt-4o-mini`, `google/gemini-2.5-flash`, `anthropic/claude-3.5-sonnet` (default `openai/gpt-4o-mini`)
- `OPENROUTER_BASE_URL` — override the OpenRouter endpoint (default `https://openrouter.ai/api/v1`)

**Retrieval (structure — how chunks are fetched and fused):**

- `RETRIEVAL_CANDIDATES_K` — candidates fetched from *each* retriever (BM25 and FAISS) before fusion (default 20)
- `FINAL_CONTEXT_K` — how many fused chunks become the RAG context (default 4)
- `RRF_K` — the Reciprocal Rank Fusion damping constant (default 60)

**Agentic RAG orchestration (the default RAG-vs-LLM decider):** a controlled layer on top of hybrid retrieval that understands the query, judges whether the retrieved evidence can actually answer your *original* question, and self-heals by rewriting + retrying before answering or falling back. All optional (defaults shown):

- `AGENTIC_RAG_ENABLED` — master switch; `true` runs the agentic loop, `false` reverts to the legacy one-pass router (default `true`)
- `AGENTIC_MAX_RETRIEVAL_ATTEMPTS` — hard cap on retrieval attempts per query, so the self-healing loop can never run forever (default 3, minimum 1)
- `AGENTIC_EVIDENCE_THRESHOLD` — a confidence **guideline** (0..1) the evidence evaluator blends with the hybrid signals; deliberately *not* a single hard score gate replacing the old FAISS threshold (default 0.70)

**Routing (evidence policy — RAG vs. general LLM):** RAG is chosen when the **semantic**, **lexical**, *or* **agreement** signal fires; otherwise the question goes to the general LLM. **With `AGENTIC_RAG_ENABLED=true` (the default) these values are advisory diagnostics fed to the evidence evaluator — they cannot decide the route on their own; they only route directly on the legacy `AGENTIC_RAG_ENABLED=false` path.** FAISS scoring itself is unchanged — it is demoted from a hard gate to one signal. All are optional (defaults shown):

- `HYBRID_RAG_SIMILARITY_THRESHOLD` — *semantic*: FAISS relevance cutoff for a strong vector match, sufficient on its own (default 0.35; **diagnostic-only in agentic mode**)
- `HYBRID_RAG_BM25_STRONG_RANK` — *lexical*: only BM25 hits ranked this high or better count as keyword evidence (default 3)
- `HYBRID_RAG_LEXICAL_MIN_TERMS` — *lexical*: minimum distinctive query terms the matched chunk must actually contain (default 1)
- `HYBRID_RAG_LEXICAL_MIN_RATIO` — *lexical*: minimum fraction of the query's distinctive terms present for a *strong* match (default 0.5)
- `HYBRID_RAG_MIN_TERM_LEN` — *lexical*: query tokens shorter than this are treated as non-distinctive, alongside stopwords (default 3)
- `HYBRID_RAG_AGREEMENT_RANK` — *agreement*: a chunk must sit in the top-N of *both* retrievers to corroborate (default 10)
- `HYBRID_RAG_FAISS_MODERATE_FLOOR` — *agreement*: minimum FAISS relevance an agreeing chunk needs — the false-positive guard against FAISS's always-returned nearest neighbours (default 0.30; **diagnostic-only in agentic mode**)

**Other tunables** (prefixed `DOCINTEL_`): `DOCINTEL_LLM_TEMPERATURE`,
`DOCINTEL_CHUNK_SIZE`, `DOCINTEL_CHUNK_OVERLAP`, `DOCINTEL_CORS_ALLOW_ORIGINS`.
See `backend/app/core/config.py` for the full list.

## API

The backend is a FastAPI service. With it running, interactive API docs (Swagger UI)
are at **http://localhost:8000/docs**. The main endpoints:

| Method & path | Purpose |
|---|---|
| `POST /api/sessions` | Create a chat session |
| `GET /api/sessions/{id}` | Session status |
| `GET /api/sessions/{id}/messages` | Message history |
| `POST /api/sessions/{id}/documents` | Upload + process PDFs (multipart) |
| `POST /api/sessions/{id}/chat` | Stream a chat turn (SSE) |
| `POST /api/sessions/{id}/clear` | Clear the conversation |
| `DELETE /api/sessions/{id}` | Delete the session |

Chat replies stream as Server-Sent Events (`token`, `sources`, `error`, `done`).

## Why this design

A lot of "production-grade" RAG demos skip the routing problem entirely — they assume every question is about the uploaded document. That breaks the moment a user asks something casual. DocIntel-AI treats routing as a first-class decision, logs the reasoning behind every choice (retrieved chunks, scores, threshold, which chain ran, how long it took), and falls back gracefully if the vector store isn't ready yet. The goal was something that feels less like a toy demo and more like a system you could actually hand to someone.

## Known limitations

Being upfront about where the current implementation falls short — these are the next things to improve:

- **Embeddings aren't normalized.** FAISS's relevance score is L2-based, not true cosine similarity. This mattered more when a single 0.35 threshold gated routing; now that score is just one advisory signal the evidence evaluator weighs, so it's lower-stakes — but normalizing the embeddings would still make the semantic signal more principled.
- **Routing signals are hand-tuned heuristics (now advisory).** The RAG-vs-LLM decision is made by the agentic layer — an LLM judges whether the retrieved chunks can actually answer your original question, blended with the three hybrid signals — so the old cut-offs (the 0.35 semantic bar, the lexical term/ratio floors, the agreement rank and its 0.30 floor) no longer gate the route; they're diagnostic inputs, and decide routing directly only on the legacy `AGENTIC_RAG_ENABLED=false` path. Those cut-offs are still hand-picked defaults, not learned.
- **The agentic loop adds latency and cost.** Before the first token, a document question makes up to one classification, *N* evidence-evaluation, and *N*−1 query-rewrite LLM calls (bounded by `AGENTIC_MAX_RETRIEVAL_ATTEMPTS`); most questions resolve on the first attempt (two extra calls). Evaluation quality also depends on the chosen OpenRouter model — there's a deterministic signal-based fallback when its JSON won't parse, but a weak model can still under- or over-judge sufficiency.
- **The BM25 index isn't persisted.** The FAISS index is saved to disk per session; the BM25 index is rebuilt in memory from the persisted chunks each time the chain loads. That's cheap and keeps the two indexes in sync, but it isn't free on very large document sets.
- **No diversity re-ranking (MMR).** RRF fuses keyword and vector rankings but doesn't de-duplicate near-identical chunks, so the final context can contain redundant passages.
- **Uploads replace, they don't accumulate.** Processing a new PDF rebuilds the session's index from the new files only and resets the conversation — there's no way to add to an existing document set yet.
- **Sessions are in-memory.** The FAISS index is persisted to disk per session, but the session *registry* is not, so restarting the backend orphans the saved index and the frontend falls back to a fresh session.

None of these are architectural dead ends — the hybrid routing design holds up. They're the concrete next steps now that the core has moved from a Streamlit demo to a FastAPI + React service.

## Status

Actively maintained as a portfolio project. The migration from the original Streamlit demo to a FastAPI backend + React frontend is complete; the fixes listed above are the ongoing work.

## Author

SUKHAD TOMAR
