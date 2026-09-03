# DocIntel-AI

A PDF chatbot that actually knows when it's talking about your documents and when it's just talking. Upload a PDF, ask it questions, and it'll dig through the document for grounded answers — but if you ask it something unrelated ("what's the capital of France?"), it won't awkwardly force-fit your PDF into the answer. It just switches to general chat mode.

That switching is the whole point of this project. Most RAG chatbots either always retrieve (even for small talk) or never know when retrieval would actually help. DocIntel-AI makes that decision per-question, automatically, using vector similarity scores.

## What it does

- Upload one or more PDFs and chat with them through a clean React interface
- Hybrid retrieval: each query is run through **both** a BM25 keyword search and a FAISS vector search, and the two rankings are fused with Reciprocal Rank Fusion (RRF) — so exact-term matches and semantic matches both count
- Score-based routing: every question is scored against your documents before deciding whether to answer from the PDF or from general knowledge
- Token-by-token streaming so responses appear live, the way ChatGPT-style interfaces do
- Source attribution: every answer tells you whether it came from your documents or general AI knowledge — and for document answers, the file and page it came from
- Console logging throughout the pipeline, so if something goes wrong (or you're just curious), you can see exactly what BM25, FAISS, and RRF produced and what the router decided and why

## How it works, in one paragraph

When you upload a PDF, it gets split into chunks, embedded, and stored in a FAISS vector index; the same chunks are also indexed for BM25 keyword search. When you ask a question, both retrievers fetch their top candidates and Reciprocal Rank Fusion merges the two rankings into a single best-of-both list — the top few fused chunks become the candidate context. Routing is a separate decision: if the best FAISS relevance score clears a confidence threshold (0.35 by default), your question and the fused chunks get sent to the RAG chain, which answers using only that retrieved context. If nothing scores high enough — or you haven't uploaded a PDF at all — the question falls through to a general LLM chain that answers from the model's own knowledge. Either way, the answer streams back token by token.

For a deeper look — the ingestion pipeline, routing logic, SSE streaming, and how the frontend and backend fit together — see **[ARCHITECTURE.md](./ARCHITECTURE.md)**.

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
│   │   ├── main.py                 # App + CORS + startup (creates storage dirs)
│   │   ├── api/routes/             # sessions, documents (upload), chat (SSE stream)
│   │   ├── chains/                 # chat_chain (router + both chains), router, rag_chain, llm_chain
│   │   ├── models/llm_model.py     # Loads the OpenRouter chat model (ChatOpenAI)
│   │   ├── ingestion/              # pdf_loader, text_splitter, embeddings, vector_store
│   │   ├── retrievers/             # hybrid_retriever (BM25 + FAISS + RRF), retriever (builders)
│   │   ├── core/config.py          # Env-overridable settings (OpenRouter key/model, chunking, retrieval k/RRF…)
│   │   └── utils/                  # colorized logging + doc helpers
│   ├── requirements.txt
│   ├── .env.example                # copy to .env, then add your OpenRouter key + model
│   └── .env                        # your local secrets (git-ignored)
└── frontend/                       # React + Vite (TypeScript)
    ├── index.html                  # applies the saved theme before first paint
    ├── src/
    │   ├── App.tsx                 # two-pane layout (sidebar + conversation)
    │   ├── components/             # Sidebar, ChatHeader (theme toggle), Composer, MessageList…
    │   ├── hooks/                  # useChat (session/streaming), useTheme (light/dark)
    │   ├── api/                    # typed REST client + SSE stream reader
    │   └── styles/tokens.css       # design tokens + light/dark themes
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

**Router / retrieval:**

- `RETRIEVAL_CANDIDATES_K` — candidates fetched from *each* retriever (BM25 and FAISS) before fusion (default 20)
- `FINAL_CONTEXT_K` — how many fused chunks become the RAG context (default 4)
- `RRF_K` — the Reciprocal Rank Fusion damping constant (default 60)
- `HYBRID_RAG_SIMILARITY_THRESHOLD` — the confidence cutoff (FAISS relevance) for routing to RAG (default 0.35)

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

Chat replies stream as Server-Sent Events (`token`, `sources`, `error`, `done`). The
event shapes and full request flow are documented in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Why this design

A lot of "production-grade" RAG demos skip the routing problem entirely — they assume every question is about the uploaded document. That breaks the moment a user asks something casual. DocIntel-AI treats routing as a first-class decision, logs the reasoning behind every choice (retrieved chunks, scores, threshold, which chain ran, how long it took), and falls back gracefully if the vector store isn't ready yet. The goal was something that feels less like a toy demo and more like a system you could actually hand to someone.

## Known limitations

Being upfront about where the current implementation falls short — these are the next things to improve:

- **Embeddings aren't normalized.** The routing threshold is tuned against FAISS's L2-based relevance score, not true cosine similarity. It works, but the number isn't as principled as it should be — normalizing the embeddings and re-tuning the threshold is next.
- **The RAG-vs-LLM routing decision still rests on the FAISS relevance score alone.** BM25 and RRF decide *which* chunks are the best context, but *whether* the question is document-related is a pure vector-space call. A strong keyword-only match that the vector model misses could therefore still route to the general LLM.
- **The BM25 index isn't persisted.** The FAISS index is saved to disk per session; the BM25 index is rebuilt in memory from the persisted chunks each time the chain loads. That's cheap and keeps the two indexes in sync, but it isn't free on very large document sets.
- **No diversity re-ranking (MMR).** RRF fuses keyword and vector rankings but doesn't de-duplicate near-identical chunks, so the final context can contain redundant passages.
- **Uploads replace, they don't accumulate.** Processing a new PDF rebuilds the session's index from the new files only and resets the conversation — there's no way to add to an existing document set yet.
- **Sessions are in-memory.** The FAISS index is persisted to disk per session, but the session *registry* is not, so restarting the backend orphans the saved index and the frontend falls back to a fresh session.

None of these are architectural dead ends — the hybrid routing design holds up. They're the concrete next steps now that the core has moved from a Streamlit demo to a FastAPI + React service.

## Status

Actively maintained as a portfolio project. The migration from the original Streamlit demo to a FastAPI backend + React frontend is complete; the fixes listed above are the ongoing work.

## Author

SUKHAD TOMAR
