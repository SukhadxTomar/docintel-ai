# DocIntel-AI

A PDF chatbot that knows when your question is about the document and when it isn't.

Upload one or more PDFs, ask questions, and DocIntel-AI searches through the documents to give you grounded answers. If you ask something unrelated, like "What's the capital of France?", it won't try to force the PDF into the answer. It simply switches to general chat.

The main idea behind the project is **intelligent retrieval and routing**. Instead of retrieving from the documents for every question, DocIntel-AI decides whether retrieval is actually useful for that question.

It uses a combination of vector similarity, keyword matching, and agreement between the two retrieval methods to make that decision.

## What it does

* Upload one or more PDFs and chat with them through a React interface
* Uses hybrid retrieval with **BM25** for keyword matching and **FAISS** for semantic search
* Combines both retrieval rankings using **Reciprocal Rank Fusion (RRF)**
* Decides whether a question should use document context or general LLM knowledge
* Uses an agentic retrieval loop that can evaluate the retrieved evidence, rewrite the query, and retry when the first retrieval isn't enough
* Streams responses token by token so answers appear as they are generated
* Shows where an answer came from, including the PDF name and page for document-based answers
* Includes detailed console logging for BM25, FAISS, RRF, routing decisions, and retrieval attempts

## How it works

When a PDF is uploaded, the document is split into smaller chunks. These chunks are embedded and stored in a FAISS index. The same chunks are also indexed using BM25 for keyword-based retrieval.

When a question comes in, both retrievers search for relevant chunks. Their rankings are then combined using Reciprocal Rank Fusion, giving us a single list of candidate chunks.

The system then decides whether the question actually needs the documents.

For document-related questions, an agentic orchestration layer looks at the retrieved evidence and asks whether the original question can actually be answered from those chunks. The semantic relevance, keyword matches, and agreement between BM25 and FAISS are used as supporting signals rather than relying on one fixed similarity threshold.

If the evidence is good enough, the question and retrieved chunks are passed to the RAG chain.

If the evidence isn't sufficient, the system can rewrite the query and try retrieval again. This is useful for cases where the document contains a term such as `deductions`, while the user's question uses `deduction`. The retry process is bounded so it cannot keep running indefinitely.

If useful evidence still cannot be found, the system answers using the available document context and clearly states when the information could not be found rather than making something up.

If the question has nothing to do with the uploaded documents, it goes directly to the general LLM chain.

The final response is streamed back to the frontend token by token.

## Tech stack

| Layer         | Tool                                   |
| ------------- | -------------------------------------- |
| Frontend      | React + Vite + TypeScript              |
| Backend / API | FastAPI with SSE streaming             |
| LLM           | Any model available through OpenRouter |
| Embeddings    | HuggingFace `BAAI/bge-small-en-v1.5`   |
| Vector store  | FAISS                                  |
| Orchestration | LangChain / LCEL                       |
| Retrieval     | BM25 + FAISS + Reciprocal Rank Fusion  |

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   ├── schemas.py
│   │   │   └── deps.py
│   │   ├── session/
│   │   │   └── manager.py
│   │   ├── chains/
│   │   ├── agents/
│   │   ├── models/
│   │   │   └── llm_model.py
│   │   ├── ingestion/
│   │   ├── retrievers/
│   │   ├── prompts/
│   │   │   └── chatbot_prompt.py
│   │   ├── core/
│   │   │   └── config.py
│   │   └── utils/
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
└── frontend/
    ├── index.html
    ├── src/
    │   ├── App.tsx
    │   ├── components/
    │   ├── hooks/
    │   ├── api/
    │   └── styles/
    └── package.json
```

## Running locally

The project runs as two separate processes:

1. A FastAPI backend that handles the RAG pipeline and LLM
2. A React frontend that provides the chat interface

### Prerequisites

* Python 3.10+
* Node.js 18+

### 1. Start the backend

```bash
cd backend

python -m venv venv
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

On Windows:

```bash
copy .env.example .env
```

Open `backend/.env` and add your OpenRouter credentials:

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Then start the API:

```bash
uvicorn app.main:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/api/health
```

On the first PDF upload, the embedding model `BAAI/bge-small-en-v1.5` will be downloaded locally. The first upload may therefore take a little longer.

### 2. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173
```

By default, it connects to the backend at:

```text
http://localhost:8000
```

To use a different backend URL, set `VITE_API_BASE` in `frontend/.env`.

### 3. Start chatting

Upload a PDF from the sidebar and wait for it to finish processing.

Once it is ready, you can start asking questions about the document.

You can also use the chatbot without uploading a PDF. In that case, it behaves like a normal general-purpose assistant.

## Light and dark mode

The UI supports both light and dark themes.

Use the sun/moon toggle in the top-right corner of the chat header to switch between them. The selected theme is saved in the browser, and on the first visit the app follows your system's light or dark preference.

## Configuration

Most settings can be changed through `backend/.env`, so you don't need to modify the code for normal configuration changes.

### LLM

The only required credential is an OpenRouter API key.

```env
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Example models include:

```env
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_MODEL=google/gemini-2.5-flash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

### Retrieval

```env
RETRIEVAL_CANDIDATES_K=20
FINAL_CONTEXT_K=4
RRF_K=60
```

These control how many candidates are retrieved from BM25 and FAISS, how many fused chunks are finally used as context, and the RRF damping constant.

### Agentic RAG

```env
AGENTIC_RAG_ENABLED=true
AGENTIC_MAX_RETRIEVAL_ATTEMPTS=3
AGENTIC_EVIDENCE_THRESHOLD=0.70
```

The agentic layer evaluates whether the retrieved chunks are actually useful for answering the original question. If they aren't, it can rewrite the query and retry retrieval.

The retry count is capped to prevent the process from running indefinitely.

### Routing signals

The following settings control the hybrid retrieval signals:

```env
HYBRID_RAG_SIMILARITY_THRESHOLD=0.35
HYBRID_RAG_BM25_STRONG_RANK=3
HYBRID_RAG_LEXICAL_MIN_TERMS=1
HYBRID_RAG_LEXICAL_MIN_RATIO=0.5
HYBRID_RAG_MIN_TERM_LEN=3
HYBRID_RAG_AGREEMENT_RANK=10
HYBRID_RAG_FAISS_MODERATE_FLOOR=0.30
```

These signals look at:

* **Semantic relevance** from FAISS
* **Lexical relevance** from BM25
* **Retriever agreement** between BM25 and FAISS

When agentic RAG is enabled, these values act as supporting evidence for the evaluator rather than hard routing gates.

### Other settings

Additional configuration options use the `DOCINTEL_` prefix, including:

* `DOCINTEL_LLM_TEMPERATURE`
* `DOCINTEL_CHUNK_SIZE`
* `DOCINTEL_CHUNK_OVERLAP`
* `DOCINTEL_CORS_ALLOW_ORIGINS`

For the complete list, check:

```text
backend/app/core/config.py
```

## API

The backend exposes interactive Swagger documentation at:

```text
http://localhost:8000/docs
```

Main endpoints:

| Method | Endpoint                       | Purpose                 |
| ------ | ------------------------------ | ----------------------- |
| POST   | `/api/sessions`                | Create a chat session   |
| GET    | `/api/sessions/{id}`           | Get session status      |
| GET    | `/api/sessions/{id}/messages`  | Get message history     |
| POST   | `/api/sessions/{id}/documents` | Upload and process PDFs |
| POST   | `/api/sessions/{id}/chat`      | Stream a chat response  |
| POST   | `/api/sessions/{id}/clear`     | Clear the conversation  |
| DELETE | `/api/sessions/{id}`           | Delete a session        |

Chat responses are streamed using Server-Sent Events with events such as:

```text
token
sources
error
done
```

## Why I built it this way

A lot of RAG demos assume that every question is about the uploaded document.

That works for a simple demo, but it becomes awkward as soon as someone asks a normal question that has nothing to do with the PDF. The system still retrieves chunks and tries to use them, even when there is no reason to.

DocIntel-AI treats routing as an important part of the RAG pipeline.

For every question, it looks at the query, retrieval results, and evidence before deciding what should answer it. The system also logs what happened during retrieval and why a particular route was chosen.

The goal is to make the project feel closer to something that could actually be used, rather than just another basic PDF chatbot.

## Known limitations

There are still a few areas that need improvement.

### Embeddings are not normalized

FAISS currently uses L2-based relevance rather than true cosine similarity. This was more important when a single similarity threshold controlled routing.

Now that semantic similarity is only one of several signals used by the agentic evaluator, the impact is smaller. Normalizing the embeddings would still make the semantic signal cleaner.

### Some routing signals are hand-tuned

The routing system uses several manually chosen defaults for semantic relevance, keyword matching, and retriever agreement.

In agentic mode, these values are advisory rather than hard gates, but they are still manually selected rather than learned from data.

### Agentic retrieval adds latency and cost

A document question can require multiple LLM calls before the final response starts streaming.

Most questions are resolved on the first retrieval attempt, but more difficult questions may trigger evaluation and query rewriting.

The quality of the evidence evaluation also depends on the model selected through OpenRouter.

### BM25 is not persisted

The FAISS index is saved to disk for each session, but the BM25 index is rebuilt in memory when the chain loads.

This keeps the indexes synchronized, but rebuilding can become more noticeable with larger document collections.

### No MMR re-ranking

RRF combines the rankings from BM25 and FAISS, but it doesn't currently remove very similar chunks.

As a result, the final context can sometimes contain redundant information.

### New uploads replace previous documents

Uploading another PDF rebuilds the session index using the new files and resets the conversation.

Documents cannot currently be added incrementally to an existing collection.

### Sessions are stored in memory

The FAISS index is persisted to disk, but the session registry itself is stored in memory.

If the backend restarts, the saved index can become disconnected from the previous session and the frontend starts a new session.

## What's next

These limitations are mostly implementation improvements rather than major architectural problems.

The core hybrid retrieval and agentic routing design is already in place. The next steps are mainly around improving retrieval quality, persistence, context diversity, and scalability.

## Status

Actively maintained as a portfolio project.

The original Streamlit version has been migrated to a FastAPI backend with a React frontend. The current focus is on improving the retrieval pipeline and addressing the limitations listed above.

## Author

**SUKHAD TOMAR**
