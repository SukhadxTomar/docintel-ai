# DocIntel-AI

A PDF chatbot that actually knows when it's talking about your documents and when it's just talking. Upload a PDF, ask it questions, and it'll dig through the document for grounded answers — but if you ask it something unrelated ("what's the capital of France?"), it won't awkwardly force-fit your PDF into the answer. It just switches to general chat mode.

That switching is the whole point of this project. Most RAG chatbots either always retrieve (even for small talk) or never know when retrieval would actually help. DocIntel-AI makes that decision per-question, automatically, using vector similarity scores.

## What it does

- Upload one or more PDFs and chat with them through a clean Streamlit interface
- Hybrid routing: every question is scored against your documents before deciding whether to answer from the PDF or from general knowledge
- Token-by-token streaming so responses appear live, the way ChatGPT-style interfaces do
- Source attribution: every answer tells you whether it came from your documents or general AI knowledge, and which file it pulled from
- Console logging throughout the pipeline, so if something goes wrong (or you're just curious), you can see exactly what the router decided and why

## How it works, in one paragraph

When you upload a PDF, it gets split into chunks, embedded, and stored in a FAISS vector index. When you ask a question, the router runs a similarity search against that index. If the best match clears a confidence threshold (0.35 by default), your question and the matching chunks get sent to the RAG chain, which answers using only that retrieved context. If nothing scores high enough — or you haven't uploaded a PDF at all — the question falls through to a general LLM chain that answers from the model's own knowledge. Either way, the answer streams back token by token.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Vector store | FAISS |
| Orchestration | LangChain (LCEL) |
| Retrieval strategy | Top-k similarity search (see note below) |

## Project structure

```
.
├── app.py                      # Streamlit UI: upload, sidebar, chat loop
├── chains/
│   ├── chat_chain.py           # HybridChatChain — wires router + both chains together
│   ├── router.py                # Decides RAG vs LLM using similarity scores
│   ├── rag_chain.py            # Answers grounded in retrieved PDF chunks
│   └── llm_chain.py            # Answers from general knowledge
├── retrievers/
│   └── retriever.py             # MMR retriever config (currently not used on the live path — see below)
├── models/
│   └── llm_model.py             # Loads the Gemini chat model
├── ingestion/
│   ├── pdf_loader.py            # Loads raw PDFs
│   ├── text_splitter.py         # Chunks documents (800 chars, 100 overlap)
│   └── vector_store.py          # Builds the FAISS index
└── utils/
    └── logger.py                 # Colorized console logging used everywhere
```

## Running it locally

1. Clone the repo and install dependencies (`pip install -r requirements.txt` — add one if you haven't yet, based on the imports above: `streamlit`, `langchain`, `langchain-google-genai`, `langchain-huggingface`, `faiss-cpu`, `python-dotenv`).
2. Create a `.env` file with your Gemini API key (`GOOGLE_API_KEY=...`).
3. Run `streamlit run app.py`.
4. Upload a PDF from the sidebar, hit "Process PDFs", and start chatting.

You can also chat without uploading anything — it just behaves as a general assistant until you give it documents to ground itself in.

## Configuration

A couple of environment variables let you tune the router without touching code:

- `HYBRID_RAG_TOP_K` — how many chunks to retrieve per query (default 4)
- `HYBRID_RAG_SIMILARITY_THRESHOLD` — the confidence cutoff for routing to RAG (default 0.35)

## Why this design

A lot of "production-grade" RAG demos skip the routing problem entirely — they assume every question is about the uploaded document. That breaks the moment a user asks something casual. DocIntel-AI treats routing as a first-class decision, logs the reasoning behind every choice (retrieved chunks, scores, threshold, which chain ran, how long it took), and falls back gracefully if the vector store isn't ready yet. The goal was something that feels less like a toy demo and more like a system you could actually hand to someone.

## Known limitations

Being upfront about where the current implementation falls short of the original design intent — these are the next things being worked on:

- **Embeddings aren't normalized yet.** The similarity threshold is currently tuned against FAISS's default L2-based relevance score, not true cosine similarity. It works, but the number isn't as principled as it should be — normalizing the embeddings and re-tuning the threshold is next.
- **MMR retrieval is configured but not wired into the live routing path.** The router currently calls plain top-k similarity search directly on the vector store rather than going through the MMR retriever, so results aren't diversity-reranked yet.
- **Citations show filename only, not page number**, even though page-level metadata is already tracked internally — this is a UI gap, not a data gap.
- **No index persistence.** Uploading a new PDF rebuilds the FAISS index from scratch, so earlier documents in the same session get replaced rather than accumulated, and everything resets on restart.
- **Logging is human-readable console output, not structured/JSON** — fine for local debugging, not yet suited for production observability.

None of these are architectural dead ends — the hybrid routing design itself holds up — they're the concrete next steps before this becomes a FastAPI + React service instead of a Streamlit demo.

## Status

Actively maintained as a portfolio project. Currently being migrated from a Streamlit UI to a FastAPI backend + React frontend, with the fixes above being addressed as part of that migration.

## Author

SUKHAD TOMAR
