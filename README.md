# DocIntel-AI

A PDF chatbot that actually knows when it's talking about your documents and when it's just talking. Upload a PDF, ask it questions, and it'll dig through the document for grounded answers — but if you ask it something unrelated ("what's the capital of France?"), it won't awkwardly force-fit your PDF into the answer. It just switches to general chat mode.

That switching is the whole point of this project. Most RAG chatbots either always retrieve (even for small talk) or never know when retrieval would actually help. DocIntel-AI makes that decision per-question, automatically, using vector similarity scores.

## What it does

- **Upload one or more PDFs** and chat with them through a clean Streamlit interface
- **Hybrid routing**: every question is scored against your documents before deciding whether to answer from the PDF or from general knowledge
- **Token-by-token streaming** so responses appear live, the way ChatGPT-style interfaces do
- **Source attribution**: every answer tells you whether it came from your documents or general AI knowledge, and which file/page it pulled from
- **Structured logging** throughout the pipeline, so if something goes wrong (or you're just curious), you can see exactly what the router decided and why

## How it works, in one paragraph

When you upload a PDF, it gets split into chunks, embedded, and stored in a FAISS vector index. When you ask a question, the router runs a similarity search against that index. If the best match clears a confidence threshold (0.50 by default), your question and the matching chunks get sent to the RAG chain, which answers using only that retrieved context. If nothing scores high enough — or you haven't uploaded a PDF at all — the question falls through to a general LLM chain that answers from the model's own knowledge. Either way, the answer streams back token by token.


## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Vector store | FAISS |
| Orchestration | LangChain (LCEL) |
| Retrieval strategy | MMR (Maximal Marginal Relevance) |

## Project structure

```
.
├── app.py                      # Streamlit UI: upload, sidebar, chat loop
├── chains/
│   ├── chat_chain.py           # HybridChatChain — wires router + both chains together
│   ├── rag_chain.py            # Answers grounded in retrieved PDF chunks
│   └── llm_chain.py            # Answers from general knowledge
├── routers/
│   └── router.py                # Decides RAG vs LLM using similarity scores
├── retrievers/
│   └── retriever.py             # Builds the FAISS-backed MMR retriever
├── models/
│   └── llm_model.py             # Loads the Gemini chat model
├── ingestion/
│   ├── pdf_loader.py            # Loads raw PDFs (not shown above, but referenced)
│   ├── text_splitter.py         # Chunks documents
│   └── vector_store.py          # Builds the FAISS index
└── utils/
    └── logger.py                 # Structured ANSI logging used everywhere
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
- `HYBRID_RAG_SIMILARITY_THRESHOLD` — the confidence cutoff for routing to RAG (default 0.35 in the router's own default, currently run at 0.50 in this build)

## Why this design

A lot of "production-grade" RAG demos skip the routing problem entirely — they assume every question is about the uploaded document. That breaks the moment a user asks something casual. DocIntel-AI treats routing as a first-class decision, logs the reasoning behind every choice (retrieved chunks, scores, threshold, which chain ran, how long it took), and falls back gracefully if retrieval fails or the vector store isn't ready yet. The goal was something that feels less like a toy demo and more like a system you could actually hand to someone.

## Status

Actively maintained as a portfolio project. Recent work has focused on observability (structured logging across the full pipeline), fixing a breaking change in `langchain-google-genai` 4.0.0 around streaming, and tightening the router's fallback behavior so a failed retrieval never crashes the chat.

# Author

SUKHAD TOMAR