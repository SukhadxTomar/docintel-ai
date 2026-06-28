# DocIntel-AI

A Streamlit-powered PDF chatbot application that converts uploaded PDF documents into an interactive retrieval-augmented generation (RAG) experience.

## Overview

`docintel-ai` lets users upload one or more PDF files, builds a semantic vector index of document content, and answers user questions by retrieving relevant passages and generating responses with a Google Gemini LLM.

## Key Features

- Upload multiple PDF files through a Streamlit interface
- Extract and chunk PDF text using LangChain
- Build a FAISS vector store for semantic retrieval
- Generate contextual answers from retrieved document passages
- Display citation sources and page references for transparency

## Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini (`langchain-google-genai`)
- FAISS vector search
- Hugging Face embeddings
- PDF parsing with `PyPDF`
- Environment variable configuration with `python-dotenv`

## Installation

1. Clone the repository:

```bash
git clone <repo-url>
cd docintel-ai
```

2. Create a Python virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

- Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows Command Prompt:

```cmd
.\.venv\Scripts\activate.bat
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with credentials required by the Google generative AI integration.

Example:

```dotenv
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

> The app reads environment variables via `python-dotenv`.

## Usage

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, upload one or more PDF files, and start interacting with the chatbot.

## Project Structure

- `app.py` â€” main Streamlit application and UI logic
- `requirements.txt` â€” Python dependencies
- `ingestion/` â€” PDF loading, document splitting, embeddings, and vector store creation
- `chains/` â€” chat chain assembly and context formatting
- `models/` â€” LLM loading and configuration
- `prompts/` â€” system prompt template for the chatbot
- `retrievers/` â€” retrieval wrapper around the vector store
- `data/` â€” local storage for uploaded PDFs and vector DB files

## Notes

- Uploaded PDF files are stored locally in `uploaded_pdfs/`.
- The app uses a `FAISS` vector index built from document chunks.
- The prompt engine is configured to rely on uploaded PDF content and avoid hallucinations.

## Recommended Improvements

- Add error handling for invalid PDF uploads
- Support additional LLM providers and embedding backends
- Add a session history export or transcript feature

## License

This project does not include a license file. Add one if you want to share or distribute the code.
## Hybrid RAG Workflow and Observability

The chatbot now runs as a hybrid assistant:

1. `app.py` handles Streamlit UI, PDF upload, chat history, and per-response source display.
2. Uploaded PDFs are loaded with `load_pdfs()`, split with `split_documents()`, embedded with HuggingFace embeddings, and indexed in FAISS.
3. `create_chat_chain()` supports both no-PDF general chat and PDF-backed RAG chat.
4. `router.py` performs one retrieval probe per question and decides whether to use RAG or the general LLM.
5. If RAG is selected, the already-retrieved chunks are passed into `rag_chain.py`; no second FAISS search is performed.
6. If retrieval confidence is below the threshold, retrieved chunks are ignored and the normal LLM chain answers from general knowledge.

### Router Behavior

Routing is based on retrieval quality, not keywords. If similarity scores are available, the router compares calibrated scores against `HYBRID_RAG_SIMILARITY_THRESHOLD`.

Default threshold:

```dotenv
HYBRID_RAG_SIMILARITY_THRESHOLD=0.55
```

If similarity scores are unavailable, the router uses a conservative scoreless confidence heuristic based on query/content overlap, chunk substance, and metadata. Retrieved documents alone are not enough to force RAG.

### Terminal Logs

Each query logs:

- user question
- router decision and reason
- retrieved chunk count
- similarity scores or `N/A`
- selected source PDFs
- prompt context length
- first 300 characters of the injected RAG context
- whether the RAG or LLM chain streamed the response
- response timing

These logs are terminal-only and are not sent to the Streamlit UI.

### Source Attribution

Every assistant response displays its source in the Streamlit UI:

- RAG responses show the actual PDF filename(s) from document metadata.
- General LLM responses show `General AI Knowledge`.

### Validation Notes

A real uploaded resume PDF was processed through `load_pdfs()`, `split_documents()`, HuggingFace embeddings, FAISS, `create_retriever()`, and `create_chat_chain()`.

Observed routing distribution:

- PDF question: scores `0.5784`, `0.5719`, `0.5678`, `0.5385`; RAG selected with three chunks above `0.55`.
- General question: scores `0.4823`, `0.4704`, `0.4642`, `0.4543`; LLM selected with zero RAG chunks.

This supports `0.55` as the default threshold because it separates relevant document hits from unrelated top-k retrieval noise on the real FAISS pipeline.

