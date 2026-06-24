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

- `app.py` — main Streamlit application and UI logic
- `requirements.txt` — Python dependencies
- `ingestion/` — PDF loading, document splitting, embeddings, and vector store creation
- `chains/` — chat chain assembly and context formatting
- `models/` — LLM loading and configuration
- `prompts/` — system prompt template for the chatbot
- `retrievers/` — retrieval wrapper around the vector store
- `data/` — local storage for uploaded PDFs and vector DB files

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
