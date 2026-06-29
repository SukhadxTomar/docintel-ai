# DocIntel-AI

An AI-powered PDF chatbot built with **Streamlit**, **LangChain**, **Google Gemini**, and **FAISS**. Upload one or more PDF files and chat with them naturally using Retrieval-Augmented Generation (RAG).

---

## Features

* 📄 Chat with one or multiple PDF documents
* 🤖 AI-powered responses using Google Gemini
* 🔍 Semantic search with FAISS and Hugging Face embeddings
* 🧠 Hybrid routing between PDF knowledge and general AI knowledge
* 📚 Source attribution for every response
* ⚡ Fast document retrieval and streaming responses
* 📝 Detailed terminal logs for debugging and observability

---

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini
* FAISS
* Hugging Face Embeddings
* PyPDF
* python-dotenv

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd docintel-ai
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
.\.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
LANGCHAIN_API_KEY=your_langchain_api_key

HYBRID_RAG_SIMILARITY_THRESHOLD=0.35
```

---

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, upload your PDFs, and start chatting.

---

## Project Structure

```
docintel-ai/
│── app.py
│── chains/
│── ingestion/
│── models/
│── prompts/
│── retrievers/
│── router/
│── data/
│── requirements.txt
```

---

## How It Works

1. Upload one or more PDF files.
2. Documents are split into smaller chunks.
3. Chunks are converted into embeddings and stored in a FAISS vector database.
4. Every question first passes through a router.
5. If the retrieved document similarity is **0.35 or higher**, the chatbot answers using the uploaded PDFs.
6. Otherwise, it responds using the general Gemini model.

This hybrid approach keeps document-related questions grounded while allowing the chatbot to answer general knowledge questions naturally.

---

## Terminal Observability

For every query, the application logs useful debugging information such as:

* User question
* Router decision
* Similarity score
* Threshold comparison
* Retrieved document count
* Source PDF names
* Context size
* Response time
* Selected response chain (RAG or LLM)

These logs are only visible in the terminal and are not shown in the Streamlit interface.

---

## Source Attribution

Each response includes its origin:

* **Uploaded PDF(s)** for document-based answers
* **General AI Knowledge** for non-document questions

---

## Future Improvements

* Conversation history export
* Support for additional LLM providers
* OCR support for scanned PDFs
* Persistent vector database
* Authentication and user sessions

---

## License

This project currently does not include a license. Add one if you plan to distribute or open-source the project.
