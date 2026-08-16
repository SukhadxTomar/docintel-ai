"""Server-side chat sessions for the FastAPI backend.

This replaces Streamlit's ``st.session_state``: each browser/client owns a
``ChatSession`` (keyed by an opaque id) that holds its conversation history,
uploaded-PDF metadata, and a lazily-built chat chain. Uploaded PDFs and the
FAISS index are persisted per-session under ``settings.storage_dir`` so the
chain can be rebuilt from disk (the "load-persisted-index" path in
``retrievers/retriever.py``) rather than kept only in process memory.

The behavior of a turn mirrors the old ``app.py`` exactly — same 12-message
history window, same "reset conversation on a new document set" rule, same
source resolution — only the transport changed.
"""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator

from app.chains.chat_chain import HybridChatChain, create_chat_chain
from app.core.config import settings
from app.ingestion.pdf_loader import load_pdfs
from app.ingestion.text_splitter import split_documents
from app.ingestion.vector_store import load_vector_store, save_vector_store
from app.retrievers.retriever import retriever_from_vector_store
from app.utils.doc_utils import indexed_chunks, vector_store_from_retriever
from app.utils.logger import log

# Most-recent messages fed back to the model as conversation history. Mirrors
# the Streamlit app's HISTORY_WINDOW_MESSAGES so turn behavior is unchanged.
HISTORY_WINDOW_MESSAGES = 12

# Label shown when an answer came from the general LLM rather than the PDFs.
GENERAL_SOURCE_LABEL = "General AI Knowledge"


def _attach_original_names(
    documents: list[Any], saved_files: list[dict[str, str]]
) -> list[Any]:
    """Tag each loaded page with the user's original filename (for citations).

    PyPDFLoader records the on-disk (uuid) path in ``metadata['source']``; map
    it back to the display name so sources read as the uploaded file, not a
    random hex name. Same logic as the old ``app.attach_original_names``.
    """
    names_by_path = {
        str(Path(file["path"]).resolve()): file["original_name"]
        for file in saved_files
    }

    for document in documents:
        metadata = document.metadata or {}
        source = metadata.get("source")
        if source:
            original_name = names_by_path.get(str(Path(source).resolve()))
            if original_name:
                metadata["original_name"] = original_name
                document.metadata = metadata

    return documents


def _source_from_final(event: dict[str, Any]) -> dict[str, Any]:
    """Turn the chain's ``final`` stream event into a stored ``source`` record."""
    if event.get("mode") == "rag":
        return {"type": "rag", "sources": list(event.get("sources") or [])}
    return {"type": "llm", "label": GENERAL_SOURCE_LABEL}


@dataclass
class ChatSession:
    """One client's conversation, uploaded documents, and chat chain."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    pdf_names: list[str] = field(default_factory=list)
    uploaded_files: list[dict[str, str]] = field(default_factory=list)
    processing_done: bool = False

    # Lazily built; rebuilt from the persisted index when missing. Not part of
    # the public/serialized shape, so kept out of repr.
    _chat_chain: Any | None = field(default=None, repr=False)
    _retriever: Any | None = field(default=None, repr=False)

    # -- Per-session storage paths -------------------------------------------
    @property
    def uploads_dir(self) -> Path:
        return settings.uploads_dir / self.session_id

    @property
    def vector_store_dir(self) -> Path:
        return settings.vector_stores_dir / self.session_id

    def has_documents(self) -> bool:
        """True when a FAISS index has been persisted for this session."""
        return self.vector_store_dir.exists() and any(self.vector_store_dir.iterdir())

    # -- Chain lifecycle ------------------------------------------------------
    def ensure_chain(self) -> Any:
        """Return this session's chat chain, building it on first use.

        Rebuilds from the persisted FAISS index when documents exist (the
        load-persisted-index path), otherwise creates a general no-PDF chain.
        """
        if self._chat_chain is not None:
            return self._chat_chain

        log.section("Ensure Chat Chain")
        log.kv("Existing Chat Chain", "NO")

        if self.has_documents():
            vector_store = load_vector_store(self.vector_store_dir)
            self._retriever = retriever_from_vector_store(vector_store)
            self._chat_chain = HybridChatChain(retriever=self._retriever)
            log.kv("Rebuilt From Persisted Index", "YES")
            log.kv("Indexed Chunks", indexed_chunks(self._retriever))
        else:
            self._chat_chain, self._retriever = create_chat_chain()
            log.kv("Created General Chat Chain", "YES")

        return self._chat_chain

    def process_documents(self, files: Iterable[tuple[str, bytes]]) -> dict[str, int]:
        """Persist uploaded PDFs, then (re)build the vector store + chat chain.

        ``files`` is an iterable of ``(original_filename, content_bytes)``.
        Returns simple counts for the API response. Resets the conversation on
        a fresh document set, matching the Streamlit app.
        """
        saved_files = self._save_uploads(files)
        if not saved_files:
            raise ValueError("No PDF files were provided.")

        pdf_paths = [file["path"] for file in saved_files]
        documents = load_pdfs(pdf_paths)
        documents = _attach_original_names(documents, saved_files)
        chunks = split_documents(documents)

        chat_chain, retriever = create_chat_chain(chunks)

        vector_store = vector_store_from_retriever(retriever)
        if vector_store is not None:
            save_vector_store(vector_store, self.vector_store_dir)

        self.messages = []
        self._chat_chain = chat_chain
        self._retriever = retriever
        self.uploaded_files = saved_files
        self.pdf_names = [file["original_name"] for file in saved_files]
        self.processing_done = True

        log.section("PDF Processing")
        log.kv("Document Count", len(documents))
        log.kv("Chunk Count", len(chunks))
        log.kv("Embedding Model", settings.embedding_model)
        log.kv("FAISS Index Size", indexed_chunks(retriever))
        log.kv("Vector Store Persisted", "YES" if vector_store is not None else "NO")

        return {"document_count": len(documents), "chunk_count": len(chunks)}

    def _save_uploads(
        self, files: Iterable[tuple[str, bytes]]
    ) -> list[dict[str, str]]:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict[str, str]] = []

        for original_name, content in files:
            suffix = Path(original_name).suffix or ".pdf"
            saved_name = f"{uuid.uuid4().hex}{suffix}"
            saved_path = self.uploads_dir / saved_name
            saved_path.write_bytes(content)
            saved.append(
                {
                    "original_name": original_name,
                    "saved_name": saved_name,
                    "path": str(saved_path),
                }
            )

        return saved

    # -- Conversation ---------------------------------------------------------
    def stream_turn(self, question: str) -> Iterator[dict[str, Any]]:
        """Stream one chat turn, yielding the chain's ``token``/``final`` events.

        Appends the user message, then — however the stream ends — records the
        assistant message with its resolved source, so server-side history
        stays consistent with what was streamed to the client.
        """
        chain = self.ensure_chain()

        self.add_message("user", question)
        payload = {"question": question, "chat_history": self._build_chat_history()}

        response_chunks: list[str] = []
        source: dict[str, Any] = {"type": "llm", "label": GENERAL_SOURCE_LABEL}
        try:
            for event in chain.stream(payload):
                event_type = event.get("type")
                if event_type == "token":
                    response_chunks.append(event.get("text", ""))
                elif event_type == "final":
                    source = _source_from_final(event)
                yield event
        finally:
            self.add_message("assistant", "".join(response_chunks), source=source)

    def add_message(
        self, role: str, content: str, source: dict[str, Any] | None = None
    ) -> None:
        message: dict[str, Any] = {"role": role, "content": content}
        if source is not None:
            message["source"] = source
        self.messages.append(message)

    def _build_chat_history(self) -> str:
        recent_messages = self.messages[-HISTORY_WINDOW_MESSAGES:]
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in recent_messages
        )

    def clear_chat(self) -> None:
        self.messages = []

    def cleanup_files(self) -> None:
        """Remove this session's uploaded PDFs and persisted FAISS index."""
        for path in (self.uploads_dir, self.vector_store_dir):
            shutil.rmtree(path, ignore_errors=True)

    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "pdf_loaded": bool(self.pdf_names),
            "chat_ready": True,
            "document_count": len(self.pdf_names),
            "pdf_names": list(self.pdf_names),
            "processing_done": self.processing_done,
            "message_count": len(self.messages),
        }


class SessionManager:
    """In-memory registry of :class:`ChatSession` objects, keyed by id."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = RLock()

    def create(self) -> ChatSession:
        session_id = uuid.uuid4().hex
        session = ChatSession(session_id=session_id)
        with self._lock:
            self._sessions[session_id] = session
        log.kv("Session Created", session_id)
        return session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.cleanup_files()
        log.kv("Session Deleted", session_id)
        return True


# Shared singleton, imported by the API layer (mirrors ``settings`` / ``log``).
session_manager = SessionManager()
