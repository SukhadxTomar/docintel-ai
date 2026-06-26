from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

import streamlit as st

from chains.chat_chain import create_chat_chain
from ingestion.pdf_loader import load_pdfs
from ingestion.text_splitter import split_documents


UPLOAD_DIR = Path("uploaded_pdfs")
HISTORY_WINDOW_MESSAGES = 12


def configure_page() -> None:
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="PDF",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def init_session_state() -> None:
    defaults = {
        "messages": [],
        "chat_chain": None,
        "retriever": None,
        "processing_done": False,
        "uploaded_files": [],
        "pdf_names": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_session() -> None:
    st.session_state.messages = []
    st.session_state.chat_chain = None
    st.session_state.retriever = None
    st.session_state.processing_done = False
    st.session_state.uploaded_files = []
    st.session_state.pdf_names = []


def clear_chat() -> None:
    st.session_state.messages = []


def ensure_chat_chain() -> bool:
    if st.session_state.chat_chain is not None:
        return True

    try:
        chat_chain, retriever = create_chat_chain()
    except Exception as exc:
        st.error(f"Could not initialize the chat model: {exc}")
        return False

    st.session_state.chat_chain = chat_chain
    st.session_state.retriever = retriever
    return True


def save_uploaded_pdfs(uploaded_files: Iterable[Any]) -> list[dict[str, str]]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        original_name = uploaded_file.name
        suffix = Path(original_name).suffix or ".pdf"
        saved_name = f"{uuid.uuid4().hex}{suffix}"
        saved_path = UPLOAD_DIR / saved_name

        saved_path.write_bytes(uploaded_file.getbuffer())

        saved_files.append(
            {
                "original_name": original_name,
                "saved_name": saved_name,
                "path": str(saved_path),
            }
        )

    return saved_files


def attach_original_names(documents: list[Any], saved_files: list[dict[str, str]]) -> list[Any]:
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


def process_pdfs(uploaded_files: Iterable[Any]) -> bool:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        return False

    try:
        saved_files = save_uploaded_pdfs(uploaded_files)
        pdf_paths = [file["path"] for file in saved_files]

        documents = load_pdfs(pdf_paths)
        documents = attach_original_names(documents, saved_files)
        chunks = split_documents(documents)
        chat_chain, retriever = create_chat_chain(chunks)

    except Exception as exc:
        st.error(f"PDF processing failed: {exc}")
        return False

    st.session_state.messages = []
    st.session_state.chat_chain = chat_chain
    st.session_state.retriever = retriever
    st.session_state.processing_done = True
    st.session_state.uploaded_files = saved_files
    st.session_state.pdf_names = [file["original_name"] for file in saved_files]

    return True


def build_chat_history() -> str:
    recent_messages = st.session_state.messages[-HISTORY_WINDOW_MESSAGES:]
    return "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_messages
    )


def stream_response(question: str):
    chain = st.session_state.chat_chain

    if chain is None:
        raise RuntimeError("Chat chain is not ready.")

    payload = {
        "question": question,
        "chat_history": build_chat_history(),
    }

    yield from chain.stream(payload)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("PDF Section")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type="pdf",
            accept_multiple_files=True,
        )

        if st.button("Process PDFs", type="primary", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                if process_pdfs(uploaded_files):
                    st.success("PDFs processed successfully.")

        if st.session_state.pdf_names:
            st.subheader("Uploaded PDFs")
            for pdf_name in st.session_state.pdf_names:
                st.write(f"PDF: {pdf_name}")

        st.divider()

        st.header("Chat Section")

        st.button(
            "Clear Chat",
            use_container_width=True,
            disabled=not bool(st.session_state.messages),
            on_click=clear_chat,
        )

        st.button(
            "Reset Session",
            use_container_width=True,
            on_click=reset_session,
        )

        st.divider()

        st.header("Status Section")

        pdf_loaded = bool(st.session_state.pdf_names)
        chat_ready = True
        document_count = len(st.session_state.pdf_names)

        st.write(f"PDF Loaded: {'Yes' if pdf_loaded else 'No'}")
        st.write(f"Chat Ready: {'Yes' if chat_ready else 'No'}")
        st.write(f"Number of Documents: {document_count}")


def render_header() -> None:
    st.title("PDF Chatbot")

    if st.session_state.processing_done:
        st.caption("Ask anything. Document questions are routed to your PDFs automatically.")
    else:
        st.caption("Ask a general question, or upload PDFs to begin document chat.")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_chat_input() -> None:
    prompt = st.chat_input("Ask a question")

    if not prompt:
        return

    if not ensure_chat_chain():
        return

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            response = st.write_stream(stream_response(prompt))

    except Exception as exc:
        response = f"Sorry, I could not generate a response: {exc}"
        st.error(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response if isinstance(response, str) else str(response),
        }
    )


def main() -> None:
    configure_page()
    init_session_state()

    render_sidebar()
    render_header()
    render_chat_history()
    handle_chat_input()


if __name__ == "__main__":
    main()
