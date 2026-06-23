import streamlit as st
import os
import time
from ingestion.pdf_loader import load_pdfs
from ingestion.text_splitter import split_documents
from chains.chat_chain import create_chat_chain

# must be the first streamlit call
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #212121;
        color: #ececec;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2e2e2e;
    }
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }

    /* hide the default streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Chat layout ── */

    /* remove streamlit's default message bubble styling */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 4px 0 !important;
        gap: 12px !important;
    }

    /* hide streamlit's built-in avatar icons */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        display: none !important;
    }

    /* user messages: push everything to the right */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"].user-message {
        flex-direction: row-reverse !important;
    }

    /* user bubble */
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 6px 0;
    }
    .user-bubble .bubble-text {
        background: #2f2f2f;
        color: #ececec;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        max-width: 70%;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    /* assistant row: icon + text side by side */
    .assistant-row {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 6px 0;
    }
    .assistant-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #10a37f;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .assistant-text {
        color: #ececec;
        font-size: 0.95rem;
        line-height: 1.65;
        max-width: 80%;
        padding-top: 4px;
    }

    /* source citation cards */
    .source-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 10px;
        padding: 6px 14px;
        margin: 4px 0;
        font-size: 0.80rem;
        color: #a0a0a0;
        display: inline-block;
    }
    .sources-wrap {
        margin-left: 46px;  /* align with assistant text, past the avatar */
        margin-top: 6px;
    }

    /* welcome card */
    .welcome-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 18px;
        padding: 40px 48px;
        max-width: 560px;
        margin: 80px auto 0 auto;
        text-align: center;
    }
    .welcome-card h2 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ececec;
        margin-bottom: 10px;
    }
    .welcome-card p {
        font-size: 0.95rem;
        color: #888;
        line-height: 1.6;
    }
    .welcome-icon { font-size: 2.8rem; margin-bottom: 16px; }

    /* status pills */
    .status-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .pill-green { background: #1a3a2a; color: #4caf82; border: 1px solid #2a5a3a; }
    .pill-red   { background: #3a1a1a; color: #cf6679; border: 1px solid #5a2a2a; }
    .pill-grey  { background: #2a2a2a; color: #888;    border: 1px solid #3a3a3a; }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 8px 12px;
        transition: background 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: #10a37f;
        border: none;
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover { background: #0e8f6f; }

    [data-testid="stFileUploader"] {
        background: #1e1e1e;
        border: 1px dashed #3a3a3a;
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #2f2f2f !important;
        border: 1px solid #3a3a3a !important;
        border-radius: 14px !important;
        color: #ececec !important;
        font-size: 0.95rem;
    }

    [data-testid="stSpinner"] p { color: #888 !important; }

    details { background: #1e1e1e !important; border-radius: 10px; border: 1px solid #2e2e2e !important; }

    hr { border-color: #2e2e2e; margin: 12px 0; }

    [data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "messages": [],
        "chat_chain": None,
        "retriever": None,
        "pdf_names": [],
        "processing_done": False,
        "show_new_upload": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _get_sources(question: str) -> list[dict]:
    """Fetch relevant docs from FAISS and return a deduplicated list of sources."""
    if st.session_state.retriever is None:
        return []
    try:
        docs = st.session_state.retriever.get_relevant_documents(question)
        seen, sources = set(), []
        for d in docs:
            meta = d.metadata or {}
            fname = os.path.basename(meta.get("source", "Unknown"))
            page = meta.get("page", None)
            key = (fname, page)
            if key not in seen:
                seen.add(key)
                sources.append({"filename": fname, "page": page})
        return sources
    except Exception:
        return []


def _render_sources(sources: list[dict], inside_assistant_row: bool = False) -> None:
    """Render small citation cards under an answer."""
    if not sources:
        return
    wrap_class = "sources-wrap" if inside_assistant_row else ""
    cards_html = "".join(
        f'📄 {s["filename"]}'
        + (f' · Page {s["page"] + 1}' if s["page"] is not None else "")
        for s in sources
    )
    # build individual cards
    cards = ""
    for s in sources:
        page_label = f" · Page {s['page'] + 1}" if s["page"] is not None else ""
        cards += f'<div class="source-card">📄 {s["filename"]}{page_label}</div> '
    st.markdown(
        f'<div class="{wrap_class}" style="margin-top:8px;">{cards}</div>',
        unsafe_allow_html=True,
    )


def _render_user_message(content: str) -> None:
    """Render a right-aligned user bubble."""
    st.markdown(
        f'<div class="user-bubble"><div class="bubble-text">{content}</div></div>',
        unsafe_allow_html=True,
    )


def _render_assistant_message(content: str, sources: list[dict] | None = None) -> None:
    """Render assistant message with the green avatar on the left."""
    import re
    # basic markdown → html for bold/italic so it renders nicely inside our div
    # (st.markdown handles full markdown; we use write_stream for new messages)
    st.markdown(
        f"""
        <div class="assistant-row">
          <div class="assistant-avatar">✦</div>
          <div class="assistant-text">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if sources:
        _render_sources(sources, inside_assistant_row=True)


def _process_files(uploaded_files) -> None:
    """Save files to disk, run the RAG pipeline, and update session state."""
    pdf_paths, names = [], []
    for uf in uploaded_files:
        path = os.path.join(UPLOAD_FOLDER, uf.name)
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        pdf_paths.append(path)
        names.append(uf.name)

    documents = load_pdfs(pdf_paths)
    chunks = split_documents(documents)

    (
        st.session_state.chat_chain,
        st.session_state.retriever,
    ) = create_chat_chain(chunks)

    st.session_state.pdf_names = names
    st.session_state.processing_done = True
    st.session_state.show_new_upload = False
    st.session_state.messages = []  # clear chat when switching to a new set of docs


# sidebar 

with st.sidebar:
    st.markdown("## 📄 PDF Chatbot")
    st.markdown("---")

    st.markdown("### Status")
    pdf_count = len(st.session_state.pdf_names)
    chat_ready = st.session_state.chat_chain is not None

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("PDFs loaded")
    with col_b:
        if pdf_count:
            st.markdown(
                f'<span class="status-pill pill-green">✔ {pdf_count}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill pill-grey">None</span>',
                unsafe_allow_html=True,
            )

    col_c, col_d = st.columns([3, 2])
    with col_c:
        st.markdown("Chat ready")
    with col_d:
        if chat_ready:
            st.markdown(
                '<span class="status-pill pill-green">✔ Yes</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill pill-red">✗ No</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # show uploader on first visit or when the user explicitly asks for it
    show_uploader = (not st.session_state.processing_done) or st.session_state.show_new_upload

    if show_uploader:
        st.markdown("### Upload PDFs")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
            for uf in uploaded_files:
                st.markdown(f"&nbsp;&nbsp;📎 `{uf.name}`", unsafe_allow_html=True)

            if st.button("⚡ Process PDFs", type="primary", key="btn_process"):
                with st.spinner("Embedding & indexing PDFs…"):
                    _process_files(uploaded_files)
                st.success("PDFs processed! Start chatting below.")
                st.rerun()
        else:
            st.caption("Drag & drop or click to browse files.")

    else:
        st.markdown("### Loaded PDFs")
        for name in st.session_state.pdf_names:
            st.markdown(f"📄 `{name}`")

    st.markdown("---")

    st.markdown("### Actions")

    if st.session_state.processing_done and not st.session_state.show_new_upload:
        if st.button("📂 Upload New PDFs", key="btn_new_upload"):
            st.session_state.show_new_upload = True
            st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History", key="btn_clear_chat"):
            st.session_state.messages = []
            st.rerun()

    if st.button("🔄 Reset Session", key="btn_reset"):
        for key in ["messages", "chat_chain", "retriever", "pdf_names",
                    "processing_done", "show_new_upload"]:
            del st.session_state[key]
        st.rerun()

    st.markdown("---")

    with st.expander("ℹ️ About this app"):
        st.markdown(
            """
            **PDF Chatbot** lets you chat with your own documents.

            **Stack:**
            - 🦜 LangChain · conversational memory
            - 🤖 Gemini API · generation
            - 🤗 HuggingFace · embeddings
            - 🗄️ FAISS · vector search
            - 🎈 Streamlit · UI

            **How to use:**
            1. Upload one or more PDFs in the sidebar.
            2. Press **Process PDFs**.
            3. Ask anything about your documents!
            """
        )


# ---------- main chat area ----------

if not st.session_state.messages:
    if not st.session_state.processing_done:
        st.markdown(
            """
            <div class="welcome-card">
              <div class="welcome-icon">📄</div>
              <h2>Chat with your PDFs</h2>
              <p>Upload one or more PDF files in the sidebar, then press
              <strong>Process PDFs</strong> to get started.<br><br>
              You can ask questions, request summaries, or dig into the details
              of any document.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="welcome-card">
              <div class="welcome-icon">💬</div>
              <h2>Ready to chat!</h2>
              <p>Your PDFs have been indexed. Type a question below to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    # replay history using our custom bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_message(msg["content"])
        else:
            _render_assistant_message(msg["content"], msg.get("sources"))


# chat input
question = st.chat_input(
    "Ask a question about your PDFs…" if st.session_state.processing_done
    else "Upload and process PDFs first…",
)

if question:
    if st.session_state.chat_chain is None:
        st.warning("⚠️ Please upload and process your PDFs first using the sidebar.")
        st.stop()

    # show the user bubble right away
    _render_user_message(question)
    st.session_state.messages.append({"role": "user", "content": question})

    chat_history = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages
    )

    # call the chain (this is where Gemini does the heavy lifting)
    with st.spinner(""):
        response = st.session_state.chat_chain.invoke(
            {"question": question, "chat_history": chat_history}
        )

    # simulate streaming: reveal the response word by word
    avatar_html = '<div class="assistant-row"><div class="assistant-avatar">✦</div><div class="assistant-text">'
    stream_placeholder = st.empty()
    displayed = ""
    words = response.split(" ")
    for i, word in enumerate(words):
        displayed += ("" if i == 0 else " ") + word
        stream_placeholder.markdown(
            avatar_html + displayed + "▌</div></div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.018)

    # final render without the cursor, then add sources
    stream_placeholder.markdown(
        avatar_html + displayed + "</div></div>",
        unsafe_allow_html=True,
    )

    sources = _get_sources(question)
    _render_sources(sources, inside_assistant_row=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )