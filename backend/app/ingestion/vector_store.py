from pathlib import Path
from time import perf_counter

from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.ingestion.embeddings import get_embeddings
from app.utils.logger import log


def create_vector_store(chunks):
    started_at = perf_counter()
    log.kv("Embedding Model", settings.embedding_model)
    log.kv("Chunks To Embed", len(chunks))
    log.info("Embedding Started")
    vector_store = FAISS.from_documents(
        documents=chunks,  # putting chunks from textsplitter into documents parameter
        embedding=get_embeddings()  # shared HuggingFace embeddings, loaded lazily on first use
    )
    log.success("Embedding Finished")
    log.success("FAISS Created")
    log.kv("Vector Count", len(vector_store.index_to_docstore_id))
    elapsed_ms = (perf_counter() - started_at) * 1000
    log.kv("Build Time", f"{elapsed_ms:.2f} ms")

    return vector_store


def save_vector_store(vector_store, folder_path):
    """Persist a FAISS index to ``<folder_path>/index.faiss`` + ``index.pkl``."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(folder_path))
    log.success("FAISS Persisted")
    log.kv("Vector Store Path", str(folder_path))

    return folder_path


def load_vector_store(folder_path):
    """Load a FAISS index previously written by :func:`save_vector_store`."""
    vector_store = FAISS.load_local(
        str(folder_path),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # only ever loading indexes this app wrote
    )
    log.kv("Vector Store Loaded", str(folder_path))
    log.kv("Vector Count", len(vector_store.index_to_docstore_id))

    return vector_store
