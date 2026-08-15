from time import perf_counter

from langchain_community.vectorstores import FAISS

from config import settings
from ingestion.embeddings import get_embeddings
from utils.logger import log


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