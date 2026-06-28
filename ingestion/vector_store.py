from time import perf_counter

from langchain_community.vectorstores import FAISS

from ingestion.embeddings import embeddings
from utils.logger import log


def create_vector_store(chunks):
    started_at = perf_counter()
    log.kv("Embedding Model", "BAAI/bge-small-en-v1.5")
    log.kv("Chunks To Embed", len(chunks))
    log.info("Embedding Started")
    vector_store = FAISS.from_documents(
        documents=chunks,  # putting chunks from textsplitter into documents parameter
        embedding=embeddings  # putting embeddings from embeddings.py into embedding parameter
    )
    log.success("Embedding Finished")
    log.success("FAISS Created")
    log.kv("Vector Count", len(vector_store.index_to_docstore_id))
    elapsed_ms = (perf_counter() - started_at) * 1000
    log.kv("Build Time", f"{elapsed_ms:.2f} ms")

    return vector_store