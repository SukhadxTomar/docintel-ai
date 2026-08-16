from config import settings
from ingestion.vector_store import create_vector_store
from utils.logger import log


def _index_size(vector_store):
    index = getattr(vector_store, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return index.ntotal

    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        return len(ids)

    return "Unknown"


def create_retriever(chunks):  # function takes chunks as input parameter
    vector_store = create_vector_store(chunks)
    log.section("Retriever Build Debug")
    log.kv("Vector Store Created", "YES" if vector_store is not None else "NO")
    log.kv("FAISS Index Size", _index_size(vector_store))
    # chunks from text_splitter.py are passed to create_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,  # final number of chunks to return to llm
            "fetch_k": settings.retriever_fetch_k,  # number of candidates fetched before reranking
            "lambda_mult": settings.retriever_lambda_mult  # balance between relevance and diversity in MMR (0.7 means more emphasis on relevance)
        }
    )
    log.success("Retriever Created")
    log.kv("Retriever Created", "YES" if retriever is not None else "NO")
    log.kv("Retriever Type", type(retriever).__name__)
    log.kv("Search Type", "mmr")
    log.kv("k", settings.retriever_k)
    log.kv("fetch_k", settings.retriever_fetch_k)
    log.kv("lambda_mult", settings.retriever_lambda_mult)

    return retriever

