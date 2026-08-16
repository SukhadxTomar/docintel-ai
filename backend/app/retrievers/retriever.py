from app.core.config import settings
from app.ingestion.vector_store import create_vector_store
from app.utils.logger import log


def _index_size(vector_store):
    index = getattr(vector_store, "index", None)
    if index is not None and hasattr(index, "ntotal"):
        return index.ntotal

    ids = getattr(vector_store, "index_to_docstore_id", None)
    if ids is not None:
        return len(ids)

    return "Unknown"


def retriever_from_vector_store(vector_store):
    """Build the standard MMR retriever from an existing (e.g. persisted) FAISS store.

    Shared by :func:`create_retriever` (build-from-chunks path) and the API's
    chat path (load-persisted-index path) so the MMR search config lives in one
    place.
    """
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,  # final number of chunks to return to llm
            "fetch_k": settings.retriever_fetch_k,  # candidates fetched before reranking
            "lambda_mult": settings.retriever_lambda_mult,  # MMR relevance vs diversity
        },
    )


def create_retriever(chunks):  # function takes chunks as input parameter
    vector_store = create_vector_store(chunks)
    log.section("Retriever Build Debug")
    log.kv("Vector Store Created", "YES" if vector_store is not None else "NO")
    log.kv("FAISS Index Size", _index_size(vector_store))
    # chunks from text_splitter.py are passed to create_vector_store()
    retriever = retriever_from_vector_store(vector_store)
    log.success("Retriever Created")
    log.kv("Retriever Created", "YES" if retriever is not None else "NO")
    log.kv("Retriever Type", type(retriever).__name__)
    log.kv("Search Type", "mmr")
    log.kv("k", settings.retriever_k)
    log.kv("fetch_k", settings.retriever_fetch_k)
    log.kv("lambda_mult", settings.retriever_lambda_mult)

    return retriever
