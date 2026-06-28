from ingestion.vector_store import create_vector_store
from utils.logger import log


def create_retriever(chunks):  # function takes chunks as input parameter
    vector_store = create_vector_store(chunks)
    # chunks from text_splitter.py are passed to create_vector_store()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,  # final number of chunks to return to llm
            "fetch_k": 10,  # number of candidates fetched before reranking
            "lambda_mult": 0.7  # balance between relevance and diversity in MMR (0.7 means more emphasis on relevance)
        }
    )
    log.success("Retriever Created")
    log.kv("Search Type", "mmr")
    log.kv("k", 4)
    log.kv("fetch_k", 10)
    log.kv("lambda_mult", 0.7)

    return retriever

