from time import perf_counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.utils.logger import log


def split_documents(documents):
    started_at = perf_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    lengths = [len(chunk.page_content) for chunk in chunks]
    average_length = sum(lengths) / len(lengths) if lengths else 0
    minimum_length = min(lengths) if lengths else 0
    maximum_length = max(lengths) if lengths else 0
    elapsed_ms = (perf_counter() - started_at) * 1000

    log.kv("Chunk Size", settings.chunk_size)
    log.kv("Chunk Overlap", settings.chunk_overlap)
    log.kv("Chunk Count", len(chunks))
    log.kv("Average Length", f"{average_length:.2f}")
    log.kv("Minimum Length", minimum_length)
    log.kv("Maximum Length", maximum_length)
    log.kv("Time Taken", f"{elapsed_ms:.2f} ms")

    return chunks