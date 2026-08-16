from pathlib import Path
from time import perf_counter

from langchain_community.document_loaders import PyPDFLoader

from utils.logger import log


def load_pdfs(pdf_paths):
    documents = []
    total_pages = 0
    started_at = perf_counter()

    for pdf in pdf_paths:  # for multiple pdf
        file_path = Path(pdf)
        loader = PyPDFLoader(pdf)
        started_pdf_at = perf_counter()
        docs = loader.load()  # actual pdf read and stored in docs list
        load_time_ms = (perf_counter() - started_pdf_at) * 1000

        documents.extend(docs)  # put pages from docs to documents list
        total_pages += len(docs)
        log.kv("filename", file_path.name)
        log.kv("size", file_path.stat().st_size if file_path.exists() else 0)
        log.kv("pages", len(docs))
        log.kv("load time", f"{load_time_ms:.2f} ms")

    total_time_ms = (perf_counter() - started_at) * 1000
    log.kv("Total PDFs", len(pdf_paths))
    log.kv("Total Pages", total_pages)
    log.kv("Total Time", f"{total_time_ms:.2f} ms")

    return documents