from langchain_community.vectorstores import FAISS
from ingestion.embeddings import embeddings


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(
        documents=chunks,#putting chunks from textsplitter into documents  parameter
        embedding=embeddings#putting embeddings from embeddings.py into embedding parameter
    )

    return vector_store