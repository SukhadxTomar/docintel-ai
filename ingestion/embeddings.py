from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import log

log.info("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
log.success("Embedding Model Loaded")
log.kv("Embedding Model Name", "BAAI/bge-small-en-v1.5")
log.success("Load Success")
