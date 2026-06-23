from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_paths):
    documents = []

    for pdf in pdf_paths: #for multiple pdf
        loader = PyPDFLoader(pdf)
        docs = loader.load() #actual pdf read and stored in docs list

        documents.extend(docs)#put pages from docs to documents list

    return documents