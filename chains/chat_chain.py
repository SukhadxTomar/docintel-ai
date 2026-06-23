from models.llm_model import load_llm
from prompts.chatbot_prompt import chat_prompt
from retrievers.retriever import create_retriever
from langchain_core.output_parsers import StrOutputParser


def create_chat_chain(chunks):

    retriever = create_retriever(chunks)
    llm = load_llm()
    parser = StrOutputParser()

    def format_docs(docs):

        formatted_docs = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            formatted_docs.append(
                f"Source: {source}, Page: {page + 1}\n{doc.page_content}"
            )

        return "\n\n".join(formatted_docs)

    chain = (
        {
            "context": lambda x: format_docs(
                retriever.invoke(x["question"])
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | chat_prompt
        | llm
        | parser
    )

    return chain, retriever