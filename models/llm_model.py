from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from config import settings

load_dotenv()

_llm = None  # cached, shared Gemini client (created on first load_llm() call)


def load_llm():
    """Return a shared Gemini chat client, created once and reused."""
    global _llm

    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            streaming=True,
        )

    return _llm
