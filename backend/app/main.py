"""DocIntel-AI FastAPI application entry point.

Run with::

    cd backend && uvicorn app.main:app --reload

Storage directories and CORS origins come from ``app.core.config.settings``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, sessions
from app.core.config import settings
from app.utils.logger import log


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_stores_dir.mkdir(parents=True, exist_ok=True)
    log.section("DocIntel API Startup")
    log.kv("Uploads Dir", str(settings.uploads_dir))
    log.kv("Vector Stores Dir", str(settings.vector_stores_dir))
    log.kv("CORS Origins", ", ".join(settings.cors_origins))
    log.success("API Ready")
    yield


app = FastAPI(title="DocIntel-AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/api/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
