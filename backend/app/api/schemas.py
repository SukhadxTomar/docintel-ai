"""Pydantic request/response models for the DocIntel-AI API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionCreatedResponse(BaseModel):
    session_id: str


class StatusResponse(BaseModel):
    session_id: str
    pdf_loaded: bool
    chat_ready: bool
    document_count: int
    pdf_names: list[str]
    processing_done: bool
    message_count: int


class ProcessResponse(BaseModel):
    session_id: str
    pdf_names: list[str]
    document_count: int
    chunk_count: int
    processing_done: bool


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class MessageOut(BaseModel):
    role: str
    content: str
    source: dict[str, Any] | None = None


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]
