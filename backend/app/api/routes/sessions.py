"""Session lifecycle endpoints: create, status, history, clear, delete."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import get_session
from app.api.schemas import (
    MessageOut,
    MessagesResponse,
    SessionCreatedResponse,
    StatusResponse,
)
from app.session.manager import ChatSession, session_manager

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "",
    response_model=SessionCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session() -> SessionCreatedResponse:
    session = session_manager.create()
    return SessionCreatedResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=StatusResponse)
def get_status(session: ChatSession = Depends(get_session)) -> StatusResponse:
    return StatusResponse(**session.status())


@router.get("/{session_id}/messages", response_model=MessagesResponse)
def get_messages(session: ChatSession = Depends(get_session)) -> MessagesResponse:
    return MessagesResponse(
        session_id=session.session_id,
        messages=[MessageOut(**message) for message in session.messages],
    )


@router.post("/{session_id}/clear", response_model=StatusResponse)
def clear_chat(session: ChatSession = Depends(get_session)) -> StatusResponse:
    session.clear_chat()
    return StatusResponse(**session.status())


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session: ChatSession = Depends(get_session)) -> None:
    session_manager.delete(session.session_id)
