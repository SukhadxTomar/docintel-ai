"""Shared FastAPI dependencies for the API routes."""
from __future__ import annotations

from fastapi import HTTPException, Path, status

from app.session.manager import ChatSession, session_manager


def get_session(
    session_id: str = Path(..., description="Chat session id"),
) -> ChatSession:
    """Resolve the ``ChatSession`` for a path id, or 404 if it does not exist."""
    session = session_manager.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return session
