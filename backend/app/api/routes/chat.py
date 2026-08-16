"""Streaming chat endpoint (Server-Sent Events).

Each turn is streamed as SSE ``data:`` lines carrying JSON events:
  - ``{"type": "token", "text": <str>}``     one per generated text delta
  - ``{"type": "sources", "mode": "rag"|"llm", "sources": [...], "label"?: <str>}``
  - ``{"type": "error", "message": <str>}``  if generation failed
  - ``{"type": "done", "request_id": <str>}``  always emitted last

This is the API-side translation of the chain's own ``token``/``final`` events
(see ``chains/chat_chain.py``); the routing decision travels in-band via the
``sources`` event rather than through shared chain state.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_session
from app.api.schemas import ChatRequest
from app.session.manager import GENERAL_SOURCE_LABEL, ChatSession
from app.utils.logger import log

router = APIRouter(prefix="/sessions", tags=["chat"])


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _event_stream(session: ChatSession, question: str) -> Iterator[str]:
    with log.request_context() as request_id:
        try:
            for event in session.stream_turn(question):
                event_type = event.get("type")
                if event_type == "token":
                    yield _sse({"type": "token", "text": event.get("text", "")})
                elif event_type == "final":
                    if event.get("mode") == "rag":
                        yield _sse(
                            {
                                "type": "sources",
                                "mode": "rag",
                                "sources": event.get("sources", []),
                            }
                        )
                    else:
                        yield _sse(
                            {
                                "type": "sources",
                                "mode": "llm",
                                "sources": [],
                                "label": GENERAL_SOURCE_LABEL,
                            }
                        )
        except Exception as exc:
            log.error(f"Chat streaming failed: {exc}")
            yield _sse(
                {
                    "type": "error",
                    "message": f"Sorry, I could not generate a response: {exc}",
                }
            )
        finally:
            yield _sse({"type": "done", "request_id": request_id})


@router.post("/{session_id}/chat")
def chat(
    request: ChatRequest,
    session: ChatSession = Depends(get_session),
) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(session, request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush
        },
    )
