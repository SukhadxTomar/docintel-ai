"""PDF upload + processing endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import get_session
from app.api.schemas import ProcessResponse
from app.session.manager import ChatSession
from app.utils.logger import log

router = APIRouter(prefix="/sessions", tags=["documents"])


def _is_pdf(upload: UploadFile) -> bool:
    if (upload.filename or "").lower().endswith(".pdf"):
        return True
    return upload.content_type in ("application/pdf", "application/octet-stream")


@router.post("/{session_id}/documents", response_model=ProcessResponse)
async def process_documents(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    session: ChatSession = Depends(get_session),
) -> ProcessResponse:
    pdfs: list[tuple[str, bytes]] = []
    for upload in files:
        if not _is_pdf(upload):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{upload.filename}' is not a PDF.",
            )
        pdfs.append((upload.filename or "document.pdf", await upload.read()))

    if not pdfs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload at least one PDF.",
        )

    # Loading + embedding is blocking and CPU-heavy: run it off the event loop.
    def _process() -> dict[str, int]:
        with log.request_context():
            return session.process_documents(pdfs)

    try:
        stats = await run_in_threadpool(_process)
    except Exception as exc:  # surface a clean 500 instead of a raw traceback
        log.error(f"PDF processing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF processing failed: {exc}",
        )

    return ProcessResponse(
        session_id=session.session_id,
        pdf_names=session.pdf_names,
        document_count=stats["document_count"],
        chunk_count=stats["chunk_count"],
        processing_done=session.processing_done,
    )
