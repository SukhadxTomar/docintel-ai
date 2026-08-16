import type { SessionStatus } from '../types'
import './ChatHeader.css'

export function ChatHeader({ status }: { status: SessionStatus | null }) {
  const pdfLoaded = !!status?.pdf_loaded
  const title = pdfLoaded ? 'Document chat' : 'General chat'
  const caption = pdfLoaded
    ? 'Grounded in your uploaded PDFs, with automatic fallback to general knowledge.'
    : 'Answering from general knowledge — upload PDFs in the sidebar to ground answers in your documents.'

  return (
    <header className="chat-header">
      <div className="chat-header__inner">
        <h1 className="chat-header__title">{title}</h1>
        <p className="chat-header__caption">{caption}</p>
      </div>
    </header>
  )
}
